"""Memory-footprint and FLOPs analysis for the gaze pipeline on a single ev_eye subject.

Kept SEPARATE from benchmark_timing.py on purpose: the instrumentation here
(tracemalloc tracing, operation counters, RSS sampling) adds allocation and
bookkeeping overhead that would poison the latency numbers if mixed in. Run the
two scripts independently and report their results side by side.

What it reports, on the REAL pipeline code paths:
  1. frame  : FLOPs + working-set memory to extract one pupil from an APS image
  2. events : FLOPs + memory of the template-tracking ICP fit, per emitted pupil
  3. regress: FLOPs to map one pupil center -> Point of Gaze
  4. system : total FLOPs and peak resident memory for the whole run
  5. energy : first-order energy estimate, FLOPs scaled by a per-FLOP pJ constant
  6. table  : thesis Table 5.1 metrics (Parameters + GFLOPs per inference) for direct
              comparison against EV-Eye / FACET / TennSt / PupilUNet

FLOPs are an analytical cost model evaluated over MEASURED operation counts:
the script runs the actual pipeline, counts how many times each primitive runs
(pixels per image, ellipse fits, ICP batches, ICP iterations, candidate points,
polynomial features, predictions), then multiplies by a documented per-element
FLOP cost. Absolute FLOPs for OpenCV/numpy/scipy kernels are not directly
observable, so these are model estimates with the assumptions stated inline;
the loop *counts* they are multiplied by are exact (measured at run time).

Run from the repo root:
    source .venv/bin/activate
    python experiments_and_plots/benchmark_memory_flops.py --subject 22 --eye left --motion saccadic --degree 2
"""
import os
import sys
import math
import argparse
import resource
import tracemalloc

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import cv2  # noqa: E402
from config import (DATASET_PATHS, get_frame_detection_config, get_gaze_config,  # noqa: E402
                    TemplateTrackingConfig, LSTMConfig)
from data.loaders import EvEyeDataset  # noqa: E402
from processing.normalization import compute_pupil_stats, normalize_pupils  # noqa: E402
from models.polynomial import GazeEstimator  # noqa: E402
import processing.frame_detection as fd  # noqa: E402
import pipeline.pipeline as pl  # noqa: E402
import tracking  # noqa: E402


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def fmt_flops(x):
    for u, s in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if x >= u:
            return f"{x / u:8.3f} {s}FLOP"
    return f"{x:8.1f}  FLOP"


def fmt_bytes(x):
    for u, s in ((1 << 30, "GB"), (1 << 20, "MB"), (1 << 10, "KB")):
        if x >= u:
            return f"{x / u:8.3f} {s}"
    return f"{x:8.0f}  B"


def fmt_gflops(x):
    """FLOPs -> GFLOPs string, matching the thesis Table 5.1 'GFLOPs' column."""
    return f"{x / 1e9:.6f} GFLOP"


def fmt_params(n):
    """Param count -> 'N (X.XXX M)', matching Table 5.1 'Parameters' column."""
    return f"{n} ({n / 1e6:.6f} M)"


def fmt_energy(j):
    for u, s in ((1.0, "J"), (1e-3, "mJ"), (1e-6, "uJ"), (1e-9, "nJ")):
        if j >= u:
            return f"{j / u:8.3f} {s}"
    return f"{j * 1e12:8.3f} pJ"


_PAGE = os.sysconf("SC_PAGE_SIZE")


def rss_bytes():
    """Current resident set size of this process (Linux /proc)."""
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * _PAGE


def maxrss_bytes():
    """Peak resident set size so far (ru_maxrss is in KiB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def sizeof_list(lst, sample=2000):
    """Shallow size of a python list: container + element references, with the
    per-element cost estimated from a sample (lists here can be ~millions long)."""
    base = sys.getsizeof(lst)
    if not lst:
        return base
    s = lst[:sample]
    return base + (sum(sys.getsizeof(x) for x in s) / len(s)) * len(lst)


# --------------------------------------------------------------------------- #
# LSTM gaze-head cost model (analytical, no TF import needed)
#
# The head is LSTMGazeEstimator (models/lstm.py): one LSTM layer -> dense stack
# (ReLU) -> linear 2-D output, consuming a length-seq_len window of 21-D ellipse
# encodings. Counts follow the SAME convention as the detector/regressor models
# above -- each multiply and add is a separate FLOP (e.g. grayscale = 3 mul + 2
# add = 5 FLOPs/px), so one matmul MAC = 2 FLOPs. This keeps the "our pipeline"
# GFLOPs on one footing instead of mixing MAC-counts with mul+add counts.
# --------------------------------------------------------------------------- #
LSTM_INPUT_DIM = 21   # ellipse_to_21d encoding fed to the LSTM (models/lstm.py)


def lstm_param_count(cfg: LSTMConfig, input_dim=LSTM_INPUT_DIM, out_dim=2):
    """Exact learnable-weight count of LSTMGazeEstimator (matches Keras summary)."""
    U = cfg.lstm_units
    p = 4 * (U * (input_dim + U) + U)          # LSTM: kernel + recurrent + bias
    prev = U
    for u in cfg.dense_units:
        p += prev * u + u                       # dense weights + bias
        prev = u
    p += prev * out_dim + out_dim               # linear output head
    return int(p)


def lstm_forward_flops(cfg: LSTMConfig, input_dim=LSTM_INPUT_DIM, out_dim=2):
    """Forward-pass cost for one seq_len window -> one gaze point.

    Returns (flops, macs):
      flops -- every multiply and add counted separately (matches the
               detector/regressor convention; this is the internally consistent
               number for the "our pipeline" GFLOPs column).
      macs  -- pure matmul multiply-accumulates only (1 MAC = 1 mul+add pair,
               no bias/activation); reconciles with MAC-reporting NN profilers.
    """
    U, Fin = cfg.lstm_units, input_dim
    # --- one LSTM cell, per timestep ---
    gate_macs   = 4 * (U * (Fin + U))           # 4 gates: W_x@x + W_h@h
    gate_bias   = 4 * U                          # 4 gate biases
    gate_act    = 4 * U                          # sigmoid x3 (i,f,o) + tanh x1 (g)
    cell_update = 3 * U                          # c = f*c_prev + i*g (2 mul + 1 add)
    hidden_out  = 2 * U                          # h = o * tanh(c) (tanh + mul)
    # --- dense head (ReLU) + linear output ---
    dense_macs = 0
    dense_extra = 0
    prev = U
    for u in cfg.dense_units:
        dense_macs += prev * u
        dense_extra += u + u                     # bias + ReLU
        prev = u
    dense_macs += prev * out_dim
    dense_extra += out_dim                       # output bias, no activation

    macs = cfg.seq_len * gate_macs + dense_macs
    non_mac = cfg.seq_len * (gate_bias + gate_act + cell_update + hidden_out) + dense_extra
    flops = 2 * macs + non_mac                   # each MAC = 1 mul + 1 add
    return int(flops), int(macs)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=22)
    ap.add_argument("--eye", default="left", choices=["left", "right"])
    ap.add_argument("--motion", default="saccadic", choices=["saccadic", "pursuit"])
    ap.add_argument("--degree", type=int, default=2,
                    help="polynomial degree to analyze the regressor at")
    ap.add_argument("--energy_pj_per_flop", type=float, default=1.0,
                    help="energy cost per FLOP in picojoules, used for the first-order "
                         "energy estimate (modern hardware ~0.1-10 pJ/FLOP)")
    opt = ap.parse_args()

    data_dir = DATASET_PATHS["ev_eye"]
    frame_config = get_frame_detection_config(opt.subject, opt.eye, "ev_eye")
    gaze_config = get_gaze_config(opt.subject)

    print(f"=== Memory/FLOPs: ev_eye subject {opt.subject}, {opt.eye} eye, {opt.motion} ===\n")

    ds = EvEyeDataset(data_dir, opt.subject, motion=opt.motion, mode="np")
    ds.collect_data(eye=opt.eye)
    frame_list = ds.frame_list
    n_frames = len(frame_list) - 1                # index 0 is invalid by convention

    # image geometry (pixels per frame) from one decoded APS image
    sample_img = cv2.imread(frame_list[1].img)
    H, W = sample_img.shape[:2]
    chans = 1 if sample_img.ndim == 2 else sample_img.shape[2]
    P = H * W
    del sample_img

    # morphological-open structuring element: count active (non-zero) elements exactly
    k = 2 * frame_config.morph_kernel_size + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    se_nnz = int(np.count_nonzero(se))

    # ===================================================================== #
    # 1. Frame pupil extraction — count primitive ops while running the
    #    real pupil_extraction_stage; measure its Python-allocation peak.
    # ===================================================================== #
    fit_calls = [0]
    fit_pts = [0]
    _orig_fit = cv2.fitEllipse

    def _wrap_fit(pts, *a, **kw):
        fit_calls[0] += 1
        fit_pts[0] += len(pts)
        return _orig_fit(pts, *a, **kw)

    cv2.fitEllipse = _wrap_fit
    tracemalloc.start()
    tracemalloc.reset_peak()
    rss0 = rss_bytes()
    pupil_centers, ellipses, screen_coords = pl.pupil_extraction_stage(ds, frame_config)
    _, frame_py_peak = tracemalloc.get_traced_memory()
    rss_after_frame = rss_bytes()
    tracemalloc.stop()
    cv2.fitEllipse = _orig_fit

    # ---- frame-detection FLOP model (per frame, image of P = H*W pixels) ----
    # grayscale BGR->GRAY : weighted sum 0.299R+0.587G+0.114B = 3 mul + 2 add = 5/px
    # threshold (binary)  : 1 compare/px
    # morph open          : erode + dilate; each output px = (nnz-1) min/max compares
    # findContours        : border following, ~2 ops/px (rough O(P) traversal)
    # fitEllipse(n pts)   : conic least squares, design matrix + 6x6 normal eqns
    #                       ~= 36*n + ~200/contour (measured n via wrapper)
    fl_gray = (5 * P if chans == 3 else 0) * n_frames
    fl_thresh = 1 * P * n_frames
    fl_morph = 2 * P * (se_nnz - 1) * n_frames
    fl_contour = 2 * P * n_frames
    fl_fit = 36 * fit_pts[0] + 200 * fit_calls[0]
    frame_flops_total = fl_gray + fl_thresh + fl_morph + fl_contour + fl_fit
    frame_flops_per = frame_flops_total / n_frames

    # Strict floating-point subset, for an apples-to-apples comparison with the paper's
    # GFLOPs. NN profilers (the basis of the paper's 0.553 GFLOPs) count only the float
    # multiply/adds of conv/linear layers. In this classical detector, threshold +
    # morphology + contour are INTEGER comparisons on uint8 -> not floating-point ops at
    # all. Only grayscale (weighted sum) and the conic least-squares ellipse fit are
    # genuine floating-point arithmetic, so this is what is comparable to a NN's FLOPs.
    frame_float_total = fl_gray + fl_fit
    frame_float_per = frame_float_total / n_frames

    # frame-detection transient working set: img + gray + binary + opened +
    # contour_img held simultaneously inside _run_detection (uint8, BGR img = chans B/px)
    frame_ws_bytes = P * (chans + 1 + 1 + 1 + 1)

    # ===================================================================== #
    # Build valid mask (drives event extraction + regressor), as in the pipeline
    # ===================================================================== #
    blink_mask = pl.noise_flagging_stage(pupil_centers)
    valid_mask = pl.build_valid_mask(
        blink_mask, screen_coords,
        skip_frames=gaze_config.saccade_skip_frames,
        skip_label_changes=False,
        post_blink_skip_frames=gaze_config.post_blink_skip_frames,
        alignment_gaps=ds.alignment_gaps,
        max_alignment_gap_us=gaze_config.max_alignment_gap_us,
    )

    # ===================================================================== #
    # 2. Event pupil extraction — count ICP batches, candidate points, and
    #    KDTree.query calls (= iterations + 1 per batch). tracemalloc is OFF
    #    here: the 13M-event Python loop would make per-allocation tracing
    #    pathologically slow; RSS delta + artifact .nbytes cover this stage.
    # ===================================================================== #
    icp_calls = [0]
    cand_total = [0]
    M_boundary = [0]
    _orig_icp = pl.points_to_edge_matching

    def _wrap_icp(cand, boundary, *a, **kw):
        icp_calls[0] += 1
        cand_total[0] += len(cand)
        M_boundary[0] = len(boundary)
        return _orig_icp(cand, boundary, *a, **kw)

    _real_tree = tracking.cKDTree
    query_count = [0]

    class _CountTree:
        def __init__(self, *a, **kw):
            self._t = _real_tree(*a, **kw)

        def query(self, *a, **kw):
            query_count[0] += 1
            return self._t.query(*a, **kw)

    pl.points_to_edge_matching = _wrap_icp
    tracking.cKDTree = _CountTree

    events_np = ds.load_events_sorted(opt.eye)
    rss_before_events = rss_bytes()
    tt_config = TemplateTrackingConfig()
    event_samples = pl.template_tracking_stage(
        events_np, ds.frame_list, ellipses, screen_coords, valid_mask, tt_config,
    )
    rss_after_events = rss_bytes()

    pl.points_to_edge_matching = _orig_icp
    tracking.cKDTree = _real_tree

    n_events = len(events_np)
    n_event_samples = len(event_samples)
    C = icp_calls[0]
    Q = query_count[0]
    M = M_boundary[0] if M_boundary[0] else tt_config.num_boundary
    avg_N = (cand_total[0] / C) if C else 0.0
    total_iters = max(Q - C, 0)                   # one extra (final) query per batch
    avg_iters = (total_iters / C) if C else 0.0
    logM = math.log2(M) if M > 1 else 1.0

    # ---- ICP FLOP model (per points_to_edge_matching call) ----
    # build cKDTree on M boundary pts : ~M*log2(M)*3
    # each query (N pts vs tree)       : N points * log2(M) node visits * ~6 (2D dist+cmp)
    # each iteration vector ops        : nearest-P, mean, T/P update ~= N*6
    fl_icp_build = C * (M * logM * 3)
    fl_icp_query = Q * (avg_N * logM * 6)
    fl_icp_iter = total_iters * (avg_N * 6)
    icp_flops_total = fl_icp_build + fl_icp_query + fl_icp_iter
    icp_flops_per = (icp_flops_total / n_event_samples) if n_event_samples else float("nan")

    # ===================================================================== #
    # 3. Regressor: FLOPs for one pupil -> Point of Gaze (exact count model)
    # ===================================================================== #
    pc = np.round(pupil_centers[valid_mask], 2)
    sc = np.round(screen_coords[valid_mask], 2)
    mean, std = compute_pupil_stats(pc)
    pc_n = normalize_pupils(pc, mean, std)

    tracemalloc.start()
    tracemalloc.reset_peak()
    est = GazeEstimator(degree=opt.degree)
    est.fit(pc_n, sc)
    _, reg_py_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Learnable parameters of the whole pipeline (Table 5.1 "Parameters" metric).
    # The frame detector (threshold + morphology + contour/ellipse fit) and the ICP
    # template updater are classical, NON-parametric algorithms: 0 learned weights.
    # The only fitted model is the polynomial gaze regressor: two per-axis linear
    # models, each with F coefficients (incl. the constant feature) + 1 intercept.
    n_params_reg = int(est.regressor_x.coef_.size + est.regressor_y.coef_.size + 2)
    n_params_total = n_params_reg          # detector + updater contribute 0

    # 2-variable polynomial of degree d -> F = (d+1)(d+2)/2 monomials
    F = (opt.degree + 1) * (opt.degree + 2) // 2
    # per prediction: ~F mults to expand monomials + two F-dim dot products (~4F)
    reg_flops_per = F + 4 * F
    # model memory: two coef vectors (F float64) + intercepts
    reg_model_bytes = 2 * F * 8 + 2 * 8

    # ===================================================================== #
    # 3b. LSTM gaze head: params + FLOPs for one window -> Point of Gaze.
    #     Analytical (no training / TF import); the alternative gaze model to
    #     the regressor, reported on the SAME mul+add FLOP convention.
    # ===================================================================== #
    lstm_cfg = LSTMConfig()
    n_params_lstm = lstm_param_count(lstm_cfg)
    lstm_flops_per, lstm_macs_per = lstm_forward_flops(lstm_cfg)
    # model memory: learnable weights as float32 (Keras default)
    lstm_model_bytes = n_params_lstm * 4

    # ===================================================================== #
    # Report
    # ===================================================================== #
    n_valid = int(valid_mask.sum())
    total_outputs = n_valid + n_event_samples

    print("\n" + "=" * 78)
    print("FLOPs  (analytical cost model over measured operation counts)")
    print("=" * 78)
    print(f"image geometry : {W} x {H} = {P} px/frame, {chans} channel(s); "
          f"morph SE {k}x{k}, {se_nnz} active elems")
    print(f"1. frame pupil  : {fmt_flops(frame_flops_per)} / frame   "
          f"(total {fmt_flops(frame_flops_total)} over {n_frames} frames)")
    print(f"     breakdown  : gray {fmt_flops(fl_gray/n_frames)}  thresh {fmt_flops(fl_thresh/n_frames)}  "
          f"morph {fmt_flops(fl_morph/n_frames)}  contour {fmt_flops(fl_contour/n_frames)}  "
          f"fit {fmt_flops(fl_fit/max(n_frames,1))}")
    print(f"2. event pupil  : {fmt_flops(icp_flops_per)} / emitted sample   "
          f"(total {fmt_flops(icp_flops_total)} over {n_event_samples} samples)")
    print(f"     per ICP fit: N~{avg_N:.1f} candidates, M={M} boundary, "
          f"{avg_iters:.1f} iters avg -> {fmt_flops(icp_flops_total/max(C,1))} / fit")
    print(f"3. regressor PoG: {fmt_flops(reg_flops_per)} / sample   "
          f"(deg {opt.degree}, {F} polynomial features)")
    print(f"3b. LSTM PoG    : {fmt_flops(lstm_flops_per)} / window   "
          f"(seq_len {lstm_cfg.seq_len}, {lstm_cfg.lstm_units} units, "
          f"dense {lstm_cfg.dense_units}; {lstm_macs_per} MACs)")

    print("\n" + "-" * 78)
    print("End-to-end FLOPs per Point of Gaze (pupil tracking + gaze estimation)")
    print("-" * 78)
    print(f"  frame -> PoG (regressor) : {fmt_flops(frame_flops_per + reg_flops_per)}")
    print(f"  frame -> PoG (LSTM)      : {fmt_flops(frame_flops_per + lstm_flops_per)}")
    print(f"  event -> PoG (regressor) : {fmt_flops(icp_flops_per + reg_flops_per)}")
    print(f"  event -> PoG (LSTM)      : {fmt_flops(icp_flops_per + lstm_flops_per)}")
    system_flops = frame_flops_total + icp_flops_total + reg_flops_per * total_outputs
    print(f"  whole run    : {fmt_flops(system_flops)} for {total_outputs} PoG "
          f"-> {fmt_flops(system_flops/max(total_outputs,1))} / PoG avg")

    print("\n" + "-" * 78)
    print(f"Energy estimate (first-order: FLOPs x {opt.energy_pj_per_flop:g} pJ/FLOP)")
    print("-" * 78)
    pj = opt.energy_pj_per_flop * 1e-12          # picojoules -> joules per FLOP
    print("  rough model only: scales the analytical FLOP counts by a constant per-FLOP")
    print("  energy; ignores memory traffic, control overhead, and kernel-specific costs.")
    print(f"  frame -> PoG : {fmt_energy((frame_flops_per + reg_flops_per) * pj)} / PoG")
    print(f"  event -> PoG : {fmt_energy((icp_flops_per + reg_flops_per) * pj)} / PoG")
    print(f"  whole run    : {fmt_energy(system_flops * pj)} for {total_outputs} PoG "
          f"-> {fmt_energy(system_flops * pj / max(total_outputs,1))} / PoG avg")

    print("\n" + "=" * 78)
    print("Table 5.1 metrics (Parameters + GFLOPs per inference) for THIS pipeline")
    print("=" * 78)
    print("Paper's GFLOPs (Sec. 4.2.2 / 5.2) = float multiply-add count of ONE network")
    print("forward pass over a 346x260 frame (conv ops dominate); ellipse fitting excluded.")
    print("'Parameters' = learnable model weights. This pipeline is classical (rule-based),")
    print("so the comparable per-inference cost is reported two ways (see note below).")
    print("")
    print("  Parameters = learnable weights. Pupil tracking (detector + ICP updater) is")
    print("  classical/rule-based -> 0 learned weights (it has only a few hand-set")
    print("  hyperparameters, not trained parameters). Gaze estimation is the only")
    print("  parametric stage, so the whole-pipeline count equals the gaze model's:")
    print(f"      detector (frame, classical CV)    : 0  (non-parametric)")
    print(f"      updater  (ICP template, classical): 0  (non-parametric)")
    print(f"      gaze regressor (deg {opt.degree})             : {n_params_reg}  "
          f"(2 axes x ({F} coefs + 1 intercept))")
    print(f"      gaze LSTM head                    : {n_params_lstm}  "
          f"({fmt_params(n_params_lstm).split('(')[1]}")
    print(f"    -> pipeline w/ Regressor : {fmt_params(n_params_reg)}")
    print(f"    -> pipeline w/ LSTM      : {fmt_params(n_params_lstm)}")
    print("")
    print("  Per-stage GFLOPs (one inference), all on the mul+add convention above:")
    print(f"      pupil tracking (frame detector, float-only) : {fmt_gflops(frame_float_per)}  "
          f"(grayscale + ellipse-fit arithmetic)")
    print(f"          (detector all primitive ops incl. int)  : {fmt_gflops(frame_flops_per)}  "
          f"(+ threshold/morphology/contour uint8 compares)")
    print(f"      pupil tracking (ICP updater, all float)     : {fmt_gflops(icp_flops_per)}")
    print(f"      gaze estimation (regressor, deg {opt.degree})          : {fmt_gflops(reg_flops_per)}")
    print(f"      gaze estimation (LSTM head)                 : {fmt_gflops(lstm_flops_per)}  "
          f"({lstm_macs_per} MACs)")
    print("")
    print("  Whole-pipeline GFLOPs = pupil tracking + gaze estimation (frame path):")
    print(f"    -> pipeline w/ Regressor : {fmt_gflops(frame_float_per + reg_flops_per)}  "
          f"(tracking dominates; regressor negligible)")
    print(f"    -> pipeline w/ LSTM      : {fmt_gflops(frame_float_per + lstm_flops_per)}  "
          f"(tracking + LSTM forward)")
    print("")
    print("  NOTE: not a like-for-like number. (1) A CNN spends its FLOPs on millions of")
    print("  float MACs across conv channels; this detector's heavy steps (morphology,")
    print("  threshold, contour) are INTEGER uint8 compares, not FLOPs -- hence the")
    print("  float-only figure is the honest comparison and is ~3 orders of magnitude")
    print("  below 0.553 GFLOPs. (2) NN profilers usually report MACs (1 MAC ~ 2 FLOPs);")
    print("  this script counts each multiply and add separately, so if the paper's number")
    print("  is MACs, halve it to compare -- the order-of-magnitude gap is unaffected.")

    print("\n" + "=" * 78)
    print("MEMORY")
    print("=" * 78)
    print("persistent artifacts (.nbytes / container size):")
    print(f"  raw events  events_np      : {fmt_bytes(events_np.nbytes)}  "
          f"({n_events} x {events_np.shape[1]} {events_np.dtype})")
    print(f"  pupil_centers              : {fmt_bytes(pupil_centers.nbytes)}")
    print(f"  screen_coords              : {fmt_bytes(screen_coords.nbytes)}")
    print(f"  valid_mask                 : {fmt_bytes(valid_mask.nbytes)}")
    print(f"  ellipses list ({len(ellipses)})         : {fmt_bytes(sizeof_list(ellipses))} (approx)")
    print(f"  event_samples list ({n_event_samples})    : {fmt_bytes(sizeof_list(event_samples))} (approx)")
    print("working sets / models:")
    print(f"  frame detection transient  : {fmt_bytes(frame_ws_bytes)} "
          f"(img+gray+binary+opened+contour, held together per frame)")
    print(f"  ICP KDTree (M={M} pts)      : {fmt_bytes(M * 2 * 8 + M * 8)} (boundary + tree nodes, approx)")
    print(f"  regressor model            : {fmt_bytes(reg_model_bytes)} (2 x {F} coefs + intercepts)")
    print(f"  LSTM model                 : {fmt_bytes(lstm_model_bytes)} ({n_params_lstm} weights, float32)")
    print("measured allocation peaks:")
    print(f"  frame stage tracemalloc pk : {fmt_bytes(frame_py_peak)} (Python objs)   "
          f"RSS delta {fmt_bytes(rss_after_frame - rss0)}")
    print(f"  event stage RSS delta      : {fmt_bytes(rss_after_events - rss_before_events)} "
          f"(incl. events_np load)")
    print(f"  regressor fit tracemalloc  : {fmt_bytes(reg_py_peak)} (Python objs)")
    print(f"  process peak resident      : {fmt_bytes(maxrss_bytes())} (ru_maxrss)")

    print("\n" + "-" * 78)
    print("Counts")
    print("-" * 78)
    print(f"  frames processed              : {n_frames}")
    print(f"  valid frame samples           : {n_valid}")
    print(f"  ellipse fits (frame stage)    : {fit_calls[0]}  ({fit_pts[0]} total contour pts)")
    print(f"  raw events streamed           : {n_events}")
    print(f"  event pupil samples emitted   : {n_event_samples}")
    print(f"  ICP batch fits                : {C}")
    print(f"  ICP KDTree queries (= iters+C): {Q}  -> {avg_iters:.2f} iters/fit avg")
    print("=" * 78)


if __name__ == "__main__":
    main()
