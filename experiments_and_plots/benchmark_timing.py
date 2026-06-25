"""Latency / throughput benchmark for the gaze pipeline on a single ev_eye subject.

Measures, on the REAL pipeline code paths:
  1. frame  : avg time to extract a pupil from one APS image   (frame_detection.extract_pupil)
  2. events : avg time to extract a pupil from a batch of events (template-tracking ICP fit)
  3. regress: avg time to map one pupil center -> Point of Gaze  (GazeEstimator.predict, 1 sample)
  4. system : implied end-to-end output frequency of the whole system

Run from the repo root:
    source .venv/bin/activate
    python notebooks/benchmark_timing.py --subject 22 --eye left --motion saccadic --degree 2
"""
import os
import sys
import time
import argparse

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import cv2  # noqa: E402
from config import (DATASET_PATHS, get_frame_detection_config, get_gaze_config,  # noqa: E402
                    TemplateTrackingConfig)
from data.loaders import EvEyeDataset  # noqa: E402
from processing.frame_detection import extract_pupil  # noqa: E402
from processing.normalization import compute_pupil_stats, normalize_pupils  # noqa: E402
from models.polynomial import GazeEstimator  # noqa: E402
import pipeline.pipeline as pl  # noqa: E402


def summary(name, times_ms):
    a = np.asarray(times_ms, dtype=np.float64)
    return (f"{name:<28} n={a.size:>7}  mean={a.mean():8.3f} ms  "
            f"median={np.median(a):8.3f} ms  std={a.std():7.3f} ms  "
            f"-> {1000.0 / a.mean():8.1f} Hz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=22)
    ap.add_argument("--eye", default="left", choices=["left", "right"])
    ap.add_argument("--motion", default="saccadic", choices=["saccadic", "pursuit"])
    ap.add_argument("--degree", type=int, default=2,
                    help="polynomial degree to benchmark the regressor at")
    ap.add_argument("--reg_reps", type=int, default=20000,
                    help="single-sample regressor predictions to time")
    ap.add_argument("--preload", action="store_true",
                    help="preload all decoded frames into RAM and serve cv2.imread from "
                         "memory, so the frame-pupil timing excludes disk read AND PNG "
                         "decode (pure detection compute)")
    opt = ap.parse_args()

    data_dir = DATASET_PATHS["ev_eye"]
    frame_config = get_frame_detection_config(opt.subject, opt.eye, "ev_eye")
    gaze_config = get_gaze_config(opt.subject)

    print(f"=== Benchmark: ev_eye subject {opt.subject}, {opt.eye} eye, {opt.motion} ===\n")

    # ---- load data -------------------------------------------------------
    ds = EvEyeDataset(data_dir, opt.subject, motion=opt.motion, mode="np")
    ds.collect_data(eye=opt.eye)

    # =====================================================================
    # 1. Frame pupil extraction (per APS image)  -- single pass
    # =====================================================================
    # Time each extract_pupil call during the normal pupil_extraction_stage by
    # wrapping the function the stage calls (no extra pass over the images).
    frame_list = ds.frame_list
    extract_ms = []
    import processing.frame_detection as fd
    _orig_extract = fd.extract_pupil

    def _timed_extract(*a, **k):
        t0 = time.perf_counter()
        out = _orig_extract(*a, **k)
        extract_ms.append((time.perf_counter() - t0) * 1000)
        return out

    # --preload: decode every frame into RAM up front (this IS the I/O, done before
    # timing) and serve cv2.imread from the cache during detection, so the timed
    # frame-pupil cost is pure detection compute — no disk read, no PNG decode.
    _orig_imread = cv2.imread
    cache = {}
    if opt.preload:
        print(f"Preloading {len(frame_list) - 1} frames into RAM...")
        for f in frame_list[1:]:
            cache[f.img] = _orig_imread(f.img)

        def _cached_imread(path, *a, **k):
            img = cache.get(path)
            return img if img is not None else _orig_imread(path, *a, **k)

        cv2.imread = _cached_imread

    fd.extract_pupil = _timed_extract
    pupil_centers, ellipses, screen_coords = pl.pupil_extraction_stage(ds, frame_config)
    fd.extract_pupil = _orig_extract  # restore
    cv2.imread = _orig_imread         # restore

    if opt.preload:
        t_imread_mean = 0.0           # I/O excluded entirely from the timed run
    else:
        # Estimate the disk-I/O share of extract_pupil on a small sample of frames.
        imread_ms = []
        for f in frame_list[1:min(len(frame_list), 301)]:
            t0 = time.perf_counter()
            _orig_imread(f.img)
            imread_ms.append((time.perf_counter() - t0) * 1000)
        t_imread_mean = float(np.mean(imread_ms))

    t_frame_mean = float(np.mean(extract_ms))
    t_detect_mean = max(t_frame_mean - t_imread_mean, 0.0)  # detection only (no disk I/O)

    # =====================================================================
    # Build valid mask (needed to drive event extraction & regressor)
    # =====================================================================
    blink_mask = pl.noise_flagging_stage(pupil_centers)
    valid_mask = pl.build_valid_mask(
        blink_mask, screen_coords,
        skip_frames=gaze_config.saccade_skip_frames,
        skip_label_changes=False,
        post_blink_skip_frames=gaze_config.post_blink_skip_frames,
        alignment_gaps=ds.alignment_gaps,
        max_alignment_gap_us=gaze_config.max_alignment_gap_us,
    )

    # =====================================================================
    # 2. Event pupil extraction (per ICP batch fit)
    # =====================================================================
    # Wrap the ICP fit used inside template_tracking_stage to time each batch fit.
    icp_ms = []
    _orig_icp = pl.points_to_edge_matching

    def _timed_icp(*a, **k):
        t0 = time.perf_counter()
        out = _orig_icp(*a, **k)
        icp_ms.append((time.perf_counter() - t0) * 1000)
        return out

    pl.points_to_edge_matching = _timed_icp
    events_np = ds.load_events_sorted(opt.eye)
    tt_config = TemplateTrackingConfig()
    t0 = time.perf_counter()
    event_samples = pl.template_tracking_stage(
        events_np, ds.frame_list, ellipses, screen_coords, valid_mask, tt_config,
    )
    tt_wall_s = time.perf_counter() - t0
    pl.points_to_edge_matching = _orig_icp  # restore

    n_events = len(events_np)
    n_event_samples = len(event_samples)
    t_event_mean = float(np.mean(icp_ms)) if icp_ms else float("nan")
    # amortized: total template-tracking wall time spread over emitted event pupils
    t_event_amortized = (tt_wall_s * 1000 / n_event_samples) if n_event_samples else float("nan")

    # =====================================================================
    # 3. Regressor: single pupil -> Point of Gaze
    # =====================================================================
    pc = np.round(pupil_centers[valid_mask], 2)
    sc = np.round(screen_coords[valid_mask], 2)
    mean, std = compute_pupil_stats(pc)
    pc_n = normalize_pupils(pc, mean, std)

    est = GazeEstimator(degree=opt.degree)
    est.fit(pc_n, sc)

    sample = pc_n[0].reshape(1, -1)
    for _ in range(100):  # warmup
        est.predict(sample)
    reg_ms = []
    reps = min(opt.reg_reps, max(1000, len(pc_n)))
    for i in range(reps):
        s = pc_n[i % len(pc_n)].reshape(1, -1)
        t0 = time.perf_counter()
        est.predict(s)
        reg_ms.append((time.perf_counter() - t0) * 1000)
    t_reg_mean = float(np.mean(reg_ms))

    # =====================================================================
    # Report
    # =====================================================================
    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    if opt.preload:
        print(summary("1. frame pupil (no I/O)", extract_ms) + "   [preloaded: pure detection]")
    else:
        print(summary("1. frame pupil (img+I/O)", extract_ms))
        print(f"{'   frame pupil (detect only)':<28} "
              f"mean={t_detect_mean:8.3f} ms  -> {1000.0/t_detect_mean:8.1f} Hz   "
              f"(I/O share ~{t_imread_mean:.3f} ms, est. on sampled frames)")
    print(summary(f"2. event pupil (ICP fit)", icp_ms))
    print(f"{'   event pupil (amortized)':<28} "
          f"per-sample over template-tracking wall time = {t_event_amortized:8.3f} ms"
          f"  -> {1000.0/t_event_amortized:8.1f} Hz")
    print(summary(f"3. regressor PoG (deg {opt.degree})", reg_ms))

    print("\n" + "-" * 78)
    print("Counts")
    print("-" * 78)
    print(f"  frames processed              : {len(frame_list)-1}")
    print(f"  valid frame samples           : {int(valid_mask.sum())}")
    print(f"  raw events streamed           : {n_events}")
    print(f"  event pupil samples emitted   : {n_event_samples}")
    print(f"  ICP batch fits                : {len(icp_ms)}")

    print("\n" + "-" * 78)
    print("System frequency (per-stage end-to-end latency = extract + regress)")
    print("-" * 78)
    frame_path_ms = t_frame_mean + t_reg_mean
    frame_path_det_ms = t_detect_mean + t_reg_mean
    event_path_ms = t_event_mean + t_reg_mean
    print(f"  frame  -> PoG : {frame_path_ms:8.3f} ms  "
          f"-> {1000.0/frame_path_ms:7.1f} Hz  (incl. image I/O)")
    print(f"  frame  -> PoG : {frame_path_det_ms:8.3f} ms  "
          f"-> {1000.0/frame_path_det_ms:7.1f} Hz  (detection only, no I/O)")
    print(f"  event  -> PoG : {event_path_ms:8.3f} ms  "
          f"-> {1000.0/event_path_ms:7.1f} Hz")

    # End-to-end throughput: every PoG the system emits / total compute time.
    total_outputs = int(valid_mask.sum()) + n_event_samples
    total_compute_s = (
        (np.sum(extract_ms) / 1000.0)         # all frame detections (incl I/O)
        + tt_wall_s                            # all event extraction
        + (t_reg_mean / 1000.0) * total_outputs  # one regression per output
    )
    print(f"\n  end-to-end throughput : {total_outputs} PoG / {total_compute_s:.2f} s "
          f"= {total_outputs/total_compute_s:7.1f} Hz")
    print("=" * 78)


if __name__ == "__main__":
    main()
