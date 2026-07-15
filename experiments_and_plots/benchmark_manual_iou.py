#!/usr/bin/env python3
"""
IoU / F1 / PE benchmark of our pupil-extraction method against the EV-Eye
*manual* ground-truth annotations (VGG Image Annotator ellipses stored in the
per-session ``user_{N}.csv`` files).

Two benchmarks are run, mirroring the EV-Eye paper (Zhao et al., NeurIPS 2023):

1. Frame-based pupil segmentation (§5.2) — apples-to-apples with the paper, which
   reports (averaged over subjects, vs. the same manual GT):

       DL-based (U-Net)    : IoU 0.9187 | F1 0.9560 | PE 0.64 px
       Model-based (EBVEYE): IoU 0.8360 | F1 0.9075 | PE 1.30 px

2. Event-based pupil tracking (§5.3) — there is no event-wise ground truth, so the
   9,011 manually-labeled frames are used as reference. For each labeled image we
   obtain the pupil region of the *last image before it* (via our frame detection),
   run the event-based tracker through the events between the two images (20 events
   per update, as in the paper), and compare the final pupil center against the
   labeled GT center. Only PE (Euclidean center error) is reported. The paper
   reports (averaged over subjects):

       Matching-based (EV-Eye): PE 1.2 px
       Model-based  (EBVEYE)  : PE 7.7 px

Run from the repo root:
    python benchmark_manual_iou.py                 # both benchmarks
    python benchmark_manual_iou.py --skip-events   # frame benchmark only
    python benchmark_manual_iou.py --no-event-filter
"""
import os, sys, csv, json, glob, math, argparse, collections
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("src")
from data.loaders import Frame, glob_imgs
from processing.frame_detection import extract_pupil
from config import (get_frame_detection_config, EV_EYE_FRAME_DETECTION_OVERRIDES,
                    TemplateTrackingConfig)
from tracking import sample_ellipse_boundary, points_to_edge_matching

# Subjects we actually fine-tuned (non-empty override entry in config.py).
TUNED_SUBJECTS = {s for s, ov in EV_EYE_FRAME_DETECTION_OVERRIDES.items() if ov}

DATA_ROOT = "eye_data/ev_eye/raw_data/data_davis"
MASK_SHAPE = (260, 346)  # (H, W) of the DAVIS346 sensor


def gt_ellipse_to_mask(d, shape=MASK_SHAPE):
    """Rasterize a VGG ellipse dict {cx,cy,rx,ry,theta(rad)} into a filled mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(
        mask,
        (int(round(d["cx"])), int(round(d["cy"]))),
        (int(round(d["rx"])), int(round(d["ry"]))),  # rx,ry are semi-axes
        math.degrees(d.get("theta", 0.0)),
        0, 360, color=1, thickness=-1,
    )
    return mask


def pred_ellipse_to_mask(ellipse, shape=MASK_SHAPE):
    """Rasterize our cv2 ellipse ((cx,cy),(w,h),angle_deg) into a filled mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    if ellipse is None:
        return mask
    (cx, cy), (w, h), angle = ellipse
    cv2.ellipse(mask, (int(round(cx)), int(round(cy))),
                (int(round(w / 2)), int(round(h / 2))),
                angle, 0, 360, color=1, thickness=-1)
    return mask


def ellipse_center(ellipse):
    if ellipse is None:
        return None
    (cx, cy), _, _ = ellipse
    return np.array([cx, cy])


def metrics(pred, gt):
    p, g = pred > 0, gt > 0
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    iou = inter / union if union else 0.0
    f1 = 2 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) else 0.0
    return iou, f1


def parse_csv_subject_eye(path):
    """user{N}/{left|right}/... -> (int N, 'left'/'right')."""
    parts = path.replace("\\", "/").split("/")
    subj = next(int(p[4:]) for p in parts if p.startswith("user") and p[4:].isdigit())
    eye = "left" if "left" in parts else "right" if "right" in parts else "left"
    return subj, eye


# --------------------------------------------------------------------------- #
# Event-based pupil tracking benchmark (EV-Eye §5.3)
# --------------------------------------------------------------------------- #

def parse_frame_ts(path):
    """`{idx}_{ts}.png` -> int absolute DAVIS timestamp (µs)."""
    return int(os.path.splitext(os.path.basename(path))[0].split('_')[1])


def build_session_index(base_dir):
    """
    Discover the DAVIS sessions under ``base_dir`` and index their frames so each
    labeled frame can be mapped to the frame that immediately precedes it (needed to
    seed the event tracker) and to its session's event stream.

    Returns:
        sessions   : session_dir -> {'ts': np.ndarray (sorted), 'paths': [str sorted by ts]}
        frame_info : png basename -> (session_dir, position in that session's sorted list)
    """
    sessions, frame_info = {}, {}
    for ev_path in glob.glob(os.path.join(base_dir, "**", "events", "events.txt"),
                             recursive=True):
        session_dir = os.path.dirname(os.path.dirname(ev_path))
        frames_dir = os.path.join(session_dir, "frames")
        if not os.path.isdir(frames_dir):
            continue
        paths = []
        for p in glob_imgs(frames_dir):
            try:
                paths.append((parse_frame_ts(p), p))
            except (IndexError, ValueError):
                continue  # non "{idx}_{ts}.png" file (e.g. timestamps.txt)
        if not paths:
            continue
        paths.sort(key=lambda x: x[0])
        plist = [p for _, p in paths]
        sessions[session_dir] = {
            'ts': np.array([t for t, _ in paths], dtype=np.int64),
            'paths': plist,
        }
        for pos, p in enumerate(plist):
            frame_info[os.path.basename(p)] = (session_dir, pos)
    return sessions, frame_info


def load_session_events(session_dir, cache):
    """
    Load a session's ``events.txt`` as (N, 4) = [polarity, row, col, timestamp] sorted by
    timestamp. Cached per session_dir (files are 100s of MB, so load each at most once).
    """
    if session_dir in cache:
        return cache[session_dir]
    path = os.path.join(session_dir, "events", "events.txt")
    # File columns: ts x(col) y(row) polarity. np.fromfile(sep=' ') parses the whole
    # text file an order of magnitude faster than np.loadtxt.
    raw = np.fromfile(path, dtype=np.int64, sep=' ').reshape(-1, 4)
    events = raw[:, [3, 2, 1, 0]]                # -> polarity, row, col, timestamp
    if not np.all(np.diff(events[:, 3]) >= 0):
        events = events[np.argsort(events[:, 3], kind='stable')]
    cache[session_dir] = events
    return events


def track_center_through_events(events, t0, t1, init_ellipse, config):
    """
    EV-Eye event-based pupil tracking between two consecutive frames.

    Seed the pupil template with ``init_ellipse`` (our frame detection on the frame just
    before the labeled one), then run points-to-edge matching through the events in
    (t0, t1], updating the template center every ``config.num_events`` candidate events
    that fall on the pupil ring. Returns (final_center [x, y], n_updates).
    """
    boundary_Q = sample_ellipse_boundary(init_ellipse, config.num_boundary)
    center = np.array(init_ellipse[0], dtype=np.float64)
    gamma_bar = float(np.mean(np.linalg.norm(boundary_Q - center, axis=1)))
    anchor = center.copy()

    ts = events[:, 3]
    lo = int(np.searchsorted(ts, t0, side='right'))
    hi = int(np.searchsorted(ts, t1, side='right'))

    buf = []
    n_updates = 0
    for i in range(lo, hi):
        row, col = events[i, 1], events[i, 2]
        dist = math.hypot(col - center[0], row - center[1])
        if config.lambda1 * gamma_bar < dist < config.lambda2 * gamma_bar:
            buf.append((col, row))
        if len(buf) < config.num_events:
            continue

        pts = np.array(buf, dtype=np.float64)
        T, residual = points_to_edge_matching(
            pts, boundary_Q, max_iter=config.max_icp_iter, convergence=config.convergence)
        buf = []

        if config.enable_filter:
            # Residual gate (E-Gaze): candidate events must lie on the pupil ring.
            if residual > config.max_residual_ratio * gamma_bar:
                continue
            # Drift bound (Angelopoulos): center can't wander far from the frame anchor.
            new_center = center - T
            if np.linalg.norm(new_center - anchor) > config.max_drift_ratio * gamma_bar:
                continue
            center = new_center
        else:
            center = center - T
        boundary_Q = boundary_Q - T
        n_updates += 1

    return center, n_updates


REF_COLORS = ["tab:red", "tab:purple"]   # one per reference, shared across panels


def draw_metric_violin(ax, data, labels, ylabel, refs=(), ylim=None,
                       show_xticklabels=True):
    """
    Draw a per-frame metric violin plot onto ``ax``: a green "All" violin pooling
    every subject, followed by one blue violin per subject (sorted, labeled with their
    real subject IDs). Short cap lines mark the min and max of each violin, and each
    entry in ``refs`` (label, value) is drawn as a dashed horizontal reference line.
    """
    palette = ["tab:green"] + ["#6ba3d6"] * (len(data) - 1)
    sns.violinplot(data=data, ax=ax, palette=palette, inner="box",
                   linewidth=1.0, cut=0, density_norm="width")
    # Short horizontal caps at the min and max of each violin.
    for x, arr in enumerate(data):
        if len(arr):
            ax.hlines([arr.min(), arr.max()], x - 0.15, x + 0.15,
                      color="black", lw=1.0)
    # Paper reference means as dashed lines (labels handled by the shared legend).
    for (name, val), c in zip(refs, REF_COLORS):
        ax.axhline(val, ls="--", lw=1.6, color=c)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels if show_xticklabels else [], fontsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_ylabel(ylabel, fontsize=17)
    if ylim:
        ax.set_ylim(*ylim)


def plot_all_metrics(records, pe_records, out_path):
    """Stack one violin plot per metric (IoU, F1, PE) into a single figure."""
    keys = sorted(records)                       # (subject, eye) sorted by subject
    labels = ["All"] + [str(k[0]) for k in keys]

    def with_all(per):
        allv = np.concatenate(per) if per else np.array([])
        return [allv] + per

    iou = with_all([np.array([r[0] for r in records[k]]) for k in keys])
    f1 = with_all([np.array([r[1] for r in records[k]]) for k in keys])
    pe = with_all([np.array(pe_records[k], dtype=float) for k in keys])

    fig, axes = plt.subplots(
        3, 1, sharex=True, figsize=(max(8, 0.55 * len(labels) + 2), 8.4),
        gridspec_kw=dict(hspace=0.2))
    draw_metric_violin(axes[0], iou, labels, "IoU score",
                       refs=[("EV-Eye U-Net", 0.9187), ("Angelopoulos model", 0.8360)],
                       ylim=(0.6, 1.0), show_xticklabels=False)
    draw_metric_violin(axes[1], f1, labels, "F1 score",
                       refs=[("EV-Eye U-Net", 0.9560), ("Angelopoulos model", 0.9075)],
                       ylim=(0.75, 1.0), show_xticklabels=False)
    draw_metric_violin(axes[2], pe, labels, "Pixel error (px)",
                       refs=[("EV-Eye U-Net", 0.64), ("Angelopoulos model", 1.30)],
                       ylim=(0, 5), show_xticklabels=True)
    axes[2].set_xlabel("Subject", fontsize=17)
    fig.align_ylabels(axes)

    # Single shared legend (reference names only; values are reported in the paper).
    ref_names = ["EV-Eye U-Net", "Angelopoulos model"]
    handles = [plt.Line2D([], [], ls="--", lw=1.6, color=c)
               for c in REF_COLORS[:len(ref_names)]]
    fig.legend(handles, ref_names, fontsize=12, ncol=len(ref_names),
               loc="upper center", bbox_to_anchor=(0.5, 0.93), framealpha=0.9)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved violin plot to {out_path}")


def plot_event_pe(ev_pe_records, out_path):
    """Violin plot of the event-based tracking pixel error per subject.

    Only PE is shown: during event tracking we follow the pupil *center* through the
    event stream and have no per-event ground truth of the pupil geometry (no mask, so
    no IoU/F1). The dashed lines are the EV-Eye paper's event-tracking references,
    averaged over subjects: matching-based 1.2 px, model-based 7.7 px.
    """
    keys = sorted(ev_pe_records)
    per = [np.array(ev_pe_records[k], dtype=float) for k in keys]
    labels = ["All"] + [str(k[0]) for k in keys]
    allv = np.concatenate(per) if per else np.array([])
    data = [allv] + per

    refs = [("EV-Eye matching-based", 1.2), ("Angelopoulos model", 7.7)]
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels) + 2), 3.4))
    draw_metric_violin(ax, data, labels, "Pixel error (px)", refs=refs,
                       ylim=(0, 10), show_xticklabels=True)
    ax.set_xlabel("Subject", fontsize=17)

    handles = [plt.Line2D([], [], ls="--", lw=1.6, color=c)
               for c in REF_COLORS[:len(refs)]]
    fig.legend(handles, [r[0] for r in refs], fontsize=12, ncol=len(refs),
               loc="upper center", bbox_to_anchor=(0.5, 1.02), framealpha=0.9)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved violin plot to {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-events", action="store_true",
                    help="run only the frame-based segmentation benchmark")
    ap.add_argument("--no-event-filter", action="store_true",
                    help="disable the residual/drift outlier filter in event tracking")
    ap.add_argument("--violin-plot", nargs="?", const="pupil_violin.png",
                    metavar="PATH", default=None,
                    help="save a single figure stacking per-subject violin plots of "
                         "frame IoU/F1/PE (default path: pupil_violin.png)")
    args = ap.parse_args()

    tt_config = TemplateTrackingConfig()
    tt_config.enable_filter = not args.no_event_filter

    csvs = sorted(glob.glob(os.path.join(DATA_ROOT, "**", "*.csv"), recursive=True))
    print(f"Found {len(csvs)} annotation CSV files under {DATA_ROOT}\n")

    # per (subject, eye) accumulators
    records = collections.defaultdict(list)     # key -> list of (iou, f1)
    pe_records = collections.defaultdict(list)  # key -> list of frame center pixel errors
    ev_pe_records = collections.defaultdict(list)  # key -> list of event center pixel errors
    config_cache = {}
    session_cache = {}   # base_dir -> (sessions, frame_info)
    n_anno = n_scored = n_no_detect = n_missing_png = n_empty = 0
    n_ev_scored = n_ev_no_prev = n_ev_no_init = n_ev_no_update = 0

    for csv_path in csvs:
        subj, eye = parse_csv_subject_eye(csv_path)
        if eye != "left":          # research is left-eye only
            continue
        if subj not in TUNED_SUBJECTS:   # only subjects we fine-tuned in config.py
            continue
        base_dir = os.path.dirname(csv_path)
        # index every png under this CSV's directory by filename
        png_index = {os.path.basename(p): p
                     for p in glob.glob(os.path.join(base_dir, "**", "*.png"), recursive=True)}

        # Per-session frame index (prev-frame + event stream lookup) for the event benchmark.
        if not args.skip_events:
            if base_dir not in session_cache:
                session_cache.clear()              # bound memory: keep one base_dir at a time
                sessions, frame_info = build_session_index(base_dir)
                session_cache[base_dir] = (sessions, frame_info, {})  # {} = event arrays
            sessions, frame_info, ev_event_cache = session_cache[base_dir]

        ckey = (subj, eye)
        if ckey not in config_cache:
            config_cache[ckey] = get_frame_detection_config(subj, eye, dataset="ev_eye")
        config = config_cache[ckey]

        with open(csv_path) as f:
            for row in csv.DictReader(f):
                try:
                    shape = json.loads(row["region_shape_attributes"])
                except Exception:
                    continue
                if shape.get("name") != "ellipse":
                    continue
                n_anno += 1
                fname = row["filename"]
                png_path = png_index.get(fname)
                if png_path is None:
                    n_missing_png += 1
                    continue

                gt = gt_ellipse_to_mask(shape)
                if not gt.any():
                    n_empty += 1
                    continue

                _, ellipse = extract_pupil(Frame(0, 0, png_path, 0), config)
                if ellipse is None:
                    n_no_detect += 1
                pred = pred_ellipse_to_mask(ellipse)

                iou, f1 = metrics(pred, gt)
                records[ckey].append((iou, f1))
                n_scored += 1

                c = ellipse_center(ellipse)
                if c is not None:
                    pe_records[ckey].append(
                        np.hypot(c[0] - shape["cx"], c[1] - shape["cy"]))

                # ---- event-based pupil tracking (EV-Eye §5.3) -----------------
                if args.skip_events:
                    continue
                loc = frame_info.get(fname)
                if loc is None or loc[1] == 0:
                    n_ev_no_prev += 1          # labeled frame is first in its session
                    continue
                session_dir, pos = loc
                sess = sessions[session_dir]
                prev_path = sess['paths'][pos - 1]
                t0, t1 = int(sess['ts'][pos - 1]), int(sess['ts'][pos])

                _, init_ellipse = extract_pupil(Frame(0, 0, prev_path, 0), config)
                if init_ellipse is None:
                    n_ev_no_init += 1          # can't seed the template from prev frame
                    continue

                events = load_session_events(session_dir, ev_event_cache)
                ev_center, n_upd = track_center_through_events(
                    events, t0, t1, init_ellipse, tt_config)
                if n_upd == 0:
                    n_ev_no_update += 1        # too few ring events between the frames
                ev_pe_records[ckey].append(
                    np.hypot(ev_center[0] - shape["cx"], ev_center[1] - shape["cy"]))
                n_ev_scored += 1

    # ---- report -----------------------------------------------------------
    print("=" * 78)
    print("Manual-GT pupil benchmark (our method)")
    print("=" * 78)
    print(f"Annotated ellipses found : {n_anno}")
    print(f"Frames scored            : {n_scored}")
    print(f"Skipped (no raw png)     : {n_missing_png}")
    print(f"Skipped (empty GT)       : {n_empty}")
    print(f"Detection failures       : {n_no_detect} "
          f"({n_no_detect / max(n_scored,1):.2%})")
    if not args.skip_events:
        print(f"Event samples scored     : {n_ev_scored} "
              f"(filter {'on' if tt_config.enable_filter else 'off'})")
        print(f"  skipped (no prev frame): {n_ev_no_prev}")
        print(f"  skipped (no seed)      : {n_ev_no_init}")
        print(f"  no event update        : {n_ev_no_update}")
    print()

    hdr = f"{'subject/eye':>14} | {'n':>5} | {'IoU':>6} | {'F1':>6} | {'PE px':>6}"
    if not args.skip_events:
        hdr += f" | {'ev n':>5} | {'ev PE':>6}"
    print(hdr)
    print("-" * len(hdr))

    subj_iou, subj_f1, subj_pe, subj_ev_pe = [], [], [], []
    for key in sorted(records):
        arr = np.array(records[key])
        miou, mf1 = arr[:, 0].mean(), arr[:, 1].mean()
        mpe = np.mean(pe_records[key]) if pe_records[key] else float("nan")
        subj_iou.append(miou); subj_f1.append(mf1); subj_pe.append(mpe)
        line = (f"{f'{key[0]} {key[1]}':>14} | {len(arr):>5} | "
                f"{miou:>6.4f} | {mf1:>6.4f} | {mpe:>6.2f}")
        if not args.skip_events:
            ev = ev_pe_records[key]
            mev = np.mean(ev) if ev else float("nan")
            subj_ev_pe.append(mev)
            line += f" | {len(ev):>5} | {mev:>6.2f}"
        print(line)

    all_arr = np.array([r for v in records.values() for r in v])
    all_ev = [p for v in ev_pe_records.values() for p in v]
    print("-" * len(hdr))
    micro = (f"{'MICRO (all)':>14} | {len(all_arr):>5} | "
             f"{all_arr[:,0].mean():>6.4f} | {all_arr[:,1].mean():>6.4f} | "
             f"{np.mean([p for v in pe_records.values() for p in v]):>6.2f}")
    macro = (f"{'MACRO (subj)':>14} | {len(subj_iou):>5} | "
             f"{np.mean(subj_iou):>6.4f} | {np.mean(subj_f1):>6.4f} | "
             f"{np.nanmean(subj_pe):>6.2f}")
    if not args.skip_events:
        micro += (f" | {len(all_ev):>5} | "
                  f"{np.mean(all_ev) if all_ev else float('nan'):>6.2f}")
        macro += f" | {'':>5} | {np.nanmean(subj_ev_pe) if subj_ev_pe else float('nan'):>6.2f}"
    print(micro)
    print(macro)

    print("\nPaper reference (macro over subjects, same manual GT):")
    print(f"{'EV-Eye U-Net':>14} |       | 0.9187 | 0.9560 |   0.64", end="")
    print(f" |       |   1.20" if not args.skip_events else "")
    print(f"{'EV-Eye model':>14} |       | 0.8360 | 0.9075 |   1.30", end="")
    print(f" |       |   7.70" if not args.skip_events else "")
    if not args.skip_events:
        print("  (frame cols: IoU/F1/PE  •  event col: PE px — matching-based vs model-based)")

    if args.violin_plot is not None:
        plot_all_metrics(records, pe_records, args.violin_plot)
        if not args.skip_events and ev_pe_records:
            root, ext = os.path.splitext(args.violin_plot)
            plot_event_pe(ev_pe_records, f"{root}_events{ext}")


if __name__ == "__main__":
    main()
