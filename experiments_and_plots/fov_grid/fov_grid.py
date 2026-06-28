#!/usr/bin/env python3
"""Spatial DoD / density map over an N x N grid of the field of view.

Standalone script (does NOT modify the pipeline): it runs the same pipeline stages and
the same events_eval protocol as

    python src/main.py --subject 7 --dataset ev_eye --eye left --motion saccadic \
                       --events_eval --eval_split blocks

i.e. it calibrates a polynomial on the calibration-block frame pupil centers and evaluates
on the held-out eval-block frames PLUS the high-frequency event centers. Each evaluation
sample is then binned by its gaze-label position, in DEGREES of field of view, into an
N x N grid centred at (0,0). Two heatmaps are written per subject:

    *_label_dod.png   — mean per-cell DoD error
    *_label_count.png — number of (frame + event) ellipses per cell

Run from the repo root:
    python experiments_and_plots/fov_grid/fov_grid.py --subject 7 --grid 8 --degree 2
    python experiments_and_plots/fov_grid/fov_grid.py --subject 4 5 6 --eval_split blocks
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'src'))

from data.loaders import EyeDataset, EvEyeDataset                       # noqa: E402
from config import (DATASET_PATHS, get_frame_detection_config,          # noqa: E402
                    get_gaze_config, TemplateTrackingConfig)
from pipeline.pipeline import (pupil_extraction_stage, noise_flagging_stage,  # noqa: E402
                               build_valid_mask, template_tracking_stage,
                               label_events_from_tobii)
from pipeline.runners import _gaze_unit_dirs, gaze_clip_bounds          # noqa: E402
from models.polynomial import GazeEstimator                            # noqa: E402


# ---------------------------------------------------------------------------------------
# Pipeline data prep (mirrors run_pipeline, single-subject, no relabeling)
# ---------------------------------------------------------------------------------------
def load_subject(subject, eye, dataset, motion):
    data_dir = DATASET_PATHS[dataset]
    frame_config = get_frame_detection_config(subject, eye, dataset)
    gaze_config = get_gaze_config(subject, dataset)

    if dataset == 'ev_eye':
        ds = EvEyeDataset(data_dir, subject, motion=motion, mode='np')
        eye_key = eye
        ds.collect_data(eye=eye_key)
    else:
        ds = EyeDataset(data_dir, subject, mode='stack')
        eye_key = 0 if eye == 'left' else 1
        ds.collect_data(eye=eye_key, motion=motion)

    pupil_centers, ellipses, screen_coords = pupil_extraction_stage(ds, frame_config)
    blink_mask = noise_flagging_stage(pupil_centers)
    skip_label_changes = (dataset == 'ebveye') and (motion == 'saccadic')
    valid_mask = build_valid_mask(
        blink_mask, screen_coords,
        skip_frames=gaze_config.saccade_skip_frames,
        skip_label_changes=skip_label_changes,
        post_blink_skip_frames=gaze_config.post_blink_skip_frames,
        alignment_gaps=getattr(ds, 'alignment_gaps', None),
        max_alignment_gap_us=gaze_config.max_alignment_gap_us,
    )

    events_np = ds.load_events_sorted(eye_key)
    event_samples = template_tracking_stage(
        events_np, ds.frame_list, ellipses, screen_coords, valid_mask,
        TemplateTrackingConfig())
    if getattr(ds, 'gaze_records', None) is not None:
        event_samples = label_events_from_tobii(event_samples, ds.gaze_records)

    frame_timestamps = np.array([f.timestamp for f in ds.frame_list], dtype=np.int64)
    return (pupil_centers, screen_coords, valid_mask, event_samples,
            frame_timestamps, gaze_config, dataset == 'ev_eye')


# ---------------------------------------------------------------------------------------
# events_eval protocol: calibrate on calibration-block frames, evaluate on eval-block
# frames + all/eval-block events. Returns the eval-set gaze-label degrees + per-sample DoD.
# (Replicates run_regressor_events_eval's blocks/within split.)
# ---------------------------------------------------------------------------------------
def build_eval_samples(pupil_centers, screen_coords, valid_mask, event_samples,
                       frame_timestamps, gaze_config, normalized, degree, eval_split):
    frame_pupils = np.round(pupil_centers[valid_mask], 2)
    frame_screens = np.round(screen_coords[valid_mask], 2)
    frame_ts = np.asarray(frame_timestamps, dtype=np.int64)[valid_mask]

    ec = np.array([[s['ellipse'][0][0], s['ellipse'][0][1]] for s in event_samples],
                  dtype=np.float64)
    el = np.array([s['screen_coord'] for s in event_samples], dtype=np.float64)
    et = np.array([s['timestamp'] for s in event_samples], dtype=np.int64)
    vev = ~np.all(el == 0, axis=1)
    if event_samples and event_samples[0].get('gap_us') is not None:
        gaps = np.array([s.get('gap_us', 0) for s in event_samples])
        vev &= gaps <= gaze_config.max_alignment_gap_us
    ec, el, et = ec[vev], el[vev], et[vev]

    # Temporal block split shared by both streams (see run_regressor_events_eval).
    n_blocks = gaze_config.n_time_blocks
    t0 = int(min(frame_ts.min(), et.min()))
    t1 = int(max(frame_ts.max(), et.max()))
    span = max(t1 - t0, 1)

    def block_ids(ts):
        return np.clip(((ts - t0) / span * n_blocks).astype(int), 0, n_blocks - 1)

    if eval_split == 'within':
        fb = block_ids(frame_ts)
        rng = np.random.default_rng(42)
        f_train = np.zeros(len(frame_ts), dtype=bool)
        for b in range(n_blocks):
            idx = np.where(fb == b)[0]
            if len(idx) == 0:
                continue
            rng.shuffle(idx)
            f_train[idx[:int(len(idx) * gaze_config.train_ratio)]] = True
        f_eval = ~f_train
        e_eval = np.ones(len(et), dtype=bool)
    else:  # blocks
        n_val = max(1, min(n_blocks - 1, round(n_blocks * gaze_config.val_ratio)))
        val_blocks = np.linspace(0, n_blocks - 1, n_val).round().astype(int)
        f_train = ~np.isin(block_ids(frame_ts), val_blocks)
        f_eval = np.isin(block_ids(frame_ts), val_blocks)
        e_eval = np.isin(block_ids(et), val_blocks)

    est = GazeEstimator(degree=degree, clip_bounds=gaze_clip_bounds(gaze_config, normalized))
    est.fit(frame_pupils[f_train], frame_screens[f_train])

    feval, feval_y = frame_pupils[f_eval], frame_screens[f_eval]
    eveval, eveval_y = ec[e_eval], el[e_eval]

    def dod(pred, gt):
        up = _gaze_unit_dirs(pred, gaze_config, normalized)
        ug = _gaze_unit_dirs(gt, gaze_config, normalized)
        return np.degrees(np.arccos(np.clip(np.sum(up * ug, axis=1), -1.0, 1.0)))

    all_dod = np.concatenate([dod(est.predict(feval), feval_y),
                              dod(est.predict(eveval), eveval_y)])
    labels = np.vstack([feval_y, eveval_y])

    # Gaze-label position in degrees from screen centre (0,0 = centre).
    y_lab, x_lab = labels[:, 0], labels[:, 1]
    if not normalized:
        x_lab = x_lab / gaze_config.screen_width_px
        y_lab = y_lab / gaze_config.screen_height_px
    deg_x = (x_lab - 0.5) * gaze_config.screen_fov_x_deg
    deg_y = (y_lab - 0.5) * gaze_config.screen_fov_y_deg
    return dict(deg_x=deg_x, deg_y=deg_y, dod=all_dod,
                n_frames=len(feval), n_events=len(eveval))


# ---------------------------------------------------------------------------------------
# Grid binning + plotting
# ---------------------------------------------------------------------------------------
def grid_stats(xs, ys, vals, x_edges, y_edges):
    """Return (mean_per_cell, count_per_cell), both shaped (ny, nx)=(rows, cols)."""
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    ix = np.clip(np.searchsorted(x_edges, xs, side='right') - 1, 0, nx - 1)
    iy = np.clip(np.searchsorted(y_edges, ys, side='right') - 1, 0, ny - 1)
    count = np.zeros((ny, nx))
    vsum = np.zeros((ny, nx))
    np.add.at(count, (iy, ix), 1)
    np.add.at(vsum, (iy, ix), vals)
    mean = np.full((ny, nx), np.nan)
    nz = count > 0
    mean[nz] = vsum[nz] / count[nz]
    return mean, count


def draw_heatmap(ax, fig, grid, x_edges, y_edges, cbar_label, fmt='{:.1f}', cmap='viridis'):
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad('#ededed')                  # empty cells: soft grey
    masked = np.ma.masked_invalid(grid)
    vmin = np.nanmin(grid) if np.isfinite(grid).any() else 0.0
    vmax = np.nanmax(grid) if np.isfinite(grid).any() else 1.0
    mesh = ax.pcolormesh(x_edges, y_edges, masked, cmap=cmap_obj, shading='flat',
                         edgecolors='white', linewidth=0.8, vmin=vmin, vmax=vmax)

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(cbar_label, fontsize=15)
    cbar.ax.tick_params(labelsize=12)
    cbar.outline.set_visible(False)

    # Subtle crosshair through the screen centre (0,0).
    ax.axhline(0, color='0.25', lw=0.9, ls='--', alpha=0.5)
    ax.axvline(0, color='0.25', lw=0.9, ls='--', alpha=0.5)

    if grid.shape[0] <= 12:                      # annotate cells when readable
        xc = (x_edges[:-1] + x_edges[1:]) / 2
        yc = (y_edges[:-1] + y_edges[1:]) / 2
        span = max(vmax - vmin, 1e-9)
        for i, yy in enumerate(yc):
            for j, xx in enumerate(xc):
                v = grid[i, j]
                if np.isnan(v):
                    continue
                ax.text(xx, yy, fmt.format(v), ha='center', va='center', fontsize=8.5,
                        color='white' if (v - vmin) / span < 0.55 else '0.1')

    ax.set_xlabel('Yaw [deg]', fontsize=16)
    ax.set_ylabel('Pitch [deg]', fontsize=16)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_aspect('equal')                       # true FoV proportions
    ax.tick_params(labelsize=12, length=0)
    ax.invert_yaxis()                            # screen-top at the top
    for spine in ax.spines.values():
        spine.set_visible(False)


def process_subject(subject, eye, dataset, motion, degree, grid, eval_split, out_dir):
    print(f"\n==== subject {subject} | {dataset} | {eye} | {motion} | events_eval "
          f"{eval_split} | degree {degree} | {grid}x{grid} ====")
    (pupil_centers, screen_coords, valid_mask, event_samples,
     frame_timestamps, gaze_config, normalized) = load_subject(subject, eye, dataset, motion)
    s = build_eval_samples(pupil_centers, screen_coords, valid_mask, event_samples,
                           frame_timestamps, gaze_config, normalized, degree, eval_split)

    os.makedirs(out_dir, exist_ok=True)
    n = grid
    fx, fy = gaze_config.screen_fov_x_deg, gaze_config.screen_fov_y_deg
    x_edges = np.linspace(-fx / 2, fx / 2, n + 1)
    y_edges = np.linspace(-fy / 2, fy / 2, n + 1)
    dod_grid, cnt_grid = grid_stats(s['deg_x'], s['deg_y'], s['dod'], x_edges, y_edges)

    tag = f"s{subject}_{eye}_{motion}_{eval_split}_d{degree}_{n}x{n}"
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    draw_heatmap(axes[0], fig, dod_grid, x_edges, y_edges, 'Mean DoD [deg]')
    draw_heatmap(axes[1], fig, cnt_grid, x_edges, y_edges, 'Ellipses (count)',
                 fmt='{:.0f}', cmap='magma')
    fig.tight_layout()
    out_path = os.path.join(out_dir, f'{tag}_label.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    -> {out_path}")

    print(f"  eval set: frames={s['n_frames']}  events={s['n_events']}  "
          f"total ellipses={len(s['dod'])}")
    print(f"  overall mean DoD = {np.nanmean(s['dod']):.2f}°")
    return float(np.nanmean(s['dod'])), len(s['dod'])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subject', type=int, nargs='+', default=[7])
    ap.add_argument('--eye', default='left', choices=['left', 'right'])
    ap.add_argument('--dataset', default='ev_eye', choices=list(DATASET_PATHS.keys()))
    ap.add_argument('--motion', default='saccadic', choices=['saccadic', 'pursuit'])
    ap.add_argument('--eval_split', default='blocks', choices=['blocks', 'within'])
    ap.add_argument('--degree', type=int, default=2)
    ap.add_argument('--grid', type=int, default=8, help='N for the NxN grid')
    ap.add_argument('--out_dir', default=PLOT_DIR)
    opt = ap.parse_args()

    summary = []
    for subj in opt.subject:
        dod, n = process_subject(subj, opt.eye, opt.dataset, opt.motion, opt.degree,
                                 opt.grid, opt.eval_split, opt.out_dir)
        summary.append((subj, dod, n))

    print("\n==== overall mean DoD per subject ====")
    for subj, dod, n in summary:
        print(f"  subject {subj}: {dod:.2f}°  (n={n} ellipses)")
    if summary:
        print(f"  mean over {len(summary)} subjects: "
              f"{np.mean([d for _, d, _ in summary]):.2f}°")


if __name__ == '__main__':
    main()
