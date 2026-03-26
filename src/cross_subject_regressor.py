"""
Leave-one-out (or single-fold) cross-subject polynomial regressor evaluation.

Trains the polynomial regressor on N-1 subjects and evaluates on the held-out
subject, with per-subject z-score normalization of pupil coordinates applied
before pooling.

Preprocessed data is cached to data_cache/ so re-runs skip the slow
pupil-extraction step. Delete data_cache/ to force re-preprocessing.
Note: --relabel changes the filtered output, so the cache key includes
whether it is set.

Usage:
    # Single fold: train on all SUBJECTS except 22, test on 22
    python src/cross_subject_regressor.py --val_subject 22

    # Full LOO over all SUBJECTS
    python src/cross_subject_regressor.py

    # With relabeling and gaze plots
    python src/cross_subject_regressor.py --val_subject 22 --relabel --ge_plots
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import GazeConfig, get_frame_detection_config, get_gaze_config
from data.loaders import EyeDataset
from data.visualization import plot_gaze_predictions
from models.polynomial import GazeEstimator
from pipeline.pipeline import (
    build_valid_mask,
    noise_flagging_stage,
    pupil_extraction_stage,
    relabeling_stage,
)
from pipeline.runners import fov_filter_mask
from processing.normalization import compute_pupil_stats, normalize_pupils

SUBJECTS   = [4, 5, 6, 7, 11, 12, 15, 18, 19, 22]
EYE        = 'left'
FOV        = (40.0, 20.0)
FOV_CENTER = (501, 879)          # set to (row_px, col_px) to shift the FoV window
CACHE_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data_cache')


def _fov_rect(gaze_config, fov, fov_center):
    """Return (row_min, row_max, col_min, col_max) in pixels for the FoV window."""
    px_per_deg_x = gaze_config.screen_width_px / gaze_config.screen_fov_x_deg
    px_per_deg_y = gaze_config.screen_height_px / gaze_config.screen_fov_y_deg
    half_w = (fov[0] / 2) * px_per_deg_x
    half_h = (fov[1] / 2) * px_per_deg_y
    if fov_center is None:
        cr = gaze_config.screen_height_px / 2
        cc = gaze_config.screen_width_px / 2
    else:
        cr, cc = fov_center
    return (cr - half_h, cr + half_h, cc - half_w, cc + half_w)


def load_subject_data(subject, data_dir, relabel, fov, fov_center):
    """
    Run the full preprocessing pipeline for one subject and return filtered
    raw (unnormalized) pupil_centers and screen_coords.

    Results are cached to CACHE_DIR so subsequent runs skip re-extraction.
    The cache key includes relabel since it affects filtering.
    """
    relabel_tag = 'rel1' if relabel else 'rel0'
    cache_path = os.path.join(CACHE_DIR, f'subject_{subject}_{EYE}_regressor_{relabel_tag}.npz')
    if os.path.exists(cache_path):
        print(f"  Subject {subject}: loading from cache")
        data = np.load(cache_path)
        return data['pupil_centers'], data['screen_coords']

    print(f"  Subject {subject}: preprocessing...")
    eye_index    = 0 if EYE == 'left' else 1
    frame_config = get_frame_detection_config(subject, EYE)
    gaze_config  = get_gaze_config(subject)

    eye_dataset = EyeDataset(data_dir, subject, mode='stack')
    eye_dataset.collect_data(eye=eye_index)

    pupil_centers, ellipses, screen_coords = pupil_extraction_stage(eye_dataset, frame_config)
    blink_mask = noise_flagging_stage(pupil_centers)

    if relabel:
        sc, saccade_mask = relabeling_stage(pupil_centers, screen_coords, gaze_config)
    else:
        sc, saccade_mask = screen_coords, None

    valid_mask = build_valid_mask(
        blink_mask, sc,
        skip_frames=gaze_config.saccade_skip_frames,
        saccade_mask=saccade_mask,
        skip_label_changes=not relabel,
        post_blink_skip_frames=gaze_config.post_blink_skip_frames,
    )

    pupil_centers = np.round(pupil_centers[valid_mask], 2)
    screen_coords = np.round(sc[valid_mask], 2)

    fov_mask = fov_filter_mask(screen_coords, fov[0], fov[1], gaze_config, center=fov_center)
    pupil_centers = pupil_centers[fov_mask]
    screen_coords = screen_coords[fov_mask]

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache_path, pupil_centers=pupil_centers, screen_coords=screen_coords)
    print(f"  Subject {subject}: {len(pupil_centers)} frames — cached to {cache_path}")
    return pupil_centers, screen_coords


def run_fold(val_subject, subject_data, ge_plots, fov, fov_center):
    """
    Train on all subjects except val_subject, evaluate on val_subject.
    Per-subject z-score normalization is applied before pooling.
    """
    gaze_config = GazeConfig()

    # Compute per-subject normalization stats from raw pupils
    stats = {
        s: compute_pupil_stats(pupil_centers)
        for s, (pupil_centers, _) in subject_data.items()
    }

    # Build training set: pool normalized pupils from all train subjects
    train_pupils, train_screens = [], []
    for s, (pupil_centers, screen_coords) in subject_data.items():
        if s == val_subject:
            continue
        mean, std = stats[s]
        train_pupils.append(normalize_pupils(pupil_centers, mean, std))
        train_screens.append(screen_coords)
    pupil_train = np.concatenate(train_pupils)
    screen_train = np.concatenate(train_screens)

    # Validation set: normalize with val subject's own stats
    val_mean, val_std = stats[val_subject]
    pupil_val, screen_val = subject_data[val_subject]
    pupil_val = normalize_pupils(pupil_val, val_mean, val_std)

    print(f"Train: {len(pupil_train)} frames  |  Val: {len(pupil_val)} frames")

    results = {}
    for deg in gaze_config.poly_degrees:
        print(f"\n  --- Degree {deg} ---")
        estimator = GazeEstimator(degree=deg)
        estimator.fit(pupil_train, screen_train)
        metrics = estimator.evaluate(pupil_val, screen_val)
        results[deg] = metrics
        print(f"  mean={metrics['mean_error']:.2f}px  rmse={metrics['rmse']:.2f}px  "
              f"median={metrics['median_error']:.2f}px")

        if ge_plots:
            val_pred = estimator.predict(pupil_val)
            plot_gaze_predictions(
                val_pred, screen_val,
                title=f'Subject {val_subject} — Degree {deg}',
                fov_rect=_fov_rect(gaze_config, fov, fov_center),
            )

    best_deg = min(results, key=lambda d: results[d]['mean_error'])
    print(f"\n  Best degree: {best_deg}  (mean={results[best_deg]['mean_error']:.2f}px)")
    return results


def main(data_dir, val_subject, relabel, ge_plots, fov=FOV, fov_center=FOV_CENTER):
    print("=" * 60)
    print("Preprocessing / loading subjects...")
    print("=" * 60)
    subject_data = {}
    for s in SUBJECTS:
        subject_data[s] = load_subject_data(s, data_dir, relabel, fov, fov_center)

    if val_subject is not None:
        if val_subject not in SUBJECTS:
            raise ValueError(f"--val_subject {val_subject} is not in SUBJECTS list: {SUBJECTS}")
        print()
        print("=" * 60)
        print(f"Fold: val = subject {val_subject}")
        print("=" * 60)
        run_fold(val_subject, subject_data, ge_plots, fov, fov_center)
        return

    # Full LOO
    all_results = {}
    for s in SUBJECTS:
        print()
        print("=" * 60)
        print(f"Fold: val = subject {s}")
        print("=" * 60)
        fold_results = run_fold(s, subject_data, ge_plots, fov, fov_center)
        best_deg = min(fold_results, key=lambda d: fold_results[d]['mean_error'])
        all_results[s] = fold_results[best_deg]

    print()
    print("=" * 60)
    print("Summary (best degree per fold)")
    print("=" * 60)
    for s in SUBJECTS:
        m = all_results[s]
        print(f"  {s:>3}: mean={m['mean_error']:.2f}px  rmse={m['rmse']:.2f}px  "
              f"median={m['median_error']:.2f}px  std={m['std_error']:.2f}px")
    mean_errors = [all_results[s]['mean_error'] for s in SUBJECTS]
    print(f"\nOverall mean error: {np.mean(mean_errors):.2f} ± {np.std(mean_errors):.2f} px")


def run(opt):
    fov = tuple(opt.fov) if opt.fov else FOV
    fov_center = tuple(opt.fov_center) if opt.fov_center else FOV_CENTER
    main(opt.data_dir, opt.val_subject, opt.relabel, opt.ge_plots, fov, fov_center)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=os.path.join(os.getcwd(), 'eye_data'))
    parser.add_argument('--val_subject', type=int, default=None,
                        help='Subject to hold out for evaluation. If omitted, runs full LOO.')
    parser.add_argument('--relabel', action='store_true',
                        help='Relabel pre-saccade frames to previous label; discard active saccade frames.')
    parser.add_argument('--ge_plots', action='store_true',
                        help='Show gaze prediction scatter plots for each degree.')
    opt = parser.parse_args()
    main(opt.data_dir, opt.val_subject, opt.relabel, opt.ge_plots)
