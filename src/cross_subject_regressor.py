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
    python src/main.py --cross_subject --model regressor
    python src/main.py --cross_subject --model regressor --val_subject 22 --relabel --ge_plots
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import GazeConfig, TemplateTrackingConfig, get_frame_detection_config, get_gaze_config
from data.loaders import EyeDataset, EvEyeDataset
from data.visualization import plot_gaze_predictions
from models.polynomial import GazeEstimator
from pipeline.pipeline import (
    build_valid_mask,
    noise_flagging_stage,
    pupil_extraction_stage,
    relabeling_stage,
    template_tracking_stage,
)
from pipeline.runners import fov_filter_mask, _fov_rect
from processing.normalization import compute_pupil_stats, normalize_pupils

SUBJECTS = {
    'ebveye': [4, 5, 6, 7, 11, 12, 15, 18, 19, 22],
    'ev_eye': [4, 5, 6, 7, 8, 33, 34, 35, 36, 44],
}
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_cache')



def load_subject_data(subject, data_dir, eye, relabel, fov, fov_center,
                      dataset='ebveye', motion='saccadic'):
    """
    Run the full preprocessing pipeline for one subject and return filtered
    raw (unnormalized) pupil_centers and screen_coords.

    Results are cached to CACHE_DIR so subsequent runs skip re-extraction.
    The cache key includes dataset and relabel since they affect output.
    """
    relabel_tag = 'rel1' if relabel else 'rel0'
    cache_path = os.path.join(CACHE_DIR, f'{dataset}_subject_{subject}_{eye}_regressor_{relabel_tag}.npz')
    if os.path.exists(cache_path):
        print(f"  Subject {subject}: loading from cache")
        data = np.load(cache_path)
        return data['pupil_centers'], data['screen_coords']

    print(f"Subject {subject}: preprocessing...")
    frame_config = get_frame_detection_config(subject, eye, dataset=dataset)
    gaze_config = get_gaze_config(subject)

    if dataset == 'ev_eye':
        eye_dataset = EvEyeDataset(
            data_dir, subject, motion=motion, mode='np',
        )
        eye_dataset.collect_data(eye=eye)
    else:
        eye_index = 0 if eye == 'left' else 1
        eye_dataset = EyeDataset(data_dir, subject, mode='stack')
        eye_dataset.collect_data(eye=eye_index, motion=motion)

    pupil_centers, ellipses, screen_coords = pupil_extraction_stage(eye_dataset, frame_config)
    blink_mask = noise_flagging_stage(pupil_centers)

    if relabel:
        sc, saccade_mask = relabeling_stage(pupil_centers, screen_coords, gaze_config)
    else:
        sc, saccade_mask = screen_coords, None

    skip_label_changes = (dataset == 'ebveye') and not relabel
    valid_mask = build_valid_mask(
        blink_mask, sc,
        skip_frames=gaze_config.saccade_skip_frames,
        saccade_mask=saccade_mask,
        skip_label_changes=skip_label_changes,
        post_blink_skip_frames=gaze_config.post_blink_skip_frames,
    )

    events_np = eye_dataset.load_events_sorted(eye if dataset == 'ev_eye' else None)
    tt_config = TemplateTrackingConfig()
    event_samples = template_tracking_stage(
        events_np, eye_dataset.frame_list, ellipses, sc, valid_mask, tt_config,
    )

    pupil_centers = np.round(pupil_centers[valid_mask], 2)
    screen_coords = np.round(sc[valid_mask], 2)

    if event_samples:
        ev_centers = np.array([[s['ellipse'][0][0], s['ellipse'][0][1]]
                               for s in event_samples], dtype=np.float32)
        ev_labels = np.array([s['screen_coord'] for s in event_samples], dtype=np.float64)
        valid_ev = ~np.all(ev_labels == 0, axis=1)
        ev_centers = ev_centers[valid_ev]
        ev_labels = ev_labels[valid_ev]
        pupil_centers = np.vstack([pupil_centers, ev_centers])
        screen_coords = np.vstack([screen_coords, ev_labels])
        print(f"  Added {valid_ev.sum()} event ellipses → total samples: {len(pupil_centers)}")

    if fov is not None:
        fov_mask = fov_filter_mask(screen_coords, fov[0], fov[1], gaze_config, center=fov_center)
        pupil_centers = pupil_centers[fov_mask]
        screen_coords = screen_coords[fov_mask]

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache_path, pupil_centers=pupil_centers, screen_coords=screen_coords)
    print(f"  Subject {subject}: {len(pupil_centers)} samples — cached to {cache_path}")
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
        print(f"  mse={metrics['mse']:.5f}px²  mean={metrics['mean_error']:.5f}px  "
              f"rmse={metrics['rmse']:.5f}px  median={metrics['median_error']:.5f}px")

        if ge_plots:
            val_pred = estimator.predict(pupil_val)
            plot_gaze_predictions(
                val_pred, screen_val,
                title=f'Subject {val_subject} — Degree {deg}',
                fov_rect=_fov_rect(fov, fov_center, gaze_config),
            )

    best_deg = min(results, key=lambda d: results[d]['mean_error'])
    print(f"\n  Best degree: {best_deg}  (mean={results[best_deg]['mean_error']:.5f}px)")
    return results


def run(opt):
    dataset = getattr(opt, 'dataset', 'ebveye')
    motion = getattr(opt, 'motion', 'saccadic')
    subjects = SUBJECTS[dataset]

    fov = tuple(opt.fov) if opt.fov else None
    fov_center = tuple(opt.fov_center) if opt.fov_center else None

    print("=" * 60)
    print(f"Preprocessing / loading subjects ({dataset})...")
    print("=" * 60)
    subject_data = {}
    for s in subjects:
        subject_data[s] = load_subject_data(
            s, opt.data_dir, opt.eye, opt.relabel, fov, fov_center,
            dataset=dataset, motion=motion,
        )

    if opt.val_subject is not None:
        if opt.val_subject not in subjects:
            raise ValueError(f"--val_subject {opt.val_subject} is not in SUBJECTS list for {dataset}: {subjects}")
        print()
        print("=" * 60)
        print(f"Fold: val = subject {opt.val_subject}")
        print("=" * 60)
        run_fold(opt.val_subject, subject_data, opt.ge_plots, fov, fov_center)
        return

    # Full LOO
    all_results = {}
    for s in subjects:
        print()
        print("=" * 60)
        print(f"Fold: val = subject {s}")
        print("=" * 60)
        fold_results = run_fold(s, subject_data, opt.ge_plots, fov, fov_center)
        best_deg = min(fold_results, key=lambda d: fold_results[d]['mean_error'])
        all_results[s] = fold_results[best_deg]

    print()
    print("=" * 60)
    print("Summary (best degree per fold)")
    print("=" * 60)
    for s in subjects:
        m = all_results[s]
        print(f"  {s:>3}: mean={m['mean_error']:.5f}px  rmse={m['rmse']:.5f}px  "
              f"median={m['median_error']:.5f}px  std={m['std_error']:.5f}px")
    mean_errors = [all_results[s]['mean_error'] for s in subjects]
    print(f"\nOverall mean error: {np.mean(mean_errors):.5f} ± {np.std(mean_errors):.5f} px")

