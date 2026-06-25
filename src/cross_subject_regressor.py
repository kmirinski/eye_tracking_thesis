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

from config import (CROSS_SUBJECT_SUBJECTS, GazeConfig, TemplateTrackingConfig,
                    get_frame_detection_config, get_gaze_config)
from data.loaders import EyeDataset, EvEyeDataset
from data.visualization import plot_gaze_predictions
from models.polynomial import GazeEstimator
from pipeline.pipeline import (
    build_valid_mask,
    label_events_from_tobii,
    noise_flagging_stage,
    pupil_extraction_stage,
    relabeling_stage,
    template_tracking_stage,
)
from pipeline.runners import (fov_filter_mask, _fov_rect, split_by_label, split_by_time_blocks,
                              errors_to_degrees, angular_dod, gaze_clip_bounds)
from processing.normalization import compute_pupil_stats, normalize_pupils
from results_io import fold_filename, metrics_row, save_fold, save_accumulated

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_cache')



def load_subject_data(subject, data_dir, eye, relabel, fov, fov_center,
                      dataset='ebveye', motion='saccadic', frame_only=False):
    """
    Run the full preprocessing pipeline for one subject and return filtered
    raw (unnormalized) pupil_centers and screen_coords.

    Results are cached to CACHE_DIR so subsequent runs skip re-extraction.
    The cache key includes dataset, relabel and frame_only since they affect output.
    """
    relabel_tag = 'rel1' if relabel else 'rel0'
    fo_tag = '_frameonly' if frame_only else ''
    cache_path = os.path.join(CACHE_DIR, f'{dataset}_subject_{subject}_{eye}_regressor_{relabel_tag}{fo_tag}.npz')
    if os.path.exists(cache_path):
        print(f"  Subject {subject}: loading from cache")
        data = np.load(cache_path)
        return data['pupil_centers'], data['screen_coords'], data['timestamps']

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

    if relabel and motion == 'saccadic' and dataset != 'ev_eye':
        sc, saccade_mask, _ = relabeling_stage(pupil_centers, screen_coords, gaze_config)
    else:
        sc, saccade_mask = screen_coords, None

    skip_label_changes = (dataset == 'ebveye') and not relabel
    valid_mask = build_valid_mask(
        blink_mask, sc,
        skip_frames=gaze_config.saccade_skip_frames,
        saccade_mask=saccade_mask,
        skip_label_changes=skip_label_changes,
        post_blink_skip_frames=gaze_config.post_blink_skip_frames,
        alignment_gaps=getattr(eye_dataset, 'alignment_gaps', None),
        max_alignment_gap_us=gaze_config.max_alignment_gap_us,
    )

    if frame_only:
        event_samples = []
        print("  Frame-only mode: skipping event extraction.")
    else:
        events_np = eye_dataset.load_events_sorted(eye if dataset == 'ev_eye' else None)
        tt_config = TemplateTrackingConfig()
        event_samples = template_tracking_stage(
            events_np, eye_dataset.frame_list, ellipses, sc, valid_mask, tt_config,
        )
        # ev_eye: label events by nearest Tobii sample in time (+ per-event alignment gap).
        if getattr(eye_dataset, 'gaze_records', None) is not None:
            event_samples = label_events_from_tobii(event_samples, eye_dataset.gaze_records)

    frame_ts = np.array([f.timestamp for f in eye_dataset.frame_list], dtype=np.int64)
    pupil_centers = np.round(pupil_centers[valid_mask], 2)
    screen_coords = np.round(sc[valid_mask], 2)
    timestamps = frame_ts[valid_mask]

    if event_samples:
        ev_centers = np.array([[s['ellipse'][0][0], s['ellipse'][0][1]]
                               for s in event_samples], dtype=np.float32)
        ev_labels = np.array([s['screen_coord'] for s in event_samples], dtype=np.float64)
        ev_ts = np.array([s['timestamp'] for s in event_samples], dtype=np.int64)
        valid_ev = ~np.all(ev_labels == 0, axis=1)
        if event_samples[0].get('gap_us') is not None:
            ev_gaps = np.array([s.get('gap_us', 0) for s in event_samples])
            valid_ev &= ev_gaps <= gaze_config.max_alignment_gap_us
        ev_centers = ev_centers[valid_ev]
        ev_labels = ev_labels[valid_ev]
        pupil_centers = np.vstack([pupil_centers, ev_centers])
        screen_coords = np.vstack([screen_coords, ev_labels])
        timestamps = np.concatenate([timestamps, ev_ts[valid_ev]])
        print(f"  Added {valid_ev.sum()} event ellipses → total samples: {len(pupil_centers)}")

    if fov is not None:
        fov_mask = fov_filter_mask(screen_coords, fov[0], fov[1], gaze_config, center=fov_center)
        pupil_centers = pupil_centers[fov_mask]
        screen_coords = screen_coords[fov_mask]
        timestamps = timestamps[fov_mask]

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache_path, pupil_centers=pupil_centers, screen_coords=screen_coords,
             timestamps=timestamps)
    print(f"  Subject {subject}: {len(pupil_centers)} samples — cached to {cache_path}")
    return pupil_centers, screen_coords, timestamps


def run_fold(val_subject, subject_data, ge_plots, fov, fov_center,
             fine_tune=False, dataset='ebveye', motion='saccadic', eye='left', relabel=False):
    """
    Train on all subjects except val_subject, evaluate on val_subject.
    Per-subject z-score normalization is applied before pooling.

    When fine_tune is set, a fine_tune_ratio fraction of the held-out subject is
    used as calibration. For each polynomial degree both a ``baseline`` model
    (trained on the other subjects only) and a ``finetuned`` model (calibration
    pooled in) are evaluated on the *same* held-out eval split, so the
    calibration gain is directly comparable. Without fine_tune, only a baseline
    is evaluated, on the whole validation subject.
    """
    gaze_config = GazeConfig()
    normalized = (dataset == 'ev_eye')

    # Compute per-subject normalization stats from raw pupils
    stats = {
        s: compute_pupil_stats(pupil_centers)
        for s, (pupil_centers, _, _) in subject_data.items()
    }

    # Build training set: pool normalized pupils from all train subjects
    train_pupils, train_screens = [], []
    for s, (pupil_centers, screen_coords, _) in subject_data.items():
        if s == val_subject:
            continue
        mean, std = stats[s]
        train_pupils.append(normalize_pupils(pupil_centers, mean, std))
        train_screens.append(screen_coords)
    pupil_train = np.concatenate(train_pupils)
    screen_train = np.concatenate(train_screens)

    # Validation set: normalize with val subject's own stats
    val_mean, val_std = stats[val_subject]
    pupil_val, screen_val, ts_val = subject_data[val_subject]
    pupil_val = normalize_pupils(pupil_val, val_mean, val_std)

    if fine_tune:
        # Split the held-out subject: calibration portion is pooled into training,
        # the remainder becomes the evaluation set. ev_eye uses a leakage-free
        # chronological block split (calib/eval are temporally disjoint); ebveye groups
        # by label.
        ft_ratio = gaze_config.fine_tune_ratio
        if dataset == 'ev_eye':
            pupil_calib, pupil_eval, screen_calib, screen_eval = split_by_time_blocks(
                pupil_val, screen_val, ts_val,
                val_ratio=1 - ft_ratio, n_blocks=gaze_config.n_time_blocks,
            )
        else:
            pupil_calib, pupil_eval, screen_calib, screen_eval = split_by_label(
                pupil_val, screen_val, val_ratio=1 - ft_ratio,
            )
        pupil_train_ft = np.concatenate([pupil_train, pupil_calib])
        screen_train_ft = np.concatenate([screen_train, screen_calib])
        print(f"Fine-tuning: pooled {len(pupil_calib)} calibration frames from subject "
              f"{val_subject} (~{ft_ratio*100:.0f}%) into training")
    else:
        pupil_eval, screen_eval = pupil_val, screen_val

    print(f"Train: {len(pupil_train)} frames  |  Eval: {len(pupil_eval)} frames")

    csv_rows = []

    def _eval(estimator, phase, deg):
        """Evaluate on the held-out eval split, attach DoD, log, build a CSV row."""
        m = estimator.evaluate(pupil_eval, screen_eval)
        dod_mean, dod_med = angular_dod(estimator.predict(pupil_eval), screen_eval,
                                        gaze_config, normalized=normalized)
        m['dod_mean'] = dod_mean
        m['dod_median'] = dod_med
        v_deg, h_deg = errors_to_degrees(m['mean_error_v'], m['mean_error_h'],
                                         gaze_config, normalized=normalized)
        print(f"  [{phase}] mse={m['mse']:.5f}px²  mean={m['mean_error']:.5f}px  "
              f"rmse={m['rmse']:.5f}px  median={m['median_error']:.5f}px")
        print(f"    per-axis: h={h_deg:.2f}°  v={v_deg:.2f}°  |  DoD: mean={dod_mean:.2f}°  median={dod_med:.2f}°")
        row = {
            'model': 'regressor', 'dataset': dataset, 'motion': motion, 'eye': eye,
            'val_subject': val_subject, 'fine_tune': int(fine_tune), 'relabel': int(relabel),
            'combined': '', 'degree': deg, 'phase': phase, 'n_eval': len(screen_eval),
        }
        row.update(metrics_row(m, gaze_config, normalized))
        csv_rows.append(row)
        return m

    clip_bounds = gaze_clip_bounds(gaze_config, normalized)
    results = {}
    for deg in gaze_config.poly_degrees:
        print(f"\n  --- Degree {deg} ---")
        baseline = GazeEstimator(degree=deg, clip_bounds=clip_bounds)
        baseline.fit(pupil_train, screen_train)
        base_metrics = _eval(baseline, 'baseline', deg)

        if fine_tune:
            finetuned = GazeEstimator(degree=deg, clip_bounds=clip_bounds)
            finetuned.fit(pupil_train_ft, screen_train_ft)
            results[deg] = _eval(finetuned, 'finetuned', deg)
            best_estimator = finetuned
        else:
            results[deg] = base_metrics
            best_estimator = baseline

        if ge_plots:
            val_pred = best_estimator.predict(pupil_eval)
            plot_gaze_predictions(
                val_pred, screen_eval,
                title=f'Subject {val_subject} — Degree {deg}',
                fov_rect=_fov_rect(fov, fov_center, gaze_config),
            )

    save_fold(csv_rows, fold_filename('regressor', dataset, motion, eye, val_subject,
                                      fine_tune, combined=False, relabel=relabel))

    best_deg = min(results, key=lambda d: results[d]['mean_error'])
    print(f"\n  Best degree: {best_deg}  (mean={results[best_deg]['mean_error']:.5f}px)")
    return results, csv_rows


def run(opt):
    dataset = getattr(opt, 'dataset', 'ebveye')
    motion = getattr(opt, 'motion', 'saccadic')
    fine_tune = getattr(opt, 'fine_tune', False)
    subjects = CROSS_SUBJECT_SUBJECTS[dataset]

    fov = tuple(opt.fov) if opt.fov else None
    fov_center = tuple(opt.fov_center) if opt.fov_center else None

    print("=" * 60)
    print(f"Preprocessing / loading subjects ({dataset})...")
    print("=" * 60)
    frame_only = getattr(opt, 'frame_only', False)
    subject_data = {}
    for s in subjects:
        subject_data[s] = load_subject_data(
            s, opt.data_dir, opt.eye, opt.relabel, fov, fov_center,
            dataset=dataset, motion=motion, frame_only=frame_only,
        )

    if opt.val_subject is not None:
        if opt.val_subject not in subjects:
            raise ValueError(f"--val_subject {opt.val_subject} is not in CROSS_SUBJECT_SUBJECTS list for {dataset}: {subjects}")
        print()
        print("=" * 60)
        print(f"Fold: val = subject {opt.val_subject}" + (" (with fine-tuning)" if fine_tune else ""))
        print("=" * 60)
        run_fold(opt.val_subject, subject_data, opt.ge_plots, fov, fov_center,
                 fine_tune=fine_tune, dataset=dataset, motion=motion, eye=opt.eye,
                 relabel=opt.relabel)
        return

    # Full LOO
    all_results = {}
    accumulated_rows = []
    for s in subjects:
        print()
        print("=" * 60)
        print(f"Fold: val = subject {s}" + (" (with fine-tuning)" if fine_tune else ""))
        print("=" * 60)
        fold_results, fold_rows = run_fold(s, subject_data, opt.ge_plots, fov, fov_center,
                                           fine_tune=fine_tune, dataset=dataset, motion=motion,
                                           eye=opt.eye, relabel=opt.relabel)
        accumulated_rows.extend(fold_rows)
        best_deg = min(fold_results, key=lambda d: fold_results[d]['mean_error'])
        all_results[s] = fold_results[best_deg]

    print()
    print("=" * 60)
    print("Summary (best degree per fold)")
    print("=" * 60)
    gaze_config = GazeConfig()
    normalized = (dataset == 'ev_eye')
    h_degs, v_degs, dods = [], [], []
    for s in subjects:
        m = all_results[s]
        v_deg, h_deg = errors_to_degrees(m['mean_error_v'], m['mean_error_h'],
                                         gaze_config, normalized=normalized)
        h_degs.append(h_deg)
        v_degs.append(v_deg)
        dods.append(m['dod_mean'])
        print(f"  {s:>3}: mean={m['mean_error']:.5f}px  rmse={m['rmse']:.5f}px  "
              f"median={m['median_error']:.5f}px  std={m['std_error']:.5f}px  "
              f"| h={h_deg:.2f}°  v={v_deg:.2f}°  DoD={m['dod_mean']:.2f}°")
    mean_errors = [all_results[s]['mean_error'] for s in subjects]
    print(f"\nOverall mean error: {np.mean(mean_errors):.5f} ± {np.std(mean_errors):.5f} px")
    print(f"Overall per-axis: horizontal {np.mean(h_degs):.2f} ± {np.std(h_degs):.2f}°  |  "
          f"vertical {np.mean(v_degs):.2f} ± {np.std(v_degs):.2f}°")
    print(f"Overall DoD: {np.mean(dods):.2f} ± {np.std(dods):.2f}°")

    # Accumulate every fold of this LOO run into one self-contained CSV at the
    # results root (separate from the aggregator's summary.csv).
    ft = 'ft1' if fine_tune else 'ft0'
    rel = 'rel1' if opt.relabel else 'rel0'
    save_accumulated(
        accumulated_rows,
        f'regressor_{dataset}_{motion}_{opt.eye}_loo_{ft}_{rel}.csv',
    )

