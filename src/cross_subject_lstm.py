"""
Leave-one-out cross-subject LSTM evaluation.

Trains the LSTM on 10 of the 11 subjects and evaluates on the held-out 11th,
repeating for all 11 folds. Reports per-fold and overall mean error.

Preprocessed sequences are cached to data_cache/ so that re-runs skip the
slow pupil-extraction step. Delete data_cache/ to force re-preprocessing.

Usage:
    python src/cross_subject_lstm.py
    python src/cross_subject_lstm.py --data_dir eye_data
    python src/cross_subject_lstm.py --val_subject 22
"""

import argparse
import os
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

from config import (CROSS_SUBJECT_SUBJECTS, GazeConfig, LSTMConfig, TemplateTrackingConfig,
                    get_frame_detection_config, get_gaze_config)
from data.loaders import EyeDataset, EvEyeDataset
from data.visualization import plot_gaze_predictions, plot_training_history
from models.lstm import LSTMGazeEstimator, build_lstm_sequences, build_lstm_sequences_combined
from pipeline.pipeline import (
    build_valid_mask, label_events_from_tobii, merge_frame_event_samples, noise_flagging_stage,
    pupil_extraction_stage, relabeling_stage, template_tracking_stage,
)
from pipeline.runners import fov_filter_mask, errors_to_degrees, angular_dod
from results_io import fold_filename, metrics_row, save_fold

CACHE_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data_cache')


def _fov_rect(fov, fov_center):
    gaze_config = GazeConfig()
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


def load_subject_data(subject, data_dir, fov, fov_center,
                      eye='left', dataset='ebveye', motion='saccadic', combined=False,
                      relabel=False):
    # Relabel only on saccadic ebveye when the flag is set; never on ev_eye (continuous
    # Tobii labels make every frame look like a transition, which discards almost everything).
    do_relabel = relabel and motion == 'saccadic' and dataset != 'ev_eye'
    suffix = 'lstm_combined' if combined else 'lstm'
    relabel_tag = 'rel1' if do_relabel else 'rel0'
    cache_path = os.path.join(CACHE_DIR, f'{dataset}_subject_{subject}_{eye}_{motion}_{suffix}_{relabel_tag}.npz')
    if os.path.exists(cache_path):
        print(f"  Subject {subject}: loading from cache")
        data = np.load(cache_path)
        return data['X'], data['y']

    print(f"  Subject {subject}: preprocessing...")
    frame_config = get_frame_detection_config(subject, eye, dataset=dataset)
    gaze_config  = get_gaze_config(subject)
    lstm_config  = LSTMConfig()

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
    if do_relabel:
        sc, saccade_mask, _ = relabeling_stage(pupil_centers, screen_coords, gaze_config)
    else:
        sc, saccade_mask = screen_coords, None
    skip_label_changes = (dataset == 'ebveye') and (motion == 'saccadic') and not do_relabel
    valid_mask = build_valid_mask(
        blink_mask, sc,
        skip_frames=gaze_config.saccade_skip_frames,
        saccade_mask=saccade_mask,
        skip_label_changes=skip_label_changes,
        post_blink_skip_frames=gaze_config.post_blink_skip_frames,
        alignment_gaps=getattr(eye_dataset, 'alignment_gaps', None),
        max_alignment_gap_us=gaze_config.max_alignment_gap_us,
    )

    if combined:
        events_np = eye_dataset.load_events_sorted(eye if dataset == 'ev_eye' else None)
        event_samples = template_tracking_stage(
            events_np, eye_dataset.frame_list, ellipses, sc, valid_mask, TemplateTrackingConfig())
        if getattr(eye_dataset, 'gaze_records', None) is not None:
            event_samples = label_events_from_tobii(event_samples, eye_dataset.gaze_records)
        merged = merge_frame_event_samples(
            ellipses, sc, valid_mask, eye_dataset.frame_list, event_samples)
        X, y = build_lstm_sequences_combined(merged, seq_len=lstm_config.seq_len)
    else:
        X, y = build_lstm_sequences(ellipses, sc, valid_mask, seq_len=lstm_config.seq_len)

    if fov is not None:
        fov_mask = fov_filter_mask(y, fov[0], fov[1], gaze_config, center=fov_center)
        X, y = X[fov_mask], y[fov_mask]

    n, s, f = X.shape
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, f)).reshape(n, s, f)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache_path, X=X, y=y)
    print(f"  Subject {subject}: {len(X)} sequences — cached to {cache_path}")
    return X, y


def run_fold(val_subject, subjects, data_dir, ge_plots, fov, fov_center,
             fine_tune=False, loss_plot=False, eye='left', dataset='ebveye', motion='saccadic',
             combined=False, relabel=False):
    X_val, y_val = load_subject_data(val_subject, data_dir, fov, fov_center,
                                     eye=eye, dataset=dataset, motion=motion, combined=combined,
                                     relabel=relabel)

    X_parts, y_parts = [], []
    for s in subjects:
        if s == val_subject:
            continue
        X_s, y_s = load_subject_data(s, data_dir, fov, fov_center,
                                     eye=eye, dataset=dataset, motion=motion, combined=combined,
                                     relabel=relabel)
        X_parts.append(X_s)
        y_parts.append(y_s)
    X_train = np.concatenate(X_parts)
    y_train = np.concatenate(y_parts)
    del X_parts, y_parts

    print(f"Train: {len(X_train)} sequences  |  Val: {len(X_val)} sequences")

    lstm_config = LSTMConfig()
    estimator = LSTMGazeEstimator(lstm_config, pre_scaled=True)
    estimator.fit(X_train, y_train, X_val, y_val)
    del X_train, y_train

    if loss_plot:
        plot_training_history(estimator.history, title=f'LSTM training — val subject {val_subject}')

    normalized = (dataset == 'ev_eye')
    gaze_config = GazeConfig()
    csv_rows = []

    def _eval(X, y, phase):
        """Evaluate, attach DoD, log a one-line summary, and build a CSV row."""
        m = estimator.evaluate(X, y)
        dod_mean, dod_med = angular_dod(estimator.predict(X), y, gaze_config, normalized=normalized)
        m['dod_mean'] = dod_mean
        m['dod_median'] = dod_med
        v_deg, h_deg = errors_to_degrees(m['mean_error_v'], m['mean_error_h'],
                                         gaze_config, normalized=normalized)
        print(f"Subject {val_subject} {phase} — mse={m['mse']:.5f}px²  mean={m['mean_error']:.5f}px  "
              f"rmse={m['rmse']:.5f}px  | h={h_deg:.2f}°  v={v_deg:.2f}°  DoD={dod_mean:.2f}°")
        row = {
            'model': 'lstm', 'dataset': dataset, 'motion': motion, 'eye': eye,
            'val_subject': val_subject, 'fine_tune': int(fine_tune), 'relabel': int(relabel),
            'combined': int(combined), 'degree': '', 'phase': phase, 'n_eval': len(y),
        }
        row.update(metrics_row(m, gaze_config, normalized))
        csv_rows.append(row)
        return m

    ft_indices = np.array([], dtype=int)
    if fine_tune:
        # Stratified sampling: take fine_tune_ratio fraction of each unique label's sequences.
        # Always leave >=1 sequence per label for eval. Frame-only subjects are sparse — many
        # labels have a single sequence, which would otherwise empty the eval split and crash
        # predict() on a zero-length array.
        fine_tune_ratio = gaze_config.fine_tune_ratio
        unique_labels = np.unique(y_val, axis=0)
        ft_list = []
        for label in unique_labels:
            label_idx = np.where(np.all(y_val == label, axis=1))[0]
            if len(label_idx) < 2:
                continue  # too few to split; keep this label entirely for eval
            n_sample = max(1, int(len(label_idx) * fine_tune_ratio))
            n_sample = min(n_sample, len(label_idx) - 1)  # always retain >=1 for eval
            ft_list.extend(np.random.choice(label_idx, n_sample, replace=False))
        ft_indices = np.array(ft_list, dtype=int)
        eval_indices = np.setdiff1d(np.arange(len(X_val)), ft_indices)

    if fine_tune and len(ft_indices) > 0:
        X_ft,   y_ft   = X_val[ft_indices],  y_val[ft_indices]
        X_eval, y_eval = X_val[eval_indices], y_val[eval_indices]

        # Baseline: zero-shot on the held-out eval split, before any fine-tuning.
        _eval(X_eval, y_eval, 'baseline')

        print(f"Fine-tuning on {len(ft_indices)} sequences from subject {val_subject} "
              f"({len(unique_labels)} labels × ~{fine_tune_ratio*100:.0f}% each)...")
        estimator.fine_tune(X_ft, y_ft)
        metrics = _eval(X_eval, y_eval, 'finetuned')
    else:
        # No fine-tuning (or too sparse to split): zero-shot on the full validation subject.
        if fine_tune:
            print(f"Subject {val_subject}: too few sequences per label to fine-tune "
                  f"(no label has >=2); recording baseline on the full val set only.")
        X_eval, y_eval = X_val, y_val
        metrics = _eval(X_eval, y_eval, 'baseline')

    save_fold(csv_rows, fold_filename('lstm', dataset, motion, eye, val_subject,
                                      fine_tune, combined=combined, relabel=relabel))

    if ge_plots:
        eval_pred = estimator.predict(X_eval)
        plot_gaze_predictions(
            eval_pred, y_eval,
            title=f'LSTM — Subject {val_subject}',
            fov_rect=_fov_rect(fov, fov_center),
        )

    return metrics


def main(data_dir, val_subject, ge_plots, fov, fov_center,
         fine_tune=False, loss_plot=False, eye='left', dataset='ebveye', motion='saccadic',
         combined=False, relabel=False, preprocess_only=False):
    subjects = CROSS_SUBJECT_SUBJECTS[dataset]

    print("=" * 60)
    print(f"Preprocessing / loading subjects ({dataset})...")
    print("=" * 60)
    for s in subjects:
        load_subject_data(s, data_dir, fov, fov_center,
                          eye=eye, dataset=dataset, motion=motion, combined=combined,
                          relabel=relabel)

    if preprocess_only:
        print(f"\nPreprocess-only: warmed {len(subjects)} subject caches; exiting before training.")
        return

    if val_subject is not None:
        if val_subject not in subjects:
            raise ValueError(f"--val_subject {val_subject} is not in CROSS_SUBJECT_SUBJECTS list for {dataset}: {subjects}")
        print()
        print("=" * 60)
        print(f"Fold: val = subject {val_subject}" + (" (with fine-tuning)" if fine_tune else ""))
        print("=" * 60)
        run_fold(val_subject, subjects, data_dir, ge_plots, fov, fov_center,
                 fine_tune=fine_tune, loss_plot=loss_plot,
                 eye=eye, dataset=dataset, motion=motion, combined=combined, relabel=relabel)
        return

    results = {}
    for s in subjects:
        print()
        print("=" * 60)
        print(f"Fold: val = subject {s}" + (" (with fine-tuning)" if fine_tune else ""))
        print("=" * 60)
        results[s] = run_fold(s, subjects, data_dir, ge_plots, fov, fov_center,
                              fine_tune=fine_tune, loss_plot=loss_plot,
                              eye=eye, dataset=dataset, motion=motion, combined=combined,
                              relabel=relabel)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    gaze_config = GazeConfig()
    normalized = (dataset == 'ev_eye')
    h_degs, v_degs, dods = [], [], []
    for s in subjects:
        m = results[s]
        v_deg, h_deg = errors_to_degrees(m['mean_error_v'], m['mean_error_h'],
                                         gaze_config, normalized=normalized)
        h_degs.append(h_deg)
        v_degs.append(v_deg)
        dods.append(m['dod_mean'])
        print(f"  {s:>3}: mean={m['mean_error']:.5f}px  rmse={m['rmse']:.5f}px  "
              f"median={m['median_error']:.5f}px  std={m['std_error']:.5f}px  "
              f"| h={h_deg:.2f}°  v={v_deg:.2f}°  DoD={m['dod_mean']:.2f}°")
    mean_errors = [results[s]['mean_error'] for s in subjects]
    print(f"\nOverall mean error: {np.mean(mean_errors):.5f} ± {np.std(mean_errors):.5f} px")
    print(f"Overall per-axis: horizontal {np.mean(h_degs):.2f} ± {np.std(h_degs):.2f}°  |  "
          f"vertical {np.mean(v_degs):.2f} ± {np.std(v_degs):.2f}°")
    print(f"Overall DoD: {np.mean(dods):.2f} ± {np.std(dods):.2f}°")


def run(opt):
    dataset = getattr(opt, 'dataset', 'ebveye')
    motion = getattr(opt, 'motion', 'saccadic')
    fov = tuple(opt.fov) if opt.fov else None
    fov_center = tuple(opt.fov_center) if opt.fov_center else None
    main(opt.data_dir, opt.val_subject, opt.ge_plots, fov, fov_center,
         fine_tune=getattr(opt, 'fine_tune', False),
         loss_plot=getattr(opt, 'loss_plot', False),
         eye=opt.eye, dataset=dataset, motion=motion,
         combined=getattr(opt, 'lstm_events', False),
         relabel=getattr(opt, 'relabel', False),
         preprocess_only=getattr(opt, 'preprocess_only', False))
