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

from config import GazeConfig, LSTMConfig, get_frame_detection_config, get_gaze_config
from data.loaders import EyeDataset, EvEyeDataset
from data.visualization import plot_gaze_predictions, plot_training_history
from models.lstm import LSTMGazeEstimator, build_lstm_sequences
from pipeline.pipeline import (
    build_valid_mask, noise_flagging_stage,
    pupil_extraction_stage, relabeling_stage,
)
from pipeline.runners import fov_filter_mask

SUBJECTS = {
    'ebveye': [4, 5, 6, 7, 11, 12, 15, 18, 19, 21, 22],
    'ev_eye': [4, 5, 6, 7, 8, 33, 34, 35, 36, 44],
}
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
                      eye='left', dataset='ebveye', motion='saccadic'):
    cache_path = os.path.join(CACHE_DIR, f'{dataset}_subject_{subject}_{eye}_lstm.npz')
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
    sc, saccade_mask = relabeling_stage(pupil_centers, screen_coords, gaze_config)
    skip_label_changes = (dataset == 'ebveye')
    valid_mask = build_valid_mask(
        blink_mask, sc,
        skip_frames=gaze_config.saccade_skip_frames,
        saccade_mask=saccade_mask,
        skip_label_changes=skip_label_changes,
        post_blink_skip_frames=gaze_config.post_blink_skip_frames,
    )

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
             fine_tune=False, loss_plot=False, eye='left', dataset='ebveye', motion='saccadic'):
    X_val, y_val = load_subject_data(val_subject, data_dir, fov, fov_center,
                                     eye=eye, dataset=dataset, motion=motion)

    X_parts, y_parts = [], []
    for s in subjects:
        if s == val_subject:
            continue
        X_s, y_s = load_subject_data(s, data_dir, fov, fov_center,
                                     eye=eye, dataset=dataset, motion=motion)
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

    if fine_tune:
        # Stratified sampling: take fine_tune_ratio fraction of each unique label's sequences
        fine_tune_ratio = GazeConfig().fine_tune_ratio
        unique_labels = np.unique(y_val, axis=0)
        ft_indices = []
        for label in unique_labels:
            label_idx = np.where(np.all(y_val == label, axis=1))[0]
            n_sample = max(1, int(len(label_idx) * fine_tune_ratio))
            ft_indices.extend(np.random.choice(label_idx, n_sample, replace=False))
        ft_indices = np.array(ft_indices)
        eval_indices = np.setdiff1d(np.arange(len(X_val)), ft_indices)

        X_ft,   y_ft   = X_val[ft_indices],  y_val[ft_indices]
        X_eval, y_eval = X_val[eval_indices], y_val[eval_indices]
        print(f"Fine-tuning on {len(ft_indices)} sequences from subject {val_subject} "
              f"({len(unique_labels)} labels × ~{fine_tune_ratio*100:.0f}% each)...")
        estimator.fine_tune(X_ft, y_ft)
    else:
        X_eval, y_eval = X_val, y_val

    metrics = estimator.evaluate(X_eval, y_eval)
    print(f"Subject {val_subject} val — mse={metrics['mse']:.5f}px²  mean={metrics['mean_error']:.5f}px  rmse={metrics['rmse']:.5f}px")

    if ge_plots:
        eval_pred = estimator.predict(X_eval)
        plot_gaze_predictions(
            eval_pred, y_eval,
            title=f'LSTM — Subject {val_subject}',
            fov_rect=_fov_rect(fov, fov_center),
        )

    return metrics


def main(data_dir, val_subject, ge_plots, fov, fov_center,
         fine_tune=False, loss_plot=False, eye='left', dataset='ebveye', motion='saccadic'):
    subjects = SUBJECTS[dataset]

    print("=" * 60)
    print(f"Preprocessing / loading subjects ({dataset})...")
    print("=" * 60)
    for s in subjects:
        load_subject_data(s, data_dir, fov, fov_center,
                          eye=eye, dataset=dataset, motion=motion)

    if val_subject is not None:
        if val_subject not in subjects:
            raise ValueError(f"--val_subject {val_subject} is not in SUBJECTS list for {dataset}: {subjects}")
        print()
        print("=" * 60)
        print(f"Fold: val = subject {val_subject}" + (" (with fine-tuning)" if fine_tune else ""))
        print("=" * 60)
        run_fold(val_subject, subjects, data_dir, ge_plots, fov, fov_center,
                 fine_tune=fine_tune, loss_plot=loss_plot,
                 eye=eye, dataset=dataset, motion=motion)
        return

    results = {}
    for s in subjects:
        print()
        print("=" * 60)
        print(f"Fold: val = subject {s}" + (" (with fine-tuning)" if fine_tune else ""))
        print("=" * 60)
        results[s] = run_fold(s, subjects, data_dir, ge_plots, fov, fov_center,
                              fine_tune=fine_tune, loss_plot=loss_plot,
                              eye=eye, dataset=dataset, motion=motion)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for s in subjects:
        m = results[s]
        print(f"  {s:>3}: mean={m['mean_error']:.5f}px  rmse={m['rmse']:.5f}px  "
              f"median={m['median_error']:.5f}px  std={m['std_error']:.5f}px")
    mean_errors = [results[s]['mean_error'] for s in subjects]
    print(f"\nOverall mean error: {np.mean(mean_errors):.5f} ± {np.std(mean_errors):.5f} px")


def run(opt):
    dataset = getattr(opt, 'dataset', 'ebveye')
    motion = getattr(opt, 'motion', 'saccadic')
    fov = tuple(opt.fov) if opt.fov else None
    fov_center = tuple(opt.fov_center) if opt.fov_center else None
    main(opt.data_dir, opt.val_subject, opt.ge_plots, fov, fov_center,
         fine_tune=getattr(opt, 'fine_tune', False),
         loss_plot=getattr(opt, 'loss_plot', False),
         eye=opt.eye, dataset=dataset, motion=motion)
