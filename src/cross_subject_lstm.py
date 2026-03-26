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
from data.loaders import EyeDataset
from data.visualization import plot_gaze_predictions
from models.lstm import LSTMGazeEstimator, build_lstm_sequences
from pipeline.pipeline import (
    build_valid_mask, noise_flagging_stage,
    pupil_extraction_stage, relabeling_stage,
)
from pipeline.runners import fov_filter_mask

SUBJECTS   = [4, 5, 6, 7, 11, 12, 15, 18, 19, 21, 22]
EYE        = 'left'
FOV        = (40.0, 20.0)
FOV_CENTER = (501, 879)          # set to (row_px, col_px) to shift the FoV window
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


def load_subject_data(subject, data_dir, fov, fov_center):
    cache_path = os.path.join(CACHE_DIR, f'subject_{subject}_{EYE}_scaled.npz')
    if os.path.exists(cache_path):
        print(f"  Subject {subject}: loading from cache")
        data = np.load(cache_path)
        return data['X'], data['y']

    print(f"  Subject {subject}: preprocessing...")
    eye_index    = 0 if EYE == 'left' else 1
    frame_config = get_frame_detection_config(subject, EYE)
    gaze_config  = get_gaze_config(subject)
    lstm_config  = LSTMConfig()

    eye_dataset = EyeDataset(data_dir, subject, mode='stack')
    eye_dataset.collect_data(eye=eye_index)

    pupil_centers, ellipses, screen_coords = pupil_extraction_stage(eye_dataset, frame_config)
    blink_mask = noise_flagging_stage(pupil_centers)
    sc, saccade_mask = relabeling_stage(pupil_centers, screen_coords, gaze_config)
    valid_mask = build_valid_mask(
        blink_mask, sc,
        skip_frames=gaze_config.saccade_skip_frames,
        saccade_mask=saccade_mask,
        skip_label_changes=False,
        post_blink_skip_frames=gaze_config.post_blink_skip_frames,
    )

    X, y = build_lstm_sequences(ellipses, sc, valid_mask, seq_len=lstm_config.seq_len)

    fov_mask = fov_filter_mask(y, fov[0], fov[1], gaze_config, center=fov_center)
    X, y = X[fov_mask], y[fov_mask]

    # Per-subject standardization of 21D vectors (as specified in the paper)
    n, s, f = X.shape
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, f)).reshape(n, s, f)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache_path, X=X, y=y)
    print(f"  Subject {subject}: {len(X)} sequences — cached to {cache_path}")
    return X, y


def run_fold(val_subject, data_dir, ge_plots, fov, fov_center):
    X_val, y_val = load_subject_data(val_subject, data_dir, fov, fov_center)

    X_parts, y_parts = [], []
    for s in SUBJECTS:
        if s == val_subject:
            continue
        X_s, y_s = load_subject_data(s, data_dir, fov, fov_center)
        X_parts.append(X_s)
        y_parts.append(y_s)
    X_train = np.concatenate(X_parts)
    y_train = np.concatenate(y_parts)
    del X_parts, y_parts

    print(f"Train: {len(X_train)} sequences  |  Val: {len(X_val)} sequences")

    estimator = LSTMGazeEstimator(LSTMConfig(), pre_scaled=True)
    estimator.fit(X_train, y_train, X_val, y_val)
    del X_train, y_train

    metrics = estimator.evaluate(X_val, y_val)
    print(f"Subject {val_subject} val — mean={metrics['mean_error']:.2f}px  rmse={metrics['rmse']:.2f}px")

    if ge_plots:
        val_pred = estimator.predict(X_val)
        plot_gaze_predictions(
            val_pred, y_val,
            title=f'LSTM — Subject {val_subject}',
            fov_rect=_fov_rect(fov, fov_center),
        )

    return metrics


def main(data_dir, val_subject, ge_plots, fov=FOV, fov_center=FOV_CENTER):
    # Warm up cache for all subjects before running folds
    print("=" * 60)
    print("Preprocessing / loading subjects...")
    print("=" * 60)
    for s in SUBJECTS:
        load_subject_data(s, data_dir, fov, fov_center)

    if val_subject is not None:
        if val_subject not in SUBJECTS:
            raise ValueError(f"--val_subject {val_subject} is not in SUBJECTS list: {SUBJECTS}")
        print()
        print("=" * 60)
        print(f"Fold: val = subject {val_subject}")
        print("=" * 60)
        run_fold(val_subject, data_dir, ge_plots, fov, fov_center)
        return

    results = {}
    for s in SUBJECTS:
        print()
        print("=" * 60)
        print(f"Fold: val = subject {s}")
        print("=" * 60)
        results[s] = run_fold(s, data_dir, ge_plots, fov, fov_center)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for s in SUBJECTS:
        m = results[s]
        print(f"  {s:>3}: mean={m['mean_error']:.2f}px  rmse={m['rmse']:.2f}px  "
              f"median={m['median_error']:.2f}px  std={m['std_error']:.2f}px")
    mean_errors = [results[s]['mean_error'] for s in SUBJECTS]
    print(f"\nOverall mean error: {np.mean(mean_errors):.2f} ± {np.std(mean_errors):.2f} px")


def run(opt):
    fov = tuple(opt.fov) if opt.fov else FOV
    fov_center = tuple(opt.fov_center) if opt.fov_center else FOV_CENTER
    main(opt.data_dir, opt.val_subject, opt.ge_plots, fov, fov_center)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=os.path.join(os.getcwd(), 'eye_data'))
    parser.add_argument('--val_subject', type=int, default=None,
                        help='Subject to hold out for evaluation. If omitted, runs full LOO.')
    parser.add_argument('--ge_plots', action='store_true',
                        help='Show gaze prediction scatter and heatmap plots after each fold.')
    opt = parser.parse_args()
    main(opt.data_dir, opt.val_subject, opt.ge_plots)
