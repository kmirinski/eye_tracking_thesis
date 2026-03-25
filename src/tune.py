"""
Hyperparameter tuning for FrameDetectionConfig, scored by gaze model performance.

For each combination of pupil detection parameters, runs the full pipeline
(pupil extraction → valid mask → train/val split → polynomial gaze estimator)
and reports both training and validation mean error.

Usage:
    python src/tune.py --subject 4 --eye left
    python src/tune.py --subject 4 --eye left --relabel
    python src/tune.py --subject 4 --eye left --fov 40 30

Output ends with a copy-pasteable entry for SUBJECT_FRAME_DETECTION_OVERRIDES in config.py.
"""

import argparse
import os
import sys
from dataclasses import asdict
from itertools import product

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import FrameDetectionConfig, get_frame_detection_config, get_gaze_config
from data.loaders import EyeDataset
from models.polynomial import GazeEstimator
from pipeline.pipeline import (
    build_valid_mask, noise_flagging_stage,
    pupil_extraction_stage, relabeling_stage,
)
from pipeline.runners import split_by_label, fov_filter_mask


FRAME_DETECTION_GRID = {
    "threshold":         [10, 15, 20],
    "morph_kernel_size": [2, 3, 4],
    "min_aspect_ratio":  [0.25, 0.32],
}


def tune(subject, eye, data_dir, fov=None, fov_center=None, relabel=False):
    eye_index    = 0 if eye == 'left' else 1
    base_config  = get_frame_detection_config(subject, eye)
    gaze_config  = get_gaze_config(subject)

    eye_dataset = EyeDataset(data_dir, subject, mode='stack')
    eye_dataset.collect_data(eye=eye_index)

    keys   = list(FRAME_DETECTION_GRID.keys())
    combos = list(product(*FRAME_DETECTION_GRID.values()))

    print(f"\n[Subject {subject} | {eye} eye | {len(combos)} combos]\n")
    print(f"  {'#':>4}  {'train_err':>10}  {'val_err':>10}  params")
    print("  " + "-" * 72)

    best_val_err, best_overrides = float('inf'), {}

    for i, values in enumerate(combos):
        overrides = dict(zip(keys, values))
        frame_config = FrameDetectionConfig(
            triangle_corner=base_config.triangle_corner,
            triangle_size=base_config.triangle_size,
            **overrides,
        )

        pupil_centers, _, screen_coords = pupil_extraction_stage(eye_dataset, frame_config)
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

        pc  = np.round(pupil_centers[valid_mask], 2)
        scv = np.round(sc[valid_mask], 2)

        if fov is not None:
            fov_mask = fov_filter_mask(scv, fov[0], fov[1], gaze_config, center=fov_center)
            pc, scv = pc[fov_mask], scv[fov_mask]

        if len(pc) < 10:
            print(f"  [{i+1:>4}/{len(combos)}] skipped (too few samples)  {overrides}")
            continue

        pt, pv, st, sv = split_by_label(pc, scv, val_ratio=gaze_config.val_ratio)
        ge = GazeEstimator(degree=6)
        ge.fit(pt, st)

        train_err = ge.evaluate(pt, st)['mean_error']
        val_err   = ge.evaluate(pv, sv)['mean_error']
        marker = ' *' if val_err < best_val_err else ''
        print(f"  [{i+1:>4}/{len(combos)}] train={train_err:7.2f}px  val={val_err:7.2f}px{marker}  {overrides}")

        if val_err < best_val_err:
            best_val_err, best_overrides = val_err, overrides

    print(f"\nBest val mean error: {best_val_err:.2f} px")
    print(f"Paste into SUBJECT_FRAME_DETECTION_OVERRIDES in config.py:")
    print(f"    {subject}: {best_overrides},")
    return best_overrides


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject',  type=int, default=22)
    parser.add_argument('--eye',      default='left', choices=['left', 'right'])
    parser.add_argument('--data_dir', default=os.path.join(os.getcwd(), 'eye_data'))
    parser.add_argument('--fov',        type=float, nargs=2, default=None, metavar=('WIDTH_DEG', 'HEIGHT_DEG'))
    parser.add_argument('--fov_center', type=float, nargs=2, default=None, metavar=('ROW', 'COL'))
    parser.add_argument('--relabel',    action='store_true')
    opt = parser.parse_args()

    tune(opt.subject, opt.eye, opt.data_dir, fov=opt.fov, fov_center=opt.fov_center, relabel=opt.relabel)
