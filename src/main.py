import argparse
import os

from pipeline.pipeline import run_pipeline
from config import DATASET_PATHS


parser = argparse.ArgumentParser(description='Arguments for reading the data')

# General options
parser.add_argument('--subject', type=int, default=22, help='choose subject')
parser.add_argument('--eye', default='left', choices=['left', 'right'],
                    help='choose left or right eye dataset')
parser.add_argument('--dataset', default='ebveye', choices=list(DATASET_PATHS.keys()),
                    help='dataset name; determines data path and loader')
parser.add_argument('--motion', default='saccadic', choices=['saccadic', 'pursuit'],
                    help='motion type to load: saccadic or smooth pursuit')
parser.add_argument('--model', default="regressor", choices=['regressor', 'lstm'],
                    help='choose model type to estimate gaze')
parser.add_argument('--frame_only', action='store_true',
                    help='regressor: train/eval on frame pupil detections only, excluding event ellipses')
parser.add_argument('--events_eval', action='store_true',
                    help='regressor: calibrate on frame centers, then evaluate gaze DoD on the '
                         'high-frequency event-tracked centers (high-frequency gaze protocol)')
parser.add_argument('--eval_split', default='blocks', choices=['blocks', 'within', 'parity'],
                    help='regressor --events_eval split protocol: "blocks" assigns whole time '
                         'blocks to calibration vs eval (leakage-free); "within" takes train_ratio '
                         'of frames from every block to calibrate and evaluates on the remaining '
                         'frames from each block plus all events; "parity" calibrates on the '
                         'odd-positioned frames (chronological order) and evaluates on the rest '
                         'plus all events (fixed 50/50, ignores val_ratio)')
parser.add_argument('--good_anchor_thresh', type=float, nargs='+', default=[5.0],
                    help='events_eval: anchor-residual threshold(s) in degrees defining the '
                         '"GOOD frames" event subset. Pass multiple values to sweep, e.g. '
                         '--good_anchor_thresh 5 7 9 11 13')
parser.add_argument('--val_ratio_sweep', type=float, nargs='+', default=None,
                    help='events_eval (blocks mode): val_ratio value(s) controlling the '
                         'calibration/evaluation set-size split (fraction of time blocks held out '
                         'for eval). Pass multiple to sweep, e.g. --val_ratio_sweep 0.1 0.2 0.3 0.4 0.5')


# Debug/Inspect options
parser.add_argument('--video', action='store_true')
parser.add_argument('--f_browse', action='store_true')
parser.add_argument('--pe_browse', action='store_true',
                    help='interactive browser showing pupil extraction stages for each frame')
parser.add_argument("--pe_plots", action='store_true')
parser.add_argument('--ge_plots', action='store_true')
parser.add_argument('--relabel_diag', action='store_true',
                    help='plot relabeling phase diagnostic (requires --relabel)')
parser.add_argument('--diff_plot', action='store_true',
                    help='plot frame-to-frame pupil displacement over time with label-change markers')
parser.add_argument('--event_diag', action='store_true',
                    help='plot event extraction diagnostic: ellipse centres over time + size distributions (requires --model lstm)')
parser.add_argument('--loss_plot', action='store_true',
                    help='plot training vs validation loss curve after LSTM training')
parser.add_argument('--no_event_filter', action='store_true',
                    help='disable event outlier filtering (residual/blink/drift gates) in template tracking')


# Relabeling options
parser.add_argument('--relabel', action='store_true',
                    help='relabel pre-saccade frames to previous label; discard active saccade frames')


# FoV options
parser.add_argument('--fov', type=float, nargs=2, metavar=('WIDTH_DEG', 'HEIGHT_DEG'),
                    default=None,
                    help='restrict training to a centered FoV window in degrees, e.g. --fov 40 20')
parser.add_argument('--fov_center', type=float, nargs=2, metavar=('ROW', 'COL'),
                    default=None,
                    help='center of FoV window in screen pixels (row col); defaults to screen center')


# Cross-subject options
parser.add_argument('--cross_subject', action='store_true',
                    help='run leave-one-out cross-subject evaluation instead of single-subject pipeline')
parser.add_argument('--val_subject', type=int, default=None,
                    help='subject to hold out for evaluation (cross-subject mode only); if omitted, runs full LOO')
parser.add_argument('--fine_tune', action='store_true',
                    help='fine-tune the cross-subject LSTM on a small portion of the val subject\'s data')
parser.add_argument('--lstm_events', action='store_true',
                    help='cross-subject LSTM: include event ellipses (frame+event) instead of frame-only')
parser.add_argument('--preprocess_only', action='store_true',
                    help='cross-subject: warm every subject cache, then exit before training (for cluster pre-warming)')


if __name__ == '__main__':
    opt = parser.parse_args()
    opt.data_dir = DATASET_PATHS[opt.dataset]
    if opt.cross_subject:
        if opt.model == 'lstm':
            from cross_subject_lstm import run
        else:
            from cross_subject_regressor import run
        run(opt)
    else:
        run_pipeline(opt)
