import argparse
import os

from pipeline.pipeline import run_pipeline


parser = argparse.ArgumentParser(description='Arguments for reading the data')

# General options
parser.add_argument('--subject', type=int, default=22, help='choose subject')
parser.add_argument('--eye', default='left', choices=['left', 'right'],
                    help='choose left or right eye dataset')
parser.add_argument('--data_dir', default=os.path.join(os.getcwd(), 'eye_data'),
                    help='absolute path to eye_data/, assumes same parent dir as this script by default')
parser.add_argument('--model', default="regressor", choices=['regressor', 'lstm'],
                    help='choose model type to estimate gaze')


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


if __name__ == '__main__':
    opt = parser.parse_args()
    if opt.cross_subject:
        if opt.model == 'lstm':
            from cross_subject_lstm import run
        else:
            from cross_subject_regressor import run
        run(opt)
    else:
        run_pipeline(opt)
