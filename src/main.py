import argparse
import os

from pipeline.pipeline import run_pipeline


parser = argparse.ArgumentParser(description='Arguments for reading the data')

parser.add_argument('--subject', type=int, default=22, help='choose subject')
parser.add_argument('--eye', default='left', choices=['left', 'right'],
                    help='choose left or right eye dataset')
parser.add_argument('--data_dir', default=os.path.join(os.getcwd(), 'eye_data'),
                    help='absolute path to eye_data/, assumes same parent dir as this script by default')
parser.add_argument('--model', default="regressor", choices=['regressor', 'lstm'],
                    help='choose model type to estimate gaze')

# Boolean flags
parser.add_argument('--video', action='store_true')
parser.add_argument('--f_browse', action='store_true')
parser.add_argument('--pe_browse', action='store_true',
                    help='interactive browser showing pupil extraction stages for each frame')
parser.add_argument("--pe_plots", action='store_true')
parser.add_argument('--ge_plots', action='store_true')
parser.add_argument('--relabel', action='store_true',
                    help='relabel pre-saccade frames to previous label; discard active saccade frames')
parser.add_argument('--diff_plot', action='store_true',
                    help='plot frame-to-frame pupil displacement over time with label-change markers')
parser.add_argument('--relabel_diag', action='store_true',
                    help='plot relabeling phase diagnostic (requires --relabel)')
parser.add_argument('--fov', type=float, nargs=2, metavar=('WIDTH_DEG', 'HEIGHT_DEG'),
                    default=None,
                    help='restrict training to a centered FoV window in degrees, e.g. --fov 20 40')



if __name__ == '__main__':
    opt = parser.parse_args()
    run_pipeline(opt)
