import argparse
import os

from pipeline import run_pipeline


parser = argparse.ArgumentParser(description='Arguments for reading the data')

parser.add_argument('--subject', type=int, default=22, help='choose subject')
parser.add_argument('--eye', default='left', choices=['left', 'right'],
                    help='choose left or right eye dataset')
parser.add_argument('--data_dir', default=os.path.join(os.getcwd(), 'eye_data'),
                    help='absolute path to eye_data/, assumes same parent dir as this script by default')

# Boolean flags
parser.add_argument('--video', action='store_true')
parser.add_argument('--frame_browser', action='store_true')
parser.add_argument("--pe_plots", action='store_true')
parser.add_argument('--ge_plots', action='store_true')



if __name__ == '__main__':
    opt = parser.parse_args()
    run_pipeline(opt)
