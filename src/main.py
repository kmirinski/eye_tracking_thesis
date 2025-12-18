import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

from processing.preprocessing import *
from processing.denoise import box_filter_events
from data.plot import plot_event_set, plot_event_image
from utils import *
from data.loaders import EyeDataset


parser = argparse.ArgumentParser(description='Arguments for reading the data')
parser.add_argument('--subject', type=int, default=22, help='choose subject')
parser.add_argument('--eye', default='left', choices=['left', 'right'],
                    help='choose left or right eye dataset')
parser.add_argument('--data_dir', default=os.path.join(os.getcwd(), 'eye_data'),
                    help='absolute path to eye_data/, assumes same parent dir as this script by default')
opt = parser.parse_args()






def main():
    eye_dataset = EyeDataset(opt.data_dir, opt.subject)
    print('Collecting data of the left eye of subject ' + str(opt.subject))
    print('Loading data from ' + opt.data_dir)
    
    with timer("Collection + Accumulation"):
        eye_dataset.collect_data(eye=0)
        event_sets = accumulate_events(eye_dataset.event_list, n_events=2000) # Add this to args later (for now hardcoded)

    with timer("Positive + Negative"):
        neg_sets = extract_polarity_sets(event_sets, 0)
        pos_sets = extract_polarity_sets(event_sets, 1)

    with timer("Filtering Positive + Negative"):
        neg_filtered = box_filter_events(neg_sets, box_size=6, threshold=4)
        pos_filtered = box_filter_events(pos_sets, box_size=6, threshold=4)

    # Plotting
    _, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=200)
    ax1, ax2, ax3, ax4 = axes.flatten()

    plot_event_set(pos_sets[0], ax1, 'Original Data')
    plot_event_set(pos_filtered[0], ax2, 'Denoised Data')
    plot_event_set(neg_sets[0], ax3, 'Original Data')
    plot_event_set(neg_filtered[0], ax4, 'Denoised Data')
    plt.tight_layout()
    plt.show()
    
    

if __name__ == '__main__':
    main()    