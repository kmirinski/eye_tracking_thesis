import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

from preprocessing.accumulator import accumulate_events, extract_polarity_sets
from preprocessing.denoise import box_filter_events
from utils import *
from data.loaders import EyeDataset


parser = argparse.ArgumentParser(description='Arguments for reading the data')
parser.add_argument('--subject', type=int, default=22, help='choose subject')
parser.add_argument('--eye', default='left', choices=['left', 'right'],
                    help='choose left or right eye dataset')
parser.add_argument('--data_dir', default=os.path.join(os.getcwd(), 'eye_data'),
                    help='absolute path to eye_data/, assumes same parent dir as this script by default')
opt = parser.parse_args()


def plot_event_set(event_set: np.ndarray, ax, title, img_width=346, img_height=260):
    """
    Plot a single event set.
    
    Args:
        event_set: Event set (n_events, 4) with [polarity, row, col, timestamp]
        ax: Matplotlib axis
        title: Plot title
        img_width: Width of the image frame
        img_height: Height of the image frame
    """
    pols = event_set[:, 0]
    rows = event_set[:, 1]
    cols = event_set[:, 2]

    neg_mask = pols == 0
    pos_mask = pols == 1
    
    if neg_mask.any():
        ax.scatter(cols[neg_mask], rows[neg_mask], c='red', s=12, label='Polarity 0', alpha=0.5)
    if pos_mask.any():
        ax.scatter(cols[pos_mask], rows[pos_mask], c='green', s=12, label='Polarity 1', alpha=0.5)
    
    ax.set_xlabel('Column (x)')
    ax.set_xlim(0, img_width)
    ax.set_ylabel('Row (y)')
    ax.set_ylim(0, img_height)
    ax.set_title(title)
    ax.legend()
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)



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

     # Just for testing
    _, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=200)
    ax1, ax2, ax3, ax4 = axes.flatten()

    plot_event_set(pos_sets[50], ax1, 'Original Data')
    plot_event_set(pos_filtered[50], ax2, 'Denoised Data')
    plot_event_set(neg_sets[50], ax3, 'Original Data')
    plot_event_set(neg_filtered[50], ax4, 'Denoised Data')
    plt.tight_layout()
    plt.show()
    
    

if __name__ == '__main__':
    main()    