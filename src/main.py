import argparse
import os
import matplotlib.pyplot as plt

from preprocessing.accumulator import accumulate_events
from preprocessing.denoise import filter_noise
from utils import *
from data.loaders import EyeDataset
from typing import List, Any


parser = argparse.ArgumentParser(description='Arguments for reading the data')
parser.add_argument('--subject', type=int, default=22, help='choose subject')
parser.add_argument('--eye', default='left', choices=['left', 'right'],
                    help='choose left or right eye dataset')
parser.add_argument('--data_dir', default=os.path.join(os.getcwd(), 'eye_data'),
                    help='absolute path to eye_data/, assumes same parent dir as this script by default')
opt = parser.parse_args()


### FIX THIS SHIT PLS
def plot_on_axis(image: np.ndarray, polarities: np.ndarray, ax, title):

    rows, cols = np.where(image > 0)
    pols = image[rows, cols]

    neg_mask = pols == 0
    pos_mask = pols == 1
    
    if neg_mask.any():
        ax.scatter(cols[neg_mask], rows[neg_mask], c='red', s=12, label='Polarity 0', alpha=0.5)
    if pos_mask.any():
        ax.scatter(cols[pos_mask], rows[pos_mask], c='green', s=12, label='Polarity 1', alpha=0.5)
    
    ax.set_xlabel('Column (x)')
    ax.set_xlim(0, 346)
    ax.set_ylabel('Row (y)')
    ax.set_ylim(0, 260)
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
    
    neg_counts = event_sets['negative_counts']
    # return {
    #     'combined_polarity': event_sets,
    #     'negative_polarity': neg_polarity,
    #     'positive_polarity': pos_polarity,
    #     'negative_counts': neg_counts,
    #     'positive_counts': pos_counts
    # }
    imgs, p = events_to_images(event_sets['combined_polarity'], counts=neg_counts)
    imgs_neg, polarities = events_to_images(event_sets['negative_polarity'], counts=neg_counts)
    



    

    # negative_denoised = filter_noise(negative, box_size=6, threshold=4) # Must be tweeked a bit
    # positive_denoised = filter_noise(positive, box_size=6, threshold=4) # Can be added to args
    # combined_denoised = filter_noise(combined, box_size=6, threshold=4)

    # Just for testing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=200)
    plot_on_axis(imgs[0], p, ax1, 'Original Data')
    plot_on_axis(imgs[0], p, ax2, 'Denoised Data')
    plt.tight_layout()
    plt.show()
    
    
    

if __name__ == '__main__':
    main()    