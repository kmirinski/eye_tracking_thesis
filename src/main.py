import argparse
import os
import matplotlib.pyplot as plt

from preprocessing.accumulator import accumulate_events
from preprocessing.denoise import filter_noise
from data.loaders import EyeDataset, Event
from typing import List, Any


parser = argparse.ArgumentParser(description='Arguments for reading the data')
parser.add_argument('--subject', type=int, default=22, help='choose subject')
parser.add_argument('--eye', default='left', choices=['left', 'right'],
                    help='choose left or right eye dataset')
parser.add_argument('--data_dir', default=os.path.join(os.getcwd(), 'eye_data'),
                    help='absolute path to eye_data/, assumes same parent dir as this script by default')
opt = parser.parse_args()

def plot_on_axis(event_params: List[Any], ax, title):
    negative_x, negative_y = [], []
    positive_x, positive_y = [], []
    
    for i in range(0, len(event_params), 4):
        polarity = event_params[i]
        row = event_params[i + 1]
        col = event_params[i + 2]
        
        if polarity == 0:
            negative_x.append(col)
            negative_y.append(row)
        else:
            positive_x.append(col)
            positive_y.append(row)
    
    ax.scatter(negative_x, negative_y, c='red', s=12, label='Polarity 0', alpha=0.5)
    ax.scatter(positive_x, positive_y, c='green', s=12, label='Polarity 1', alpha=0.5)
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
    eye_dataset.collect_data(eye=0)
    event_sets = accumulate_events(eye_dataset.event_list, n_events=2000) # Add this to args later (for now hardcoded)
    negative = event_sets['negative_polarity']
    positive = event_sets['positive_polarity']
    combined = event_sets['combined_polarity']
    negative_denoised = filter_noise(negative, box_size=3, threshold=1) # Must be tweeked a bit
    positive_denoised = filter_noise(positive, box_size=3, threshold=1) # Can be added to args

    # Just for testing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=200)
    plot_on_axis(positive[0], ax1, 'Original Data')
    plot_on_axis(positive_denoised[0], ax2, 'Denoised Data')
    plt.tight_layout()
    plt.show()
    
    
    

if __name__ == '__main__':
    main()    