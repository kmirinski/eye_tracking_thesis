import argparse
import os
import matplotlib.pyplot as plt

from preprocessing.accumulator import accumulate_events
from data.loaders import EyeDataset, Event
from typing import List, Any


parser = argparse.ArgumentParser(description='Arguments for reading the data')
parser.add_argument('--subject', type=int, default=22, help='choose subject')
parser.add_argument('--eye', default='left', choices=['left', 'right'],
                    help='choose left or right eye dataset')
parser.add_argument('--data_dir', default=os.path.join(os.getcwd(), 'eye_data'),
                    help='absolute path to eye_data/, assumes same parent dir as this script by default')
opt = parser.parse_args()

def test_plot(event_params: List[Any]):
    negative_x, negative_y = [], []
    positive_x, positive_y = [], []
    
    for i in range(0, len(event_params), 4):
        polarity = event_params[i]
        row = event_params[i + 1]
        col = event_params[i + 2]
        # timestamp = event_params[i + 3]  # if needed
        
        if polarity == 0:
            negative_x.append(col)
            negative_y.append(row)
        else:
            positive_x.append(col)
            positive_y.append(row)

    plt.figure(figsize=(10, 8), dpi=200)
    plt.scatter(negative_x, negative_y, c='red', s=1, label='Polarity 0', alpha=0.6)
    plt.scatter(positive_x, positive_y, c='green', s=1, label='Polarity 1', alpha=0.6)
    plt.xlabel('Column (x)')
    plt.xlim(0, 346)
    plt.ylabel('Row (y)')
    plt.ylim(0, 260)
    plt.title('Event Camera Data')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.show()



def main():
    eye_dataset = EyeDataset(opt.data_dir, opt.subject)
    print('Collecting data of the left eye of subject ' + str(opt.subject))
    print('Loading data from ' + opt.data_dir)
    eye_dataset.collect_data(eye=0)
    event_sets = accumulate_events(eye_dataset.event_list, n_events=2000) # Add this to args later (for now hardcoded)
    # test_plot(event_sets['all_polarity'][0])
    # test_plot(event_sets['negative_polarity'][0])
    # test_plot(event_sets['positive_polarity'][0])
    
    

if __name__ == '__main__':
    main()    