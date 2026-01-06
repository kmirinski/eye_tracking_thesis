import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

from processing.preprocessing import *
from processing.filtering import *
from data.plot import *
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

    img_neg = event_to_image(neg_sets[0])
    img_pos = event_to_image(pos_sets[0])
    img = event_to_image(event_sets[0])
    
    noise_mask = generate_noise_mask(img_neg, img_pos)
    eyelid_glint_mask = generate_eyelid_glint_mask(img_neg, img_pos, noise_mask)
    eyelash_mask = generate_eyelash_mask(img, eyelid_glint_mask)
    pupil_iris_mask = generate_pupil_iris_mask(noise_mask, eyelid_glint_mask, eyelash_mask)

    pupil_iris = apply_mask(img, pupil_iris_mask, keep_masked=True)

    images = [
        (img_neg, 'Original Negative Image'),
        (img_neg, 'Filtered Negative Image'),
        (eyelash_mask, 'Eyelash Mask'),
        (img_pos, 'Original Positive Image'),
        (img_neg, 'Filtered Positive Image'),
        (pupil_iris, 'Pupil and Iris'),
    ]

    plot_axes(2, 3, images)
    plt.tight_layout()
    plt.show()


    

    
    

if __name__ == '__main__':
    main()    