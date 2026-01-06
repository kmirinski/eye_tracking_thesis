import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import random

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


    img_idxs = random.sample(range(1, len(event_sets) + 1), 6)
    images = generate_eye_images(neg_sets, pos_sets, event_sets, img_idxs)

    plot_axes(2, 3, images)
    plt.tight_layout()
    plt.show()


def generate_eye_images(neg_sets, pos_sets, event_sets, img_idxs):

    images = []

    for i in img_idxs:
        img_neg = event_to_image(neg_sets[i])
        img_pos = event_to_image(pos_sets[i])
        img = event_to_image(event_sets[i])

        noise_mask = generate_noise_mask(img_neg, img_pos)
        eyelid_glint_mask = generate_eyelid_glint_mask(img_neg, img_pos, noise_mask)
        eyelash_mask = generate_eyelash_mask(img, eyelid_glint_mask)
        pupil_iris_mask = generate_pupil_iris_mask(noise_mask, eyelid_glint_mask, eyelash_mask)

        pupil_iris = apply_mask(img, pupil_iris_mask, keep_masked=True)

        images.append((pupil_iris, f"Image {i}"))

    return images

    

    
    

if __name__ == '__main__':
    main()    