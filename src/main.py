import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import random

from processing.preprocessing import *
from processing.filtering import *
from processing.pupil_finding import *
from data.plot import *
from utils import *
from data.loaders import EyeDataset, Frame, Event
from frame_processing.frame_processing import process_frame
from tracking import track_pupil


parser = argparse.ArgumentParser(description='Arguments for reading the data')
parser.add_argument('--subject', type=int, default=22, help='choose subject')
parser.add_argument('--eye', default='left', choices=['left', 'right'],
                    help='choose left or right eye dataset')
parser.add_argument('--data_dir', default=os.path.join(os.getcwd(), 'eye_data'),
                    help='absolute path to eye_data/, assumes same parent dir as this script by default')
opt = parser.parse_args()


def main():
    eye_dataset = EyeDataset(opt.data_dir, opt.subject, mode='stack')
    print('Collecting data of the left eye of subject ' + str(opt.subject))
    print('Loading data from ' + opt.data_dir)
    
    with timer("Collection + Accumulation"):
        eye_dataset.collect_data(eye=0)
        # event_sets = accumulate_events(eye_dataset.event_list, n_events=2000) # Add this to args later (for now hardcoded)
    
    
    # pupil_tracking = track_pupil(eye_dataset, 50)
    ellipse = None
    img_idxs = random.sample(range(1, len(eye_dataset.frame_list) + 1), 1)
    # There is an image with index -706 in which the subject is shot during a blink, therefore the pupil extraction fails
    # img_idxs = [-706]
    for idx in img_idxs:
        # (x, y) (w, h) phi -> x_center, y_center, width, height, clockwise rotation
        with timer("Process frame"):
            ellipse = process_frame(eye_dataset.frame_list[idx], visualize=True)

    # # This ellipse is the region of interest
    # # Expand the region of interest

    print(ellipse) # 0.04s -> 25Hz (how often we get frames)
    
    # with timer("Positive + Negative"):
    #     neg_sets = extract_polarity_sets(event_sets, 0)
    #     pos_sets = extract_polarity_sets(event_sets, 1)


    # img_idxs = random.sample(range(1, len(event_sets) + 1), 2)
    # img_idxs.append(0)
    # img_idxs = random.sample(range(1, len(event_sets) + 1), 3)
    # images, centers = generate_eye_images(neg_sets, pos_sets, event_sets, img_idxs)
    # print(centers)

    # plot_axes(2, 3, images, centers)
    # plt.tight_layout()
    # plt.show()


def generate_eye_images(neg_sets, pos_sets, event_sets, img_idxs):
    n = len(img_idxs)
    images = [None] * (2 * n)
    centers = [None] * n

    for idx, i in enumerate(img_idxs):
        img_neg = event_to_image(neg_sets[i])
        img_pos = event_to_image(pos_sets[i])
        img = event_to_image(event_sets[i])
        pupil_iris = generate_and_apply_masks(img_neg, img_pos, img)
        center_x, center_y, _ = locate_pupil_center_kde(pupil_iris)

        images[idx] = (img, f"Image {i}")
        images[n + idx] = (pupil_iris, f"Image {i} extracted") 
        centers[idx] = (center_x, center_y) 

    return images, centers
    

if __name__ == '__main__':
    main()    