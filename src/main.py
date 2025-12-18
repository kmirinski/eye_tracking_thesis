import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

from processing.preprocessing import *
from processing.denoise import denoise_image, create_eyelid_glint_mask
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

    img_neg = event_to_image(neg_sets[0])
    img_pos = event_to_image(pos_sets[0])

    filtered_neg, mask_neg = denoise_image(img_neg)
    filtered_pos, mask_pos = denoise_image(img_pos)

    mask = create_eyelid_glint_mask(filtered_neg, filtered_pos, dilation_size=5)

    # Apply to remove eyelids and glints from your images
    dilated_neg = filtered_neg * (~mask)
    dilated_pos = filtered_pos * (~mask)

    _, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=200)
    plot_event_image(img_neg, axes[0][0], 'Original Image')
    plot_event_image(filtered_neg, axes[0][1], 'Filtered Image')
    plot_event_image(dilated_neg, axes[0][2], 'Dilated Image')
    # plot_event_image(mask_neg.astype(np.uint8), axes[0][2], 'Mask')
    plot_event_image(img_pos, axes[1][0], 'Original Image')
    plot_event_image(filtered_pos, axes[1][1], 'Filtered Image')
    plot_event_image(dilated_pos, axes[1][2], 'Dilated Image')
    # plot_event_image(mask_pos.astype(np.uint8), axes[1][2], 'Mask')

    plt.tight_layout()
    plt.show()

    
    

if __name__ == '__main__':
    main()    