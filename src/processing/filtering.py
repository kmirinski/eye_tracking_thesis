import numpy as np
import cv2

from data.plot import plot_event_image, plot_event_image_standalone
from typing import List
from scipy.ndimage import uniform_filter, binary_dilation, label

def denoise_image(image: np.ndarray, box_size=6 , threshold=4):
    neighbourhood_sum = uniform_filter(image.astype(np.float32), size=box_size, mode='constant') * (box_size ** 2)
    keep_mask = neighbourhood_sum >= threshold
    removed_mask = image * (~keep_mask)
    filtered_image = np.where(keep_mask, image, 0).astype(image.dtype)
    return filtered_image, removed_mask

def create_eyelid_glint_mask(image_neg: np.ndarray, image_pos: np.ndarray, dilation_size=3):
    struct_elem = np.ones((dilation_size, dilation_size), dtype=bool)
    negative_dilated = binary_dilation(image_neg > 0, structure=struct_elem)
    positive_dilated = binary_dilation(image_pos > 0, structure=struct_elem)

    overlap = positive_dilated & negative_dilated

    eyelid_glint_mask = binary_dilation(overlap, structure=struct_elem)
    
    return eyelid_glint_mask


def remove_noise(img_neg: np.ndarray, img_pos: np.ndarray, threshold: float = 0.09, kernel_size: int = 6) -> tuple[np.ndarray, np.ndarray]:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
    print(kernel)

    neg_filtered = cv2.filter2D(img_neg.astype(np.float32), -1, kernel)
    pos_filtered = cv2.filter2D(img_pos.astype(np.float32), -1, kernel)

    neg_noise_mask = neg_filtered < threshold
    pos_noise_mask = pos_filtered < threshold

    img_neg_denoised = img_neg.copy()
    img_pos_denoised = img_pos.copy()
    img_neg_denoised[neg_noise_mask] = 0
    img_pos_denoised[pos_noise_mask] = 0

    return img_neg_denoised, img_pos_denoised





def generate_eyelid_glint_mask(img_neg_filtered: np.ndarray, img_pos_filtered: np.ndarray) -> np.ndarray:
    neg_binary = (img_neg_filtered > 0).astype(np.uint8)
    pos_binary = (img_pos_filtered > 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    neg_dilated = cv2.dilate(neg_binary, kernel, iterations=1)
    pos_dilated = cv2.dilate(pos_binary, kernel, iterations=1)

    overlapping_mask = np.logical_and(pos_dilated, neg_dilated).astype(np.uint8)

    kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    eyelid_glint_mask = cv2.dilate(overlapping_mask, kernel_expand, iterations=1)

    return eyelid_glint_mask



def generate_eyelash_mask(img: np.ndarray, eyelid_glint_mask: np.ndarray) -> np.ndarray:
    blur_mask = cv2.medianBlur(img, ksize=3) # may need to adjust
    
    if blur_mask.max() > 1:
        _, blur_mask = cv2.threshold(blur_mask, 0, 1, cv2.THRESH_BINARY)

    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3)) # may need to adjust
    morph_mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_horizontal)

    # plot_event_image_standalone(morph_mask, "A")

    kernel_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_CLOSE, kernel_disk)
    # plot_event_image_standalone(morph_mask, "B")
    morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_CLOSE, kernel_disk)
    # plot_event_image_standalone(morph_mask, "C")

    morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_OPEN, kernel_disk)
    # plot_event_image_standalone(morph_mask, "D")

    if morph_mask.max() > 1:
        _, morph_mask = cv2.threshold(morph_mask, 0, 1, cv2.THRESH_BINARY)

    # plot_event_image_standalone(morph_mask, "E")
    union_mask = np.logical_or(eyelid_glint_mask, morph_mask).astype(np.uint8)
    # plot_event_image_standalone(union_mask, "F")
    combined_mask = np.logical_and(union_mask, blur_mask).astype(np.uint8)
    # plot_event_image_standalone(combined_mask, "G")

    labeled_mask, num_features = label(combined_mask)

    if num_features == 0:
        return np.zeros(img, dtype=np.uint8)
    
    blob_sizes = [(labeled_mask == i).sum() for i in range(1, num_features + 1)]
    largest_blob_label = np.argmax(blob_sizes) + 1

    eyelash_mask = (labeled_mask == largest_blob_label).astype(np.uint8)

    return eyelash_mask




