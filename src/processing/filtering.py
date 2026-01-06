import numpy as np
import cv2

from data.plot import plot_event_image_standalone, plot_axes
from typing import List
from scipy.ndimage import label, median_filter

def generate_noise_mask(img_neg: np.ndarray, img_pos: np.ndarray, threshold: float = 0.18, kernel_size: int = 6) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

    neg_filtered = cv2.filter2D(img_neg.astype(np.float32), -1, kernel)
    pos_filtered = cv2.filter2D(img_pos.astype(np.float32), -1, kernel)

    neg_noise = (img_neg > 0) & (neg_filtered < threshold)
    pos_noise = (img_pos > 0) & (pos_filtered < threshold)

    noise_mask = np.logical_or(neg_noise, pos_noise).astype(np.uint8)
    
    return noise_mask


def generate_eyelid_glint_mask(img_neg_filtered: np.ndarray, img_pos_filtered: np.ndarray, noise_mask: np.ndarray) -> np.ndarray:
    img_neg_filtered = apply_mask(img_neg_filtered, noise_mask)
    img_pos_filtered = apply_mask(img_pos_filtered, noise_mask)
    
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
    blur_mask_15 = img.copy()
    for i in range(15):
        blur_mask_15 = median_filter(blur_mask_15, size=(1, 2))  # (height, width)
    blur_mask = blur_mask_15
    
    if blur_mask.max() > 1:
        _, blur_mask = cv2.threshold(blur_mask, 0, 1, cv2.THRESH_BINARY)

    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 4)) # may need to adjust
    morph_mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_horizontal)

    kernel_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_CLOSE, kernel_disk)
    morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_CLOSE, kernel_disk)

    morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_OPEN, kernel_disk)

    if morph_mask.max() > 1:
        _, morph_mask = cv2.threshold(morph_mask, 0, 1, cv2.THRESH_BINARY)

    union_mask = np.logical_or(eyelid_glint_mask, morph_mask).astype(np.uint8)

    # return union_mask
    combined_mask = np.logical_and(union_mask, blur_mask).astype(np.uint8)

    labeled_mask, num_features = label(combined_mask)

    if num_features == 0:
        return np.zeros(img, dtype=np.uint8)
    
    blob_sizes = [(labeled_mask == i).sum() for i in range(1, num_features + 1)]
    largest_blob_label = np.argmax(blob_sizes) + 1

    eyelash_mask = (labeled_mask == largest_blob_label).astype(np.uint8)

    return eyelash_mask

def generate_pupil_iris_mask(noise_mask:np.ndarray, eyelid_glint_mask: np.ndarray, eyelash_mask: np.ndarray) -> np.ndarray:
    unified_mask = np.logical_or(noise_mask, eyelid_glint_mask)
    # plot_event_image_standalone(unified_mask, "Noise or eyelid_glint mask")
    unified_mask = np.logical_or(unified_mask, eyelash_mask)
    # plot_event_image_standalone(unified_mask, "Noise or eyelid_glint mask")

    pupil_iris_mask = np.logical_not(unified_mask).astype(np.uint8)
    
    return pupil_iris_mask

def apply_mask(img: np.ndarray, mask: np.ndarray, keep_masked: bool = False) -> np.ndarray:
    filtered_img = img.copy()
    
    if keep_masked:
        filtered_img[mask == 0] = 0
    else:
        filtered_img[mask == 1] = 0
    
    return filtered_img




