import numpy as np
import cv2

from data.plot import plot_event_image_standalone, plot_axes
from typing import List
from scipy.ndimage import label, median_filter

def generate_noise_mask(img_neg: np.ndarray, img_pos: np.ndarray, threshold: float = 0.09, kernel_size: int = 6) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
    print(kernel)

    neg_filtered = cv2.filter2D(img_neg.astype(np.float32), -1, kernel)
    pos_filtered = cv2.filter2D(img_pos.astype(np.float32), -1, kernel)

    neg_noise = (img_neg > 0) & (neg_filtered < threshold)
    pos_noise = (img_pos > 0) & (pos_filtered < threshold)

    noise_mask = np.logical_or(neg_noise, pos_noise).astype(np.uint8)
    
    return noise_mask

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
    # blur_mask = cv2.medianBlur(img, ksize=3) # may need to 

    # blur_mask = median_filter(img, size=(2, 4))  # (height, width)
    size = (1, 2)
    blur_mask_1 = img.copy()
    blur_mask_1 = median_filter(blur_mask_1, size=size)  # (height, width)
    
    blur_mask_5 = img.copy()
    for i in range(5):
        # blur_mask = cv2.medianBlur(blur_mask, ksize=3)
        blur_mask_5 = median_filter(blur_mask_5, size=size)  # (height, width)

    blur_mask_15 = img.copy()
    for i in range(15):
        # blur_mask = cv2.medianBlur(blur_mask, ksize=3)
        blur_mask_15 = median_filter(blur_mask_15, size=size)  # (height, width)
    blur_mask = blur_mask_15
    
    # noise_images = [
    #     (blur_mask_1, 'Blur mask 1'),
    #     (blur_mask_5, 'Blur mask 5'),
    #     (blur_mask_15, 'Blur mask 15'),
    #     # (filtered_pos_img, 'Filtered Image'),
    # ]
    # plot_axes(1, 3, noise_images)
    
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

    if morph_mask.max() > 1:
        _, morph_mask = cv2.threshold(morph_mask, 0, 1, cv2.THRESH_BINARY)

    # plot_event_image_standalone(morph_mask, "E")
    union_mask = np.logical_or(eyelid_glint_mask, morph_mask).astype(np.uint8)

    return union_mask
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

def generate_pupil_iris_mask(noise_mask:np.ndarray, eyelid_glint_mask: np.ndarray, eyelash_mask: np.ndarray) -> np.ndarray:
    unified_mask = np.logical_or(noise_mask, eyelid_glint_mask)
    plot_event_image_standalone(unified_mask, "Noise or eyelid_glint mask")
    unified_mask = np.logical_or(unified_mask, eyelash_mask)
    plot_event_image_standalone(unified_mask, "Noise or eyelid_glint mask")

    pupil_iris_mask = np.logical_not(unified_mask).astype(np.uint8)
    
    return pupil_iris_mask

def apply_mask(img: np.ndarray, mask: np.ndarray, keep_masked: bool = False) -> np.ndarray:
    filtered_img = img.copy()
    
    if keep_masked:
        filtered_img[mask == 0] = 0
    else:
        filtered_img[mask == 1] = 0
    
    return filtered_img




