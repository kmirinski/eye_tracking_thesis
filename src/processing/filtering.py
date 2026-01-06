import numpy as np
import cv2

from data.plot import plot_event_image, plot_event_image_standalone
from typing import List
from scipy.ndimage import uniform_filter, binary_dilation, label

def box_filter_events(event_sets: List[np.ndarray], box_size=6, threshold=4,
                           event_resolution=(260, 346)):
    """
    sensor_shape = (height, width) of event camera
    """
    half = box_size // 2
    filtered_sets = []

    for event_set in event_sets:
        if len(event_set) == 0:
            filtered_sets.append(event_set)
            continue

        rows = event_set[:, 1].astype(int)
        cols = event_set[:, 2].astype(int)

        # Build occupancy grid
        grid = np.zeros(event_resolution, dtype=np.int32)
        np.add.at(grid, (rows, cols), 1)

        # Integral image (summed-area table)
        integral = grid.cumsum(axis=0).cumsum(axis=1)

        # Neighborhood count for each event
        r0 = np.clip(rows - half, 0, event_resolution[0] - 1)
        r1 = np.clip(rows + half, 0, event_resolution[0] - 1)
        c0 = np.clip(cols - half, 0, event_resolution[1] - 1)
        c1 = np.clip(cols + half, 0, event_resolution[1] - 1)

        counts = (
            integral[r1, c1]
            - np.where(c0 > 0, integral[r1, c0 - 1], 0)
            - np.where(r0 > 0, integral[r0 - 1, c1], 0)
            + np.where((r0 > 0) & (c0 > 0), integral[r0 - 1, c0 - 1], 0)
        )

        keep_mask = counts >= threshold
        filtered_sets.append(event_set[keep_mask])

    return filtered_sets

def denoise_image(image: np.ndarray, box_size=6, threshold=4):
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

def generate_eyelash_mask(image: np.ndarray, eyelid_mask: np.ndarray) -> np.ndarray:
    blur_mask = cv2.medianBlur(image, ksize=3) # may need to adjust
    
    if blur_mask.max() > 1:
        _, blur_mask = cv2.threshold(blur_mask, 0, 1, cv2.THRESH_BINARY)

    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3)) # may need to adjust
    morph_mask = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_horizontal)

    plot_event_image_standalone(morph_mask, "A")

    kernel_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_CLOSE, kernel_disk)
    plot_event_image_standalone(morph_mask, "B")
    morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_CLOSE, kernel_disk)
    plot_event_image_standalone(morph_mask, "C")

    morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_OPEN, kernel_disk)
    # plot_event_image_standalone(morph_mask, "D")

    if morph_mask.max() > 1:
        _, morph_mask = cv2.threshold(morph_mask, 0, 1, cv2.THRESH_BINARY)

    plot_event_image_standalone(morph_mask, "E")
    union_mask = np.logical_or(eyelid_mask, morph_mask).astype(np.uint8)
    plot_event_image_standalone(union_mask, "F")
    combined_mask = np.logical_and(union_mask, blur_mask).astype(np.uint8)
    plot_event_image_standalone(combined_mask, "G")

    labeled_mask, num_features = label(combined_mask)

    if num_features == 0:
        return np.zeros(image, dtype=np.uint8)
    
    blob_sizes = [(labeled_mask == i).sum() for i in range(1, num_features + 1)]
    largest_blob_label = np.argmax(blob_sizes) + 1

    eyelash_mask = (labeled_mask == largest_blob_label).astype(np.uint8)

    return eyelash_mask




