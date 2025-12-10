import numpy as np
from collections import defaultdict
from scipy.ndimage import convolve, binary_dilation
from typing import List, Any

def filter_noise(events: List[List[Any]], img_width=346, img_height=260, box_size=2, threshold=2):
    filtered_events = []
    kernel = np.ones((box_size, box_size))
    
    for event_set in events:
        image = np.zeros((img_height, img_width))
        for i in range(0, len(event_set), 4):
            image[event_set[i + 1], event_set[i + 2]] += 1
        filtered_image = convolve(image, kernel, mode='constant')
        filtered_set = []
        for i in range(0, len(event_set), 4):
            density = filtered_image[event_set[i + 1], event_set[i + 2]]
            if density >= threshold:
                filtered_set.extend(event_set[i:i+4])
        filtered_events.append(filtered_set)
    return filtered_events



