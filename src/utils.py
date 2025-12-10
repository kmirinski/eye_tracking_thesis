import numpy as np
import time

from contextlib import contextmanager
from typing import List, Any
from collections import defaultdict

@contextmanager
def timer(name="Operation"):
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"{name}: {elapsed_ms:.2f} ms")


def events_to_images(events: np.ndarray, img_width=346, img_height=260, counts=None):
    """
    Convert event list to images and preserve metadata.

    Returns:
        images: Numpy array containing all the images (one image per event set)
        metadata: List of dictionaries with coordiante mappings (one per event set)
    """
    n_sets = events.shape[0]
    images = np.zeros((n_sets, img_height, img_width), dtype=np.int32)
    polarities = np.full((n_sets, img_height, img_width), -1, dtype=np.int8)

    for i in range(n_sets):
        event_set = events[i, :counts[i]] if counts is not None else events[i]
    
        rows = event_set[:, 1].astype(int)
        cols = event_set[:, 2].astype(int)
        pols = event_set[:, 0].astype(int)

        np.add.at(images[i], (rows, cols), 1)
        polarities[i, rows, cols] = pols
    
    return images, polarities
