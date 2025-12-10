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


def events_to_images(events: np.ndarray, img_width=346, img_height=260):
    """
    Convert event list to images and preserve metadata.

    Returns:
        images: Numpy array containing all the images (one image per event set)
        metadata: List of dictionaries with coordiante mappings (one per event set)
    """
    n_sets = events.shape[0]
    images = np.zeros((n_sets, img_height, img_width))
    metadata = [None] * n_sets

    for i in range(n_sets):
        event_set = events[i]
        rows = event_set[:, 1].astype(int)
        cols = event_set[:, 2].astype(int)
        np.add.at(images[i], (rows, cols), 1)

        event_map = defaultdict(list)
        for j, (polarity, row, col, timestamp) in enumerate(event_set):
            event_map[(int(row), int(col))].append((polarity, timestamp, j))
        
        metadata[i] = dict(event_map)
    
    return images, metadata

def images_to_events(images: np.ndarray, metadata: List[dict]):
    """
    Convert images back to event format.
    The initial sequence is preserved by sorting by the timestamps.

    Returns:
        events: List of events
    """

    n_sets = images.shape[0]

    max_events = max(
        sum(len(event_list) for event_list in meta.values())
        for meta in metadata
    )

    


    events = []

    for image, meta in zip(images, metadata):
        event_list = []
        rows, cols = np.where(image > 0)

        for row, col in zip (rows, cols):
            coord = (int(row), int(col))
            if coord in meta:
                for polarity, timestamp, idx in meta[coord]:
                    event_list.append((idx, polarity, row, col, timestamp))
        
        event_list.sort(key=lambda x: x[0])

        event_set = []
        for _, polarity, row, col, timestamp in event_list:
            event_set.extend([polarity, row.item(), col.item(), timestamp])
        events.append(event_set)
    
    return events
