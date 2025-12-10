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


def events_to_images(events: List[List[Any]], img_width=346, img_height=260):
    """
    Convert event list to images and preserve metadata.

    Returns:
        images: List of numpy arrays (one per event set)
        metadata: List of dictionaries with coordiante mappings (one per event set)
    """

    images = []
    metadata = []

    for event_set in events:
        image = np.zeros((img_height, img_width))
        event_map = defaultdict(list)

        for i in range(0, len(event_set), 4):
            polarity = event_set[i]
            row = event_set[i + 1]
            col = event_set[i + 2]
            timestamp = event_set[i + 3]
            image[row, col] += 1
            event_map[(row, col)].append((polarity, timestamp, i))
        
        images.append(image)
        metadata.append(dict(event_map))
    
    return images, metadata

def images_to_events(images: List[np.ndarray], metadata: List[dict]):
    """
    Convert images back to event format.
    The initial sequence is preserved by sorting by the timestamps.

    Returns:
        events: List of events
    """
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
