import numpy as np

from typing import List

def box_filter_events(event_sets: List[np.ndarray], box_size=6, threshold=4,
                           sensor_shape=(260, 346)):
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
        grid = np.zeros(sensor_shape, dtype=np.int32)
        np.add.at(grid, (rows, cols), 1)

        # Integral image (summed-area table)
        integral = grid.cumsum(axis=0).cumsum(axis=1)

        # Neighborhood count for each event
        r0 = np.clip(rows - half, 0, sensor_shape[0] - 1)
        r1 = np.clip(rows + half, 0, sensor_shape[0] - 1)
        c0 = np.clip(cols - half, 0, sensor_shape[1] - 1)
        c1 = np.clip(cols + half, 0, sensor_shape[1] - 1)

        counts = (
            integral[r1, c1]
            - np.where(c0 > 0, integral[r1, c0 - 1], 0)
            - np.where(r0 > 0, integral[r0 - 1, c1], 0)
            + np.where((r0 > 0) & (c0 > 0), integral[r0 - 1, c0 - 1], 0)
        )

        keep_mask = counts >= threshold
        filtered_sets.append(event_set[keep_mask])

    return filtered_sets





