import numpy as np

from typing import List

def box_filter_events(event_sets: List[np.ndarray], box_size=2, threshold=2):
    
    filtered_sets = []

    for event_set in event_sets:
        rows = event_set[:, 1].astype(int)
        cols = event_set[:, 2].astype(int)

        keep_mask = np.zeros(len(event_set), dtype=bool)

        for j in range(len(event_set)):
            r, c = rows[j], cols[j]

            in_neighborhood = (
                (rows >= r - box_size//2) & 
                (rows <= r + box_size//2) &
                (cols >= c - box_size//2) & 
                (cols <= c + box_size//2)
            )
            
            if np.sum(in_neighborhood) >= threshold:
                keep_mask[j] = True
        
        filtered_sets.append(event_set[keep_mask])
    
    return filtered_sets




