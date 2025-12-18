import numpy as np

def accumulate_events(events: np.ndarray, n_events=2000):
    n_complete_sets = len(events) // n_events
    events = events[:n_complete_sets * n_events]
    event_sets = events.reshape(n_complete_sets, n_events, 4)

    return event_sets

def extract_polarity_sets(event_sets: np.ndarray, polarity: int):
    n_sets = event_sets.shape[0]
    polarity_events = []

    for i in range(n_sets):
        mask = event_sets[i, :, 0] == polarity
        polarity_events.append(event_sets[i][mask]) 

    return polarity_events
