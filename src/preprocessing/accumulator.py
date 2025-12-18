import numpy as np

def accumulate_events(events: np.ndarray, n_events=2000):
    n_complete_sets = len(events) // n_events
    events = events[:n_complete_sets * n_events]
    event_sets = events.reshape(n_complete_sets, n_events, 4)

    return {
        'combined_polarity': event_sets,
    }