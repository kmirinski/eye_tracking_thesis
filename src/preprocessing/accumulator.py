import numpy as np

def accumulate_events(events: np.ndarray, n_events=2000):
    n_complete_sets = len(events) // n_events
    events = events[:n_complete_sets * n_events]

    event_sets = events.reshape(n_complete_sets, n_events, 4)

    neg_mask = event_sets[:, :, 0] == 0
    pos_mask = event_sets[:, :, 0] == 1

    neg_counts = neg_mask.sum(axis=1)
    pos_counts = pos_mask.sum(axis=1)

    neg_polarity = np.where(neg_mask[:, :, np.newaxis], event_sets, 0)
    pos_polarity = np.where(pos_mask[:, :, np.newaxis], event_sets, 0)

    return {
        'combined_polarity': event_sets,
        'negative_polarity': neg_polarity,
        'positive_polarity': pos_polarity,
        'negative_counts': neg_counts,
        'positive_counts': pos_counts
    }




    

    

