import numpy as np

def accumulate_events(events: np.ndarray, n_events=2000):
    n_complete_sets = len(events) // n_events
    events = events[:n_complete_sets * n_events]

    event_sets = events.reshape(n_complete_sets, n_events, 4)

    combined_polarity = event_sets
    negative_polarity = [event_sets[i][event_sets[i, :, 0] == 0] for i in range(n_complete_sets)]
    positive_polarity = [event_sets[i][event_sets[i, :, 0] == 1] for i in range(n_complete_sets)]

    return {
        'combined_polarity': combined_polarity, 
        'negative_polarity': negative_polarity, 
        'positive_polarity': positive_polarity
    }




    

    

