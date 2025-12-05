import numpy as np

from typing import List, Any
from data.loaders import EyeDataset, Event

def accumulate_events(event_params: List[Any], n_events=2000):
    params_per_set = n_events * 4
    n_complete_sets = len(event_params) // params_per_set
    combined_polarity = []
    negative_polarity = []
    positive_polarity = []

    for i in range(0, n_complete_sets * params_per_set, params_per_set):
        event_set = event_params[i:i + params_per_set]

        negative_polarity_set = []
        positive_polarity_set = []
        
        for j in range(0, params_per_set, 4):
            polarity = event_set[j]
            if polarity == 0:
                negative_polarity_set.extend(event_set[j:j+4])
            else:
                positive_polarity_set.extend(event_set[j:j+4]) # No need to keep polarity if it is already in the set
        
        combined_polarity.append(event_set)
        negative_polarity.append(negative_polarity_set)
        positive_polarity.append(positive_polarity_set)

    return {
        'combined_polarity': combined_polarity, 
        'negative_polarity': negative_polarity, 
        'positive_polarity': positive_polarity
    }




    

    

