import numpy as np


def compute_pupil_stats(pupil_centers):
    """
    Compute per-axis mean and std from pupil detections, ignoring failed
    detections marked as (-1, -1).

    Parameters
    ----------
    pupil_centers : np.ndarray, shape (N, 2)

    Returns
    -------
    mean : np.ndarray, shape (2,)
    std  : np.ndarray, shape (2,)
    """
    valid = pupil_centers[:, 0] != -1
    pts = pupil_centers[valid]
    return pts.mean(axis=0), pts.std(axis=0)


def normalize_pupils(pupil_centers, mean, std):
    """
    Apply per-axis z-score normalization.

    Parameters
    ----------
    pupil_centers : np.ndarray, shape (N, 2)  — already filtered (no -1 entries)
    mean          : np.ndarray, shape (2,)
    std           : np.ndarray, shape (2,)

    Returns
    -------
    np.ndarray, shape (N, 2)
    """
    return (pupil_centers - mean) / std
