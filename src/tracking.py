import cv2
import numpy as np
from scipy.spatial import cKDTree

from data.loaders import EyeDataset, Frame, Event
from processing.frame_detection import extract_pupil
from config import TrackingConfig, TemplateTrackingConfig


def fit_ellipse(events):
    if len(events) < 5:
        return None

    points = np.array([[e.col, e.row] for e in events], dtype=np.float32).reshape(-1, 1, 2)
    try:
        return cv2.fitEllipse(points)
    except Exception:
        return None


def calculate_fit_score(ellipse, events):
    if ellipse is None or len(events) == 0:
        return 0.0

    (xp, yp), (wp, hp), phi_p = ellipse

    event_coords = np.array([[e.col, e.row] for e in events])
    centered = event_coords - np.array([xp, yp])

    sin_phi = np.sin(np.deg2rad(phi_p))
    cos_phi = np.cos(np.deg2rad(phi_p))

    R = np.array([[cos_phi, -sin_phi],
                  [sin_phi, cos_phi]])
    rotated = centered @ R.T

    if wp == 0 or hp == 0:
        return 0.0
    scaled = rotated / np.array([wp/2, hp/2])
    distances = np.linalg.norm(scaled, axis=1)

    mean_deviation = np.mean(np.abs(distances - 1))
    fit_score = 1 - mean_deviation

    return max(0.0, fit_score)


def get_roi_events(events, prev_ellipse, expansion_factor=1.5):
    (xp, yp), (wp, hp), phi_p = prev_ellipse

    expanded_w = wp * expansion_factor
    expanded_h = hp * expansion_factor

    roi_events = []

    phi_rad = np.deg2rad(phi_p)
    cos_phi = np.cos(-phi_rad)
    sin_phi = np.sin(-phi_rad)

    for event in events:
        dx = event.col - xp
        dy = event.row - yp

        rotated_x = dx * cos_phi - dy * sin_phi
        rotated_y = dx * sin_phi + dy * cos_phi

        normalized = (rotated_x / (expanded_w/2))**2 + (rotated_y / (expanded_h/2))**2

        if normalized <= 1.0:
            roi_events.append(event)

    return roi_events


def sample_ellipse_boundary(ellipse, num_points=360):
    (cx, cy), (w, h), angle_deg = ellipse
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    pts = np.column_stack([(w / 2) * np.cos(theta), (h / 2) * np.sin(theta)])
    phi = np.deg2rad(angle_deg)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    R = np.array([[cos_phi, -sin_phi],
                  [sin_phi, cos_phi]])
    pts = pts @ R.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return pts


def select_candidate_events(event_coords, center, gamma_bar, lambda1, lambda2):
    dists = np.linalg.norm(event_coords - center, axis=1)
    mask = (dists > lambda1 * gamma_bar) & (dists < lambda2 * gamma_bar)
    return np.where(mask)[0]


def points_to_edge_matching(candidates, boundary_Q, max_iter=50, convergence=0.01):
    tree = cKDTree(boundary_Q)
    P = candidates.copy()
    T_total = np.zeros(2)

    for _ in range(max_iter):
        _, indices = tree.query(P)
        nearest = boundary_Q[indices]
        delta_T = np.mean(nearest - P, axis=0)

        T_total += delta_T
        P += delta_T

        t_norm = np.linalg.norm(T_total)
        if t_norm > 0 and np.linalg.norm(delta_T) / t_norm < convergence:
            break

    return T_total


class PupilTracker:

    def __init__(self, config: TrackingConfig = None):
        self.config = config or TrackingConfig()

    def track(self, eye_dataset: EyeDataset) -> list:
        initial_frame = eye_dataset.get_initial_frame()
        prev_ellipse = extract_pupil(initial_frame, visualize=False)

        tracked_ellipses = []
        if prev_ellipse is not None:
            tracked_ellipses.append({
                'ellipse': prev_ellipse,
                'timestamp': initial_frame.timestamp,
                'fit_score': 1.0,
                'source': 'frame'
            })

        events_buffer = []
        use_roi = True  # initially true since pupil is successfully extracted from the first frame

        while eye_dataset.frame_list or eye_dataset.event_stack:
            itm = eye_dataset.__get_item__(0)

            if type(itm) is Event:
                events_buffer.append(itm)

                if len(events_buffer) >= self.config.num_events:
                    if use_roi and prev_ellipse is not None:
                        roi_events = get_roi_events(events_buffer, prev_ellipse, expansion_factor=self.config.roi_expansion)
                        if len(roi_events) >= 5:
                            current_ellipse = fit_ellipse(roi_events)
                            fit_score = calculate_fit_score(current_ellipse, events_buffer)
                        else:
                            # Not correct
                            # Not enough events in ROI, fall back to full processing
                            current_ellipse = fit_ellipse(events_buffer)
                            fit_score = calculate_fit_score(current_ellipse, events_buffer)
                    else:
                        # First event set or previous fit was bad
                        current_ellipse = fit_ellipse(events_buffer)
                        fit_score = calculate_fit_score(current_ellipse, events_buffer)

                    if fit_score >= self.config.fit_threshold and current_ellipse is not None:
                        tracked_ellipses.append({
                            'ellipse': current_ellipse,
                            'timestamp': events_buffer[-1].timestamp,
                            'fit_score': fit_score,
                            'source': 'events'
                        })
                        prev_ellipse = current_ellipse
                        use_roi = True
                    else:
                        use_roi = False

                    events_buffer = []

            elif type(itm) is Frame:
                frame_ellipse = extract_pupil(itm, visualize=False)
                if frame_ellipse is not None:
                    tracked_ellipses.append({
                        'ellipse': frame_ellipse,
                        'timestamp': itm.timestamp,
                        'fit_score': 1.0,
                        'source': 'frame'
                    })
                    prev_ellipse = frame_ellipse
                    use_roi = True

                events_buffer = []

        return tracked_ellipses
