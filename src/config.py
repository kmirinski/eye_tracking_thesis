import os
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATHS = {
    "ebveye": os.path.join(BASE_DIR, "eye_data/ebveye"),
    "ev_eye": os.path.join(BASE_DIR, "eye_data/ev_eye/raw_data"),
}

@dataclass
class FrameDetectionConfig:
    threshold: int = 15
    morph_kernel_size: int = 3
    min_aspect_ratio: float = 0.32
    min_axis_px: int = 0            # Currently not in use
    max_axis_px: int = 120          # Currently not in use
    center_min: tuple = None        # (x_min, y_min) accepted pupil center in px; None = no limit
    center_max: tuple = None        # (x_max, y_max) accepted pupil center in px; None = no limit
    triangle_corner: str = None     # 'upper_right' (left eye) or 'upper_left' (right eye)
    triangle_size: int = 100         # leg length in px of the corner triangle to exclude
    extra_triangle_corners: tuple = ()  # additional corners to exclude, e.g. ('lower_left',)
    min_ellipse_area: float = 210   # π * (w/2) * (h/2) in px²
    max_ellipse_area: float = None  # π * (w/2) * (h/2) in px²; None = no limit


# Per-subject overrides for FrameDetectionConfig.
# Only list fields that differ from the dataclass defaults above.
EBVEYE_FRAME_DETECTION_OVERRIDES: dict = {
    4:  {"threshold": 10, "morph_kernel_size": 4, 'min_aspect_ratio': 0.38, "triangle_size": 150},
    5:  {"threshold": 13, "morph_kernel_size": 3},
    6: {'threshold': 10, 'morph_kernel_size': 4, 'min_aspect_ratio': 0.25},
    7: {'threshold': 10, 'morph_kernel_size': 2, 'min_aspect_ratio': 0.25},
    11: {'threshold': 15, 'morph_kernel_size': 3, 'min_aspect_ratio': 0.25},
    12: {'threshold': 15, 'morph_kernel_size': 3, 'min_aspect_ratio': 0.25},
    15: {'threshold': 25, 'morph_kernel_size': 2, 'min_aspect_ratio': 0.38},
    18: {'threshold': 20, 'morph_kernel_size': 2, 'min_aspect_ratio': 0.25},
    19: {'threshold': 15, 'morph_kernel_size': 2, 'min_aspect_ratio': 0.25},
    21: {'threshold': 10, 'morph_kernel_size': 4, 'min_aspect_ratio': 0.25},
    22: {'threshold': 10, 'morph_kernel_size': 4, 'min_aspect_ratio': 0.25},
}

EV_EYE_FRAME_DETECTION_OVERRIDES: dict = {
    4:  {"extra_triangle_corners": ('lower_left', 'upper_left')},
    5: {"threshold": 20, "morph_kernel_size": 2, "extra_triangle_corners": ('lower_left', 'upper_left')},
    6: {"threshold": 20, "morph_kernel_size": 2, "triangle_size": 180, "extra_triangle_corners": ('lower_left', "upper_right"), "max_ellipse_area": 5000},
    7: {"morph_kernel_size": 2, "extra_triangle_corners": ('lower_left', 'upper_right'), "min_ellipse_area": 190},
    8: {"morph_kernel_size": 2, "triangle_size": 150, "extra_triangle_corners": ('lower_left', 'upper_right'), "min_ellipse_area": 140},
    25: {"threshold": 30, "morph_kernel_size": 4, 'min_aspect_ratio': 0.38, "extra_triangle_corners": ('lower_left',)},
    22: {"morph_kernel_size": 2, "triangle_size": 200, "extra_triangle_corners": ('lower_left',)},
    29: {"threshold": 24, "morph_kernel_size": 2, "triangle_size": 200, "extra_triangle_corners": ('lower_left', 'upper_left'), "min_ellipse_area": 300}, # A bit sus
    30: {"threshold": 20, "morph_kernel_size": 2, "triangle_size": 120, "extra_triangle_corners": ('upper_left', 'upper_right')},
    31: {"threshold": 20, "morph_kernel_size": 2, "extra_triangle_corners": ('upper_left', 'lower_left')},
    32: {"morph_kernel_size": 2},
    33: {"morph_kernel_size": 2},
    34: {"threshold": 20, "morph_kernel_size": 2, 'min_aspect_ratio': 0.25, "triangle_size": 200, "extra_triangle_corners": ('lower_left',)},
    35: {"triangle_size": 150, "extra_triangle_corners": ('upper_left', 'lower_left'), 'min_aspect_ratio': 0.25, "min_ellipse_area": 140},
    36: {"threshold":20, "morph_kernel_size": 2, 'min_aspect_ratio': 0.25, "min_ellipse_area": 140},
    40: {},
    41: {},
    42: {},
    43: {},
    44: {"threshold": 20, "morph_kernel_size": 2, "triangle_size": 180, "extra_triangle_corners": ('lower_left',)},
}


def get_frame_detection_config(subject: int, eye: str, dataset: str = 'ebveye') -> FrameDetectionConfig:
    """Return a FrameDetectionConfig with defaults + per-subject overrides applied."""
    corner = 'upper_right' if eye == 'left' else 'upper_left'

    if dataset == 'ev_eye':
        overrides = dict(EV_EYE_FRAME_DETECTION_OVERRIDES.get(subject, {}))
        return FrameDetectionConfig(triangle_corner=corner, **overrides)

    overrides = EBVEYE_FRAME_DETECTION_OVERRIDES.get(subject, {})
    return FrameDetectionConfig(triangle_corner=corner, **overrides)


@dataclass
class TrackingConfig:
    num_events: int = 2000       # legacy global tracker
    fit_threshold: float = 0.8   # used by legacy global tracker only
    roi_expansion: float = 1.1
    num_events_roi: int = 80    # events to accumulate within ROI per batch

@dataclass
class TemplateTrackingConfig:
    num_events: int = 20
    lambda1: float = 0.8
    lambda2: float = 1.2
    convergence: float = 0.01
    max_icp_iter: int = 50
    num_boundary: int = 360

@dataclass
class KDEConfig:
    pupil_radius: int = 30
    bandwidth_large: float = 35.0
    bandwidth_small: float = 15.0

@dataclass
class GazeConfig:
    poly_degrees: list = (5, 6, 7, 8, 12)
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    fine_tune_ratio: float = 0.4         # fraction of held-out subject data used for fine-tuning/calibration
    saccade_skip_frames: int = 20
    relabel_diff_threshold: float = 1.5  # px; eye displacement below this = stable fixation
    relabel_max_frames: int = 20         # safety cap: never relabel more than this many frames per label change
    post_blink_skip_frames: int = 1      # valid frames to discard after each blink run
    post_saccade_stability_window: int = 6  # consecutive stable frames required before Phase C begins
    screen_width_px: int = 1920
    screen_height_px: int = 1080
    screen_fov_x_deg: float = 96.0       # full horizontal FoV of the screen in degrees
    screen_fov_y_deg: float = 64.0       # full vertical FoV of the screen in degrees


# Per-subject overrides for GazeConfig.
# Only list fields that differ from the dataclass defaults above.
SUBJECT_GAZE_OVERRIDES: dict = {
    # example:
    # 3:  {"saccade_skip_frames": 30, "relabel_diff_threshold": 1.2},
}


def get_gaze_config(subject: int) -> GazeConfig:
    """Return a GazeConfig with defaults + per-subject overrides applied."""
    overrides = SUBJECT_GAZE_OVERRIDES.get(subject, {})
    return GazeConfig(**overrides)


@dataclass
class LSTMConfig:
    seq_len: int = 10
    lstm_units: int = 128
    dense_units: tuple = (64, 32, 16)
    l1_reg: float = 1e-4
    epochs: int = 1500
    batch_size: int = 10
    learning_rate: float = 2e-4
    lr_decay_rate: float = 0.98
    lr_decay_steps: int = 1000
    early_stop_patience: int = 5000           # epochs without val_loss improvement before stopping
    fine_tune_lr: float = 2e-5              # 10× lower than initial LR
    fine_tune_epochs: int = 150
    fine_tune_batch_size: int = 32
    freeze_lstm: bool = True                # if True, freeze LSTM layer during fine-tuning
