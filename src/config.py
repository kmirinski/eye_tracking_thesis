from dataclasses import dataclass

@dataclass
class FrameDetectionConfig:
    threshold: int = 10
    morph_kernel_size: int = 2
    min_aspect_ratio: float = 0.32
    min_axis_px: int = 0            # Currently not in use
    max_axis_px: int = 120          # Currently not in use
    center_min: tuple = None        # (x_min, y_min) accepted pupil center in px; None = no limit
    center_max: tuple = None        # (x_max, y_max) accepted pupil center in px; None = no limit
    triangle_corner: str = None     # 'upper_right' (left eye) or 'upper_left' (right eye)
    triangle_size: int = 80         # leg length in px of the corner triangle to exclude
    min_ellipse_area: float = 210   # π * (w/2) * (h/2) in px²

@dataclass
class TrackingConfig:
    num_events: int = 2000
    fit_threshold: float = 0.8
    roi_expansion: float = 1.5

@dataclass
class KDEConfig:
    pupil_radius: int = 30
    bandwidth_large: float = 35.0
    bandwidth_small: float = 15.0

@dataclass
class GazeConfig:
    poly_degrees: list = (5, 6, 7, 8)
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    saccade_skip_frames: int = 20
    relabel_diff_threshold: float = 1.5  # px; eye displacement below this = stable fixation
    relabel_max_frames: int = 20         # safety cap: never relabel more than this many frames per label change
    post_blink_skip_frames: int = 1      # valid frames to discard after each blink run
    post_saccade_stability_window: int = 6  # consecutive stable frames required before Phase C begins

@dataclass
class LSTMConfig:
    seq_len: int = 10
    lstm_units: int = 128
    dense_units: tuple = (64, 32, 16)
    l1_reg: float = 1e-4
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
