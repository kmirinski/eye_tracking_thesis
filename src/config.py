from dataclasses import dataclass

@dataclass
class FrameDetectionConfig:
    threshold: int = 20
    morph_kernel_size: int = 3
    min_aspect_ratio: float = 0.37
    min_axis_px: int = 0
    max_axis_px: int = 120
    center_min: tuple = (0, 70)   # (x_min, y_min) accepted pupil center in px
    center_max: tuple = (260, 150)  # (x_max, y_max) accepted pupil center in px
    min_ellipse_area: float = 210  # π * (w/2) * (h/2) in px²

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
    poly_degrees: list = (3, 4, 5, 6, 7, 8, 9, 10)
    train_ratio: float = 0.85
    val_ratio: float = 0.075
    saccade_skip_frames: int = 20
