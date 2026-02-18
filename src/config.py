from dataclasses import dataclass

@dataclass
class FrameDetectionConfig:
    threshold: int = 20
    morph_kernel_size: int = 6
    edge_margin: int = 30

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
    poly_degrees: list = (1, 2, 3, 4, 5)
    train_ratio: float = 0.85
    val_ratio: float = 0.075
