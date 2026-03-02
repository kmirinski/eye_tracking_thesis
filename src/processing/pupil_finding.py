from KDEpy import FFTKDE

import numpy as np
from config import KDEConfig
from processing.preprocessing import event_to_image
from processing.filtering import generate_and_apply_masks

def locate_pupil_center_kde(img: np.ndarray, config: KDEConfig = None,
                        img_width: int = 346, img_height: int = 260) -> tuple[int, int, np.ndarray]:
    if config is None:
        config = KDEConfig()

    rows, cols = np.nonzero(img)
    if len(rows) == 0:
        raise ValueError("No events found in the image")

    coords = np.column_stack([cols, rows])

    x_grid = np.arange(0, img_width)
    y_grid = np.arange(0, img_height)

    grid_points = np.array(np.meshgrid(x_grid, y_grid)).T.reshape(-1, 2)

    kde_large = FFTKDE(bw=config.bandwidth_large).fit(coords)
    density_large = kde_large.evaluate(grid_points)
    density_large = density_large.reshape(img_height, img_width)

    kde_small = FFTKDE(bw=config.bandwidth_small).fit(coords)
    density_small = kde_small.evaluate(grid_points)
    density_small = density_small.reshape(img_height, img_width)

    density_donut = density_large - density_small
    density_donut = np.maximum(density_donut, 0)
    center_y, center_x = np.unravel_index(np.argmax(density_donut), density_donut.shape)

    return int(center_x), int(center_y), density_donut

def segment_pupil_events(img: np.ndarray, center_x: int, center_y: int, config: KDEConfig = None) -> np.ndarray:
    if config is None:
        config = KDEConfig()

    rows, cols = np.indices(img.shape)
    distances = np.sqrt((cols - center_x)**2 + (rows - center_y)**2)
    pupil_mask = (distances <= config.pupil_radius).astype(np.uint8)

    return pupil_mask

def extract_pupil_and_iris(img_pupil_iris: np.ndarray, config: KDEConfig = None) -> dict:
    if config is None:
        config = KDEConfig()

    center_x, center_y, density_map = locate_pupil_center_kde(img_pupil_iris, config=config)

    pupil_mask = segment_pupil_events(img_pupil_iris, center_x, center_y, config=config)
    has_events = (img_pupil_iris > 0).astype(np.uint8)
    iris_mask = has_events & (~pupil_mask.astype(bool)).astype(np.uint8)

    return {
        'center_x': center_x,
        'center_y': center_y,
        'pupil_mask': pupil_mask,
        'iris_mask': iris_mask,
        'density_map': density_map
    }

def generate_eye_images(neg_sets, pos_sets, event_sets, img_idxs, kde_config: KDEConfig = None):
    if kde_config is None:
        kde_config = KDEConfig()

    n = len(img_idxs)
    images = [None] * (2 * n)
    centers = [None] * n

    for idx, i in enumerate(img_idxs):
        img_neg = event_to_image(neg_sets[i])
        img_pos = event_to_image(pos_sets[i])
        img = event_to_image(event_sets[i])
        pupil_iris = generate_and_apply_masks(img_neg, img_pos, img)
        center_x, center_y, _ = locate_pupil_center_kde(pupil_iris, config=kde_config)

        images[idx] = (img, f"Image {i}")
        images[n + idx] = (pupil_iris, f"Image {i} extracted")
        centers[idx] = (center_x, center_y)

    return images, centers
