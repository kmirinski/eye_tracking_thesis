from KDEpy import FFTKDE

import numpy as np

def locate_pupil_center_kde(img: np.ndarray, pupil_radius: int = 30, 
                        bandwidth_large: float = 35.0, bandwidth_small: float = 15.0,
                        img_width: int = 346, img_height: int = 260) -> tuple[int, int, np.ndarray]:
    rows, cols = np.nonzero(img)
    if len(rows) == 0:
        raise ValueError("No events found in the image")
    
    coords = np.column_stack([cols, rows])

    x_grid = np.arange(0, img_width)
    y_grid = np.arange(0, img_height)

    grid_points = np.array(np.meshgrid(x_grid, y_grid)).T.reshape(-1, 2)

    kde_large = FFTKDE(bw=bandwidth_large).fit(coords)
    density_large = kde_large.evaluate(grid_points)
    density_large = density_large.reshape(img_height, img_width)
    
    kde_small = FFTKDE(bw=bandwidth_small).fit(coords)
    density_small = kde_small.evaluate(grid_points)
    density_small = density_small.reshape(img_height, img_width)

    density_donut = density_large - density_small
    density_donut = np.maximum(density_donut, 0)
    center_y, center_x = np.unravel_index(np.argmax(density_donut), density_donut.shape)

    return int(center_x), int(center_y), density_donut

def segment_pupil_events(img: np.ndarray, center_x: int, center_y: int, pupil_radius: int = 30) -> np.ndarray:
    rows, cols = np.indices(img.shape)
    distances = np.sqrt((cols - center_x)**2 + (rows - center_y)**2)
    pupil_mask = (distances <= pupil_radius).astype(np.uint8)
    
    return pupil_mask

def extract_pupil_and_iris(img_pupil_iris: np.ndarray, pupil_radius: int = 30, bandwidth_large: float = 35.0, bandwidth_small: float = 15.0) -> dict:
    center_x, center_y, density_map = locate_pupil_center_kde(
        img_pupil_iris, 
        pupil_radius=pupil_radius,
        bandwidth_large=bandwidth_large,
        bandwidth_small=bandwidth_small
    )

    pupil_mask = segment_pupil_events(img_pupil_iris, center_x, center_y, pupil_radius)
    has_events = (img_pupil_iris > 0).astype(np.uint8)
    iris_mask = has_events & (~pupil_mask.astype(bool)).astype(np.uint8)
    
    return {
        'center_x': center_x,
        'center_y': center_y,
        'pupil_mask': pupil_mask,
        'iris_mask': iris_mask,
        'density_map': density_map
    }



