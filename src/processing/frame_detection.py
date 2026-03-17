import cv2
import numpy as np
from tqdm import tqdm
from data.loaders import Frame
from config import FrameDetectionConfig

def ellipse_area(ellipse) -> float:
    """Return the area of a cv2 ellipse (axes are full diameters)."""
    w, h = ellipse[1]
    return np.pi * (w / 2) * (h / 2)


def _run_detection(frame: Frame, config: FrameDetectionConfig):
    """Run the full detection pipeline and return all intermediate stages."""
    img = cv2.imread(frame.img)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    _, binary = cv2.threshold(gray, config.threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * config.morph_kernel_size + 1, 2 * config.morph_kernel_size + 1))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    best_ellipse = None
    best_contour = None
    best_area = 0

    for cnt in contours:
        if len(cnt) < 5:
            continue

        ellipse = cv2.fitEllipse(cnt)
        minor, major = ellipse[1][0], ellipse[1][1]
        if minor > major:
            minor, major = major, minor
        aspect_ratio = minor / major if major > 0 else 0

        cx, cy = ellipse[0]
        area = ellipse_area(ellipse)

        valid = (aspect_ratio >= config.min_aspect_ratio and area >= config.min_ellipse_area)

        if config.center_min is not None:
            valid = valid and config.center_min[0] < cx <= config.center_max[0]
            valid = valid and config.center_min[1] < cy <= config.center_max[1]

        if config.triangle_corner == 'upper_right':
            W = img.shape[1]
            in_triangle = (cx - cy >= W - config.triangle_size)
        elif config.triangle_corner == 'upper_left':
            in_triangle = (cx + cy <= config.triangle_size)
        else:
            in_triangle = False
        valid = valid and not in_triangle

        if valid:
            contour_area = cv2.contourArea(cnt)
            if contour_area > best_area:
                best_ellipse = ellipse
                best_contour = cnt
                best_area = contour_area

    contour_img = np.zeros_like(opened)
    cv2.drawContours(contour_img, contours, -1, 255, 1)

    selected_points = best_contour.reshape(-1, 2) if best_contour is not None else np.zeros((0, 2), dtype=np.int32)

    return img, binary, opened, contour_img, selected_points, best_ellipse


def extract_pupil(frame: Frame, config: FrameDetectionConfig = None):
    if config is None:
        config = FrameDetectionConfig()

    _, _, _, _, _, best_ellipse = _run_detection(frame, config)

    if best_ellipse is not None:
        return np.array(best_ellipse[0], dtype=np.float32), best_ellipse
    else:
        return np.array((-1, -1), dtype=np.float32), None


def extract_pupil_centers(frame_list, config: FrameDetectionConfig = None):
    n = len(frame_list)
    pupil_centers = np.zeros((n, 2))
    ellipses = [None] * n
    # Index 0 is invalid, so we start from 1
    for idx in tqdm(range(1, n)):
        center, ellipse = extract_pupil(frame_list[idx], config=config, visualize=False)
        pupil_centers[idx] = center
        ellipses[idx] = ellipse
    return pupil_centers, ellipses
