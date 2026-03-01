import cv2
import numpy as np
from tqdm import tqdm
from data.loaders import Frame
from config import FrameDetectionConfig

def ellipse_area(ellipse) -> float:
    """Return the area of a cv2 ellipse (axes are full diameters)."""
    w, h = ellipse[1]
    return np.pi * (w / 2) * (h / 2)


def extract_pupil(frame: Frame, config: FrameDetectionConfig = None, visualize=True):
    if config is None:
        config = FrameDetectionConfig()

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
        # print(aspect_ratio)

        cx, cy = ellipse[0]
        area = ellipse_area(ellipse)
        valid = (
            aspect_ratio >= config.min_aspect_ratio and
            major >= config.min_axis_px and
            major <= config.max_axis_px and
            area >= config.min_ellipse_area and
            config.center_min[0] < cx <= config.center_max[0] and
            config.center_min[1] < cy <= config.center_max[1]
        )

        if valid:
            contour_area = cv2.contourArea(cnt)
            if contour_area > best_area:
                best_ellipse = ellipse
                best_contour = cnt
                best_area = contour_area

    contour_img = np.zeros_like(opened)
    cv2.drawContours(contour_img, contours, -1, 255, 1)

    selected_points = best_contour.reshape(-1, 2) if best_contour is not None else np.zeros((0, 2), dtype=np.int32)

    if visualize:
        visualize_detection(img, binary, opened, contour_img, selected_points, best_ellipse)

    if best_ellipse is not None:
        return np.array(best_ellipse[0], dtype=np.float32), best_ellipse
    else:
        return np.array((-1, -1), dtype=np.float32), None


def visualize_detection(img, binary, opened, contour_img, candidate_points, ellipse):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=150)

    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Binarized image
    axes[0, 1].imshow(binary, cmap='gray')
    axes[0, 1].set_title('Binarized (Hθ)')
    axes[0, 1].axis('off')

    # After morphological opening
    axes[0, 2].imshow(opened, cmap='gray')
    axes[0, 2].set_title('After Opening (◦ Sσ)')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(contour_img, cmap='gray')
    axes[1, 0].set_title('Contours')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(img, cmap='gray')
    if len(candidate_points) > 0:
        axes[1, 1].scatter(candidate_points[:, 0], candidate_points[:, 1],
                          c='red', s=1, alpha=0.5)
    axes[1, 1].set_title(f'Selected Contour ({len(candidate_points)} points)')
    axes[1, 1].axis('off')

    img_with_ellipse = img.copy()
    if ellipse is not None:
        cv2.ellipse(img_with_ellipse, ellipse, 255, 1)

    axes[1, 2].imshow(img_with_ellipse, cmap='gray')
    if ellipse is not None:
        center = (int(ellipse[0][0]), int(ellipse[0][1]))
        axes[1, 2].set_title(f'Fitted Ellipse\nCenter: {center}')
    else:
        axes[1, 2].set_title('No Ellipse Fitted')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

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
