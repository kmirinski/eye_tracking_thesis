import cv2
import numpy as np
from tqdm import tqdm
from data.loaders import Frame
from config import FrameDetectionConfig

def extract_pupil(frame: Frame, config: FrameDetectionConfig = None, visualize=True):
    if config is None:
        config = FrameDetectionConfig()

    img = cv2.imread(frame.img)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    height, width = gray.shape

    _, binary = cv2.threshold(gray, config.threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * config.morph_kernel_size + 1, 2 * config.morph_kernel_size + 1))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    edges = cv2.Canny(opened, 50, 100)

    candidate_points = np.column_stack(np.where(edges > 0))
    candidate_points = np.flip(candidate_points, axis=1)

    if len(candidate_points) > 0:
        mask = (
            (candidate_points[:, 0] >= config.edge_margin) &
            (candidate_points[:, 0] < width - config.edge_margin) &
            (candidate_points[:, 1] >= config.edge_margin) &
            (candidate_points[:, 1] < height - config.edge_margin)
        )
        candidate_points_filtered = candidate_points[mask]
    else:
        candidate_points_filtered = candidate_points

    if len(candidate_points_filtered) >= 5:
        # print(f"img: {frame.img}, {len(candidate_points_filtered)}")
        # if len(candidate_points_filtered) < 5:
        #     ellipse = cv2.fitEllipse(candidate_points.astype(np.float32)) 
        # else:
        ellipse = cv2.fitEllipse(candidate_points_filtered.astype(np.float32)) 
        if visualize:
            visualize_detection(img, binary, opened, edges, candidate_points_filtered, ellipse)
        return np.array(ellipse[0], dtype=np.float32), ellipse
    else:
        if visualize:
            visualize_detection(img, binary, opened, edges, candidate_points_filtered, None)
        return np.array((-1, -1), dtype=np.float32), None

    

def visualize_detection(img, binary, opened, edges, candidate_points, ellipse): 
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

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

    axes[1, 0].imshow(edges, cmap='gray')
    axes[1, 0].set_title('Edge Detection (K)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(img, cmap='gray')
    if len(candidate_points) > 0:
        axes[1, 1].scatter(candidate_points[:, 0], candidate_points[:, 1], 
                          c='red', s=1, alpha=0.5)
    axes[1, 1].set_title(f'Candidate Points ({len(candidate_points)} points)')
    axes[1, 1].axis('off')

    img_with_ellipse = img.copy()
    if ellipse is not None:
        # Draw ellipse
        cv2.ellipse(img_with_ellipse, ellipse, 255, 1)
        roi = ((ellipse[0][0], ellipse[0][1]), (ellipse[1][0] + 5, ellipse[1][1] + 5), ellipse[2])
        # cv2.ellipse(img_with_ellipse, roi, 255, 1)

        # Draw center
        # center = (int(ellipse[0][0]), int(ellipse[0][1]))
        # cv2.circle(img_with_ellipse, center, 3, 200, -1)
    
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
    for idx in tqdm(range(1, n)):
        center, ellipse = extract_pupil(frame_list[idx], config=config, visualize=False)
        pupil_centers[idx] = center
        ellipses[idx] = ellipse
    return pupil_centers, ellipses

