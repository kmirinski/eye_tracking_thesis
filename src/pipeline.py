import numpy as np
from sklearn.model_selection import train_test_split

from processing.preprocessing import *
from processing.filtering import *
from processing.pupil_finding import *
from data.visualization import *
from utils import *
from data.loaders import EyeDataset
from processing.frame_detection import extract_pupil, extract_pupil_centers
from gaze_estimator import GazeEstimator
from tracking import PupilTracker
from config import FrameDetectionConfig, TrackingConfig, KDEConfig, GazeConfig


def run_pipeline(opt):
    frame_config = FrameDetectionConfig()
    tracking_config = TrackingConfig()
    kde_config = KDEConfig()
    gaze_config = GazeConfig()

    eye_dataset = EyeDataset(opt.data_dir, opt.subject, mode='stack')
    print('Collecting data of the left eye of subject ' + str(opt.subject))
    print('Loading data from ' + opt.data_dir)

    with timer("Collection"):
        eye_dataset.collect_data(eye=0)

    # with timer("Accumulation"):
    #     event_sets = accumulate_events(eye_dataset.event_list, n_events=tracking_config.num_events)

    with timer("Center extraction + Screen coordinates extraction"):
        pupil_centers = extract_pupil_centers(eye_dataset.frame_list, config=frame_config)
        screen_coords = np.array([(frame.row, frame.col) for frame in eye_dataset.frame_list])

        combined = np.hstack([pupil_centers, screen_coords])
        valid_mask = ~(np.all(combined[:, :2] < 5, axis=1) | np.all(combined[:, 2:] == 0, axis=1))
        combined = combined[valid_mask]

    for c in combined:
        print(c)

    pupil_centers, screen_coords = combined[:, :2], combined[:, 2:]

    test_val_ratio = 1 - gaze_config.train_ratio
    val_in_test_val = gaze_config.val_ratio / test_val_ratio

    pupil_train, pupil_test_val, screen_train, screen_test_val = train_test_split(
        pupil_centers, screen_coords, test_size=test_val_ratio, random_state=42
    )

    pupil_test, pupil_val, screen_test, screen_val = train_test_split(
        pupil_test_val, screen_test_val, test_size=val_in_test_val, random_state=42
    )

    print(f"Training set size: {len(pupil_train)}")
    print(f"Validation set size: {len(pupil_val)}")
    print(f"Test set size: {len(pupil_test)}")

    for deg in gaze_config.poly_degrees:
        print(f"\n--- Degree {deg} ---")
        gaze_estimator = GazeEstimator(degree=deg)

        gaze_estimator.fit(pupil_train, screen_train)

        val_metrics = gaze_estimator.evaluate(pupil_val, screen_val)
        print(f"Validation RMSE: {val_metrics['rmse']:.2f} pixels")
        print(f"Validation Mean Error: {val_metrics['mean_error']:.2f} pixels")


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
