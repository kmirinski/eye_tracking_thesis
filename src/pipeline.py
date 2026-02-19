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


def build_valid_mask(pupil_centers, screen_coords, skip_frames=15):
    n = len(screen_coords)

    # Work in chronological order (arrays are stored reversed)
    pupil_chron = pupil_centers[::-1]
    screen_chron = screen_coords[::-1]

    valid = np.ones(n, dtype=bool)
    low, high = np.array([100, 80]), np.array([260, 150])
    valid &= np.all((pupil_chron > low) & (pupil_chron <= high), axis=1)
    valid &= ~np.all(screen_chron == 0, axis=1)     # zero-coord frames

    non_zero = ~np.all(screen_chron == 0, axis=1)

    # Find contiguous non-zero sections
    sections = []
    in_section = False
    for i in range(n):
        if non_zero[i] and not in_section:
            start = i
            in_section = True
        elif not non_zero[i] and in_section:
            sections.append((start, i))
            in_section = False
    if in_section:
        sections.append((start, n))

    # Section 0: saccades — skip first N frames at start and after every target change
    if len(sections) >= 1:
        sac_start, sac_end = sections[0]
        valid[sac_start:sac_start + skip_frames] = False
        prev = screen_chron[sac_start]
        for i in range(sac_start + 1, sac_end):
            if not np.array_equal(screen_chron[i], prev):
                valid[i:i + skip_frames] = False
                prev = screen_chron[i]

    # Section 1: smooth pursuit — skip only the first N frames
    if len(sections) >= 2:
        sp_start, _ = sections[1]
        valid[sp_start:sp_start + skip_frames] = False

    removed_basic = n - np.sum(~np.all(pupil_chron < 5, axis=1) | ~np.all(screen_chron == 0, axis=1))
    removed_total = n - np.sum(valid)
    print(f"Frames removed (basic filter): {n - np.count_nonzero(~np.all(pupil_chron < 5, axis=1) & ~np.all(screen_chron == 0, axis=1))}")
    print(f"Frames removed (temporal skip): {removed_total - (n - np.count_nonzero(~np.all(pupil_chron < 5, axis=1) & ~np.all(screen_chron == 0, axis=1)))}")
    print(f"Frames removed (total): {removed_total} / {n}")

    # Return mask in the original (reversed) array order
    return valid[::-1]


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
        pupil_centers, ellipses = extract_pupil_centers(eye_dataset.frame_list, config=frame_config)
        screen_coords = np.array([(frame.row, frame.col) for frame in eye_dataset.frame_list])
        write_ellipse_video(eye_dataset.frame_list, ellipses, screen_coords)

        valid_mask = build_valid_mask(pupil_centers, screen_coords, skip_frames=gaze_config.saccade_skip_frames)
        
        plot_pupil_centers_over_time_all(pupil_centers, screen_coords, valid_mask)
        plot_pupil_centers_over_time(pupil_centers, screen_coords, valid_mask)
        
        combined = np.hstack([pupil_centers, screen_coords])
        combined = combined[valid_mask]
        combined = np.round(combined, 2)

    # for c in combined:
    #     print(c)

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
