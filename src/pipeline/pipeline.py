import numpy as np

from data.visualization import write_ellipse_video, browse_ellipse_frames, plot_pupil_centers_over_time_all, plot_pupil_centers_over_time
from utils import timer
from data.loaders import EyeDataset
from processing.frame_detection import extract_pupil_centers
from config import FrameDetectionConfig, GazeConfig
from pipeline.runners import run_regressor, run_lstm



def build_valid_mask(pupil_centers, screen_coords, skip_frames=15):
    n = len(screen_coords)

    # Work in chronological order (arrays are stored reversed)
    pupil_chron = pupil_centers[::-1]
    screen_chron = screen_coords[::-1]

    valid = np.ones(n, dtype=bool)
    valid &= ~np.all(pupil_chron == -1, axis=1)     # failed detections
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

    # Section 1: smooth pursuit — exclude entirely (eye lags target, corrupts mapping)
    if len(sections) >= 2:
        sp_start, sp_end = sections[1]
        valid[sp_start:sp_end] = False

    basic_removed = np.sum(np.all(pupil_chron == -1, axis=1) | np.all(screen_chron == 0, axis=1))
    removed_total = n - np.sum(valid)
    print(f"Frames removed (basic filter): {basic_removed}")
    print(f"Frames removed (temporal skip): {removed_total - basic_removed}")
    print(f"Frames removed (total): {removed_total} / {n}")

    # Return mask in the original (reversed) array order
    return valid[::-1]



def pupil_center_extraction_stage(eye_dataset: EyeDataset, gaze_config, opt):
    frame_config = FrameDetectionConfig()

    pupil_centers, ellipses = extract_pupil_centers(eye_dataset.frame_list, config=frame_config)
    screen_coords = np.array([(frame.row, frame.col) for frame in eye_dataset.frame_list])

    if opt.video:
        write_ellipse_video(eye_dataset.frame_list, ellipses, screen_coords)

    if opt.frame_browser:
        browse_ellipse_frames(eye_dataset.frame_list, ellipses, screen_coords)

    valid_mask = build_valid_mask(pupil_centers, screen_coords, skip_frames=gaze_config.saccade_skip_frames)

    if opt.pe_plots:
        plot_pupil_centers_over_time_all(pupil_centers, screen_coords, valid_mask)
        plot_pupil_centers_over_time(pupil_centers, screen_coords, valid_mask)

    return pupil_centers, screen_coords, ellipses, valid_mask



def run_pipeline(opt):
    eye_dataset = EyeDataset(opt.data_dir, opt.subject, mode='stack')
    eye_index = 0 if opt.eye == 'left' else 1

    print(f'Collecting data of the {opt.eye} eye of subject {opt.subject}')
    print('Loading data from ' + opt.data_dir)

    gaze_config = GazeConfig()

    with timer("Collection"):
        eye_dataset.collect_data(eye=eye_index)

    with timer("Center extraction + Screen coordinates extraction"):
        pupil_centers, screen_coords, ellipses, valid_mask = pupil_center_extraction_stage(eye_dataset, gaze_config, opt)

    if opt.model == 'regressor':
        run_regressor(pupil_centers, screen_coords, valid_mask, gaze_config, opt)
    elif opt.model == 'lstm':
        run_lstm(ellipses, screen_coords, valid_mask, gaze_config, opt)
