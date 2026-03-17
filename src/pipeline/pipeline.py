import numpy as np

from data.visualization import write_ellipse_video, browse_ellipse_frames, browse_pupil_extraction, plot_pupil_centers_over_time_all, plot_pupil_centers_over_time, plot_pupil_diffs
from utils import timer
from data.loaders import EyeDataset
from processing.frame_detection import extract_pupil_centers, extract_pupil
from config import FrameDetectionConfig, GazeConfig
from pipeline.runners import run_regressor, run_lstm


def _find_sections(screen_chron):
    '''
    Find contiguous non-zero sections in chronological screen_coords.

    Using the ebv-eye dataset, the sections are expected to be two:
        1. Saccadic movements
        2. Smooth pursuit movements
    '''
    n = len(screen_chron)
    non_zero = ~np.all(screen_chron == 0, axis=1)
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
    return sections

def relabel_transition_frames(pupil_centers, screen_coords, threshold, max_relabel_frames):
    '''
    For each label change (after the first) in the saccadic section:
        - Phase A: while the eye is stable (dist to last_valid < threshold), relabel frames to old label
        - Phase B: while the eye is moving (dist >= threshold), mark frames to discard (saccade)
        - Phase C: remaining frames keep the new label
    
    Blink frames (pupil == -1) in Phase A are relabeled (filtered anyway by basic mask).
    Blink frames in Phase B are discarded. Diffs are always vs. last valid pupil to avoid 
    false saccade triggers from blinks.

    Returns (screen_coords_relabled, saccade_discard_mask) in storage order.
    '''
    n = len(screen_coords)
    pupil_chron = pupil_centers[::-1]
    screen_chron = screen_coords[::-1].copy()
    saccade_mask_chron = np.zeros(n, dtype=bool)

    sections = _find_sections(screen_chron)
    if not sections:
        return screen_coords.copy(), saccade_mask_chron[::-1]

    sac_start, sac_end = sections[0]

    # Pass 1: detect label changes on ORIGINAL screen_chron
    change_points = []
    prev = screen_chron[sac_start].copy()
    for i in range(sac_start + 1, sac_end):
        if not np.array_equal(screen_chron[i], prev):
            change_points.append((i, prev.copy()))
            prev = screen_chron[i].copy()
    
    # Pass 2: process each label change
    for change_idx, old_label in change_points:
        last_valid = None
        for k in range(change_idx - 1, sac_start - 1, -1):
            if not np.all(pupil_chron[k] == -1):
                last_valid = pupil_chron[k].copy()
                break
        if last_valid is None:
            continue

        # Phase A: pre-saccade - relabel while eye is stable
        m = change_idx
        while m < sac_end and (m - change_idx) < max_relabel_frames:
            if np.all(pupil_chron[m] == -1):
                screen_chron[m] = old_label     # blink: relabel (filtered by basic mask anyway)
                m += 1
                continue
            dist = np.linalg.norm(pupil_chron[m] - last_valid)
            if dist < threshold:
                screen_chron[m] = old_label
                last_valid = pupil_chron[m].copy()
                m += 1
            else:
                break   # saccade detected

        # Phase B: saccade - discard frames where eye is actively moving
        while m < sac_end:
            if np.all(pupil_chron[m] == -1):
                saccade_mask_chron[m] = True    # blink during saccade: discard
                m += 1
                continue
            dist = np.linalg.norm(pupil_chron[m] - last_valid)
            if dist >= threshold:
                saccade_mask_chron[m] = True
                last_valid = pupil_chron[m].copy()
                m += 1
            else:
                break   # eye settled, Phase C begins (screen_chron unchanged = new label)

    return screen_chron[::-1], saccade_mask_chron[::-1]



def detect_blink_artifacts(pupil_chron, threshold):
    """
    Detect frames where pupil detection succeeded but returned a wrong position
    due to a partial blink (eyelid detected as pupil, etc.).

    Criterion: a valid frame is a blink artifact if
      - its displacement from the previous valid frame is > threshold  (large jump in)
      - AND the next valid frame's displacement from it is > threshold  (large jump out)

    This spike-and-return pattern distinguishes blink artifacts from saccades,
    which stay at the new position.

    Returns a boolean mask in chronological order (True = artifact, discard).
    """
    n = len(pupil_chron)
    artifact = np.zeros(n, dtype=bool)

    # Precompute index of next valid frame for each position
    next_valid = np.full(n, -1, dtype=int)
    last = -1
    for i in range(n - 1, -1, -1):
        if not np.all(pupil_chron[i] == -1):
            last = i
        next_valid[i] = last

    # Compute displacement from previous valid frame
    dist_in = np.zeros(n)
    last_valid_pos = None
    for i in range(n):
        if np.all(pupil_chron[i] == -1):
            continue
        if last_valid_pos is not None:
            dist_in[i] = np.linalg.norm(pupil_chron[i] - last_valid_pos)
        last_valid_pos = pupil_chron[i].copy()

    # Flag spike-and-return frames
    for i in range(n):
        if np.all(pupil_chron[i] == -1):
            continue
        j = next_valid[i + 1] if i + 1 < n else -1
        if j == -1:
            continue
        dist_out = np.linalg.norm(pupil_chron[j] - pupil_chron[i])
        if dist_in[i] > threshold and dist_out > threshold:
            artifact[i] = True

    return artifact


def build_valid_mask(pupil_centers, screen_coords, skip_frames, skip_label_changes=True,
                     blink_artifact_threshold=None):
    n = len(screen_coords)

    # Work in chronological order (arrays are stored reversed)
    pupil_chron = pupil_centers[::-1]
    screen_chron = screen_coords[::-1]

    valid = np.ones(n, dtype=bool)
    valid &= ~np.all(pupil_chron == -1, axis=1)     # failed detections
    valid &= ~np.all(screen_chron == 0, axis=1)     # zero-coord frames

    if blink_artifact_threshold is not None:
        valid &= ~detect_blink_artifacts(pupil_chron, blink_artifact_threshold)

    sections = _find_sections(screen_chron)

    # Section 0: saccades — always skip first N frames; optionally skip after every target change
    if len(sections) >= 1:
        sac_start, sac_end = sections[0]
        valid[sac_start:sac_start + skip_frames] = False
        if skip_label_changes:
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



def pupil_center_extraction_stage(eye_dataset: EyeDataset, gaze_config: GazeConfig, opt):
    corner = 'upper_right' if opt.eye == 'left' else 'upper_left'
    frame_config = FrameDetectionConfig(triangle_corner=corner)

    pupil_centers, ellipses = extract_pupil_centers(eye_dataset.frame_list, config=frame_config)
    screen_coords = np.array([(frame.row, frame.col) for frame in eye_dataset.frame_list])

    if opt.relabel:
        screen_coords, saccade_mask = relabel_transition_frames(
            pupil_centers, screen_coords,
            threshold=gaze_config.relabel_diff_threshold,
            max_relabel_frames=gaze_config.relabel_max_frames,
        )
        valid_mask = build_valid_mask(pupil_centers, screen_coords,
                                      skip_frames=gaze_config.saccade_skip_frames,
                                      skip_label_changes=False,
                                      blink_artifact_threshold=gaze_config.blink_artifact_threshold)
        valid_mask &= ~saccade_mask
    else:
        valid_mask = build_valid_mask(pupil_centers, screen_coords,
                                      skip_frames=gaze_config.saccade_skip_frames,
                                      blink_artifact_threshold=gaze_config.blink_artifact_threshold)

    if opt.video:
        write_ellipse_video(eye_dataset.frame_list, ellipses, screen_coords)

    if opt.f_browse:
        sm = saccade_mask if opt.relabel else None
        browse_ellipse_frames(eye_dataset.frame_list, ellipses, screen_coords, saccade_mask=sm)

    if opt.pe_browse:
        browse_pupil_extraction(eye_dataset.frame_list, frame_config, screen_coords)

    if opt.pe_plots:
        plot_pupil_centers_over_time_all(pupil_centers, screen_coords, valid_mask)
        plot_pupil_centers_over_time(pupil_centers, screen_coords, valid_mask)

    if opt.diff_plot:
        plot_pupil_diffs(pupil_centers, screen_coords)

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

    with timer("Model training"):
        if opt.model == 'regressor':
            run_regressor(pupil_centers, screen_coords, valid_mask, gaze_config, opt)
        elif opt.model == 'lstm':
            run_lstm(ellipses, screen_coords, valid_mask, gaze_config, opt)
