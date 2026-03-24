import numpy as np

from data.visualization import write_ellipse_video, browse_ellipse_frames, browse_pupil_extraction, plot_pupil_centers_over_time_all, plot_pupil_centers_over_time, plot_pupil_diffs
from utils import timer
from data.loaders import EyeDataset
from processing.frame_detection import extract_pupil_centers
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

def relabel_transition_frames(pupil_centers, screen_coords, threshold, max_relabel_frames,
                               stability_window=0):
    '''
    For each label change (after the first) in the saccadic section:
        - Phase A: while the eye is stable (dist to last_valid < threshold), relabel frames to old label
        - Phase B: while the eye is moving (dist >= threshold), mark frames to discard (saccade)
        - Phase D: discard until stability_window consecutive frames are all below threshold
        - Phase C: remaining frames keep the new label

    Blink frames (pupil == -1) in Phase A are relabeled (filtered anyway by basic mask).
    Blink frames in Phase B/D are discarded. Diffs are always vs. last valid pupil to avoid
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
                break   # large movement ended, Phase D begins

        # Phase D: stability window — discard until stability_window consecutive frames are below threshold
        stable_count = 0
        while m < sac_end and stable_count < stability_window:
            if np.all(pupil_chron[m] == -1):
                saccade_mask_chron[m] = True   # blink resets window
                stable_count = 0
                m += 1
                continue
            dist = np.linalg.norm(pupil_chron[m] - last_valid)
            if dist >= threshold:
                saccade_mask_chron[m] = True   # spike: discard and reset window
                last_valid = pupil_chron[m].copy()
                stable_count = 0
                m += 1
            else:
                saccade_mask_chron[m] = True   # within stability window: discard
                last_valid = pupil_chron[m].copy()
                stable_count += 1
                m += 1
        # Phase C begins at m (eye settled)

    return screen_chron[::-1], saccade_mask_chron[::-1]


def build_valid_mask(blink_mask, screen_coords, skip_frames,
                     saccade_mask=None, skip_label_changes=True, post_blink_skip_frames=1):
    n = len(screen_coords)

    # Work in chronological order (arrays are stored reversed)
    blink_mask_chron = blink_mask[::-1]
    screen_chron = screen_coords[::-1]

    # Post-blink skip: discard first N valid frames after each blink run
    post_blink_chron = np.zeros(n, dtype=bool)
    post_blink_chron[1:] = blink_mask_chron[:-1] & ~blink_mask_chron[1:]
    for _ in range(post_blink_skip_frames - 1):
        post_blink_chron[1:] |= post_blink_chron[:-1] & ~blink_mask_chron[1:]

    valid = np.ones(n, dtype=bool)
    valid &= ~blink_mask_chron                          # failed detections
    valid &= ~post_blink_chron                          # first frame after blink run
    valid &= ~np.all(screen_chron == 0, axis=1)        # zero-coord frames

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

    if saccade_mask is not None:
        valid &= ~saccade_mask[::-1]

    basic_removed = np.sum(blink_mask_chron | post_blink_chron | np.all(screen_chron == 0, axis=1))
    removed_total = n - np.sum(valid)
    print(f"Frames removed (basic filter): {basic_removed}")
    print(f"Frames removed (temporal skip): {removed_total - basic_removed}")
    print(f"Frames removed (total): {removed_total} / {n}")

    # Return mask in the original (reversed) array order
    return valid[::-1]


def pupil_extraction_stage(eye_dataset: EyeDataset, frame_config: FrameDetectionConfig):
    pupil_centers, ellipses = extract_pupil_centers(eye_dataset.frame_list, config=frame_config)
    screen_coords = np.array([(frame.row, frame.col) for frame in eye_dataset.frame_list])
    return pupil_centers, ellipses, screen_coords


def noise_flagging_stage(pupil_centers):
    """
    Flag noisy frames. Currently: frames where pupil detection failed (returned -1).
    Returns a boolean mask in storage order (True = noisy/blink, discard).
    """
    return np.all(pupil_centers == -1, axis=1)


def relabeling_stage(pupil_centers, screen_coords, gaze_config: GazeConfig):
    """Relabel pre-saccade frames to previous label; discard active saccade frames."""
    return relabel_transition_frames(
        pupil_centers, screen_coords,
        threshold=gaze_config.relabel_diff_threshold,
        max_relabel_frames=gaze_config.relabel_max_frames,
        stability_window=gaze_config.post_saccade_stability_window,
    )


def compute_phase_labels(screen_coords_original_chron, screen_coords_relabeled_chron,
                         saccade_mask_chron, blink_mask_chron):
    """
    Classify each frame in the saccade section as 'A', 'B', 'C', or 'none'.
    All inputs in chronological order.

    Phase A: pre-saccade frame that was relabeled to the previous target
    Phase B: active saccade frame (discarded)
    Phase C: frame kept with the new label
    """
    n = len(screen_coords_original_chron)
    phase = np.full(n, 'none', dtype=object)

    sections = _find_sections(screen_coords_original_chron)
    if not sections:
        return phase
    sac_start, sac_end = sections[0]

    for i in range(sac_start, sac_end):
        if saccade_mask_chron[i]:
            phase[i] = 'B'
        elif not np.array_equal(screen_coords_original_chron[i], screen_coords_relabeled_chron[i]):
            phase[i] = 'A'
        else:
            phase[i] = 'C'

    return phase


def run_pipeline(opt):
    eye_dataset = EyeDataset(opt.data_dir, opt.subject, mode='stack')
    eye_index = 0 if opt.eye == 'left' else 1
    corner = 'upper_right' if opt.eye == 'left' else 'upper_left'
    frame_config = FrameDetectionConfig(triangle_corner=corner)
    gaze_config = GazeConfig()

    print(f'Collecting data of the {opt.eye} eye of subject {opt.subject}')
    print('Loading data from ' + opt.data_dir)

    with timer("Collection"):
        eye_dataset.collect_data(eye=eye_index)

    with timer("Pupil extraction"):
        pupil_centers, ellipses, screen_coords = pupil_extraction_stage(eye_dataset, frame_config)

    with timer("Noise flagging"):
        blink_mask = noise_flagging_stage(pupil_centers)

    saccade_mask = None
    screen_coords_original = screen_coords.copy()
    if opt.relabel:
        with timer("Relabeling"):
            screen_coords, saccade_mask = relabeling_stage(pupil_centers, screen_coords, gaze_config)

    valid_mask = build_valid_mask(
        blink_mask, screen_coords,
        skip_frames=gaze_config.saccade_skip_frames,
        saccade_mask=saccade_mask,
        skip_label_changes=not opt.relabel,
        post_blink_skip_frames=gaze_config.post_blink_skip_frames,
    )

    if opt.relabel_diag and opt.relabel:
        from data.visualization import plot_relabeling_diagnostic
        phase_labels = compute_phase_labels(
            screen_coords_original[::-1],
            screen_coords[::-1],
            saccade_mask[::-1],
            blink_mask[::-1],
        )
        plot_relabeling_diagnostic(pupil_centers[::-1], screen_coords_original[::-1],
                                   phase_labels, blink_mask[::-1])

    if opt.video:
        write_ellipse_video(eye_dataset.frame_list, ellipses, screen_coords)

    if opt.f_browse:
        browse_ellipse_frames(eye_dataset.frame_list, ellipses, screen_coords, saccade_mask=saccade_mask)

    if opt.pe_browse:
        browse_pupil_extraction(eye_dataset.frame_list, frame_config, screen_coords)

    if opt.pe_plots:
        plot_pupil_centers_over_time_all(pupil_centers, screen_coords, valid_mask)
        plot_pupil_centers_over_time(pupil_centers, screen_coords, valid_mask)

    if opt.diff_plot:
        plot_pupil_diffs(pupil_centers, screen_coords)

    with timer("Model training"):
        if opt.model == 'regressor':
            run_regressor(pupil_centers, screen_coords, valid_mask, gaze_config, opt)
        elif opt.model == 'lstm':
            run_lstm(ellipses, screen_coords, valid_mask, gaze_config, opt)
