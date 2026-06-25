import numpy as np

from data.visualization import write_ellipse_video, browse_ellipse_frames, browse_pupil_extraction, plot_pupil_centers_over_time_all, plot_pupil_centers_over_time, plot_pupil_diffs
from utils import timer
from data.loaders import EyeDataset, EvEyeDataset
from processing.frame_detection import extract_pupil_centers
from config import FrameDetectionConfig, get_frame_detection_config, GazeConfig, get_gaze_config, TemplateTrackingConfig
from pipeline.runners import (run_regressor, run_regressor_events_eval, run_lstm,
                              run_lstm_combined)
from tracking import sample_ellipse_boundary, points_to_edge_matching


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
        - Phase A (pre-saccade):  while the eye is stable (dist to last_valid < threshold), relabel frames to old label
        - Phase B (saccade):      while the eye is moving (dist >= threshold), mark frames to discard
        - Phase C (settling):     discard until stability_window consecutive frames are all below threshold
        - Phase D (post-saccade): remaining frames keep the new label

    Blink frames (pupil == -1) in Phase A are relabeled (filtered anyway by basic mask).
    Blink frames in Phase B/C are discarded. Diffs are always vs. last valid pupil to avoid
    false saccade triggers from blinks.

    Returns (screen_coords_relabled, saccade_discard_mask, phase_labels) in storage order.
    '''
    n = len(screen_coords)
    pupil_chron = pupil_centers[::-1]
    screen_chron = screen_coords[::-1].copy()
    saccade_mask_chron = np.zeros(n, dtype=bool)
    # Per-frame phase (chronological), aligned with the thesis terminology:
    #   'A' pre-saccade (relabeled to previous target)
    #   'B' saccade     (active flight, discarded)
    #   'C' settling    (stability window, discarded)
    #   'D' post-saccade(kept with the current/new label)
    phase_chron = np.full(n, 'none', dtype=object)

    sections = _find_sections(screen_chron)
    if not sections:
        return screen_coords.copy(), saccade_mask_chron[::-1], phase_chron[::-1]

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
                phase_chron[m] = 'A'
                m += 1
                continue
            dist = np.linalg.norm(pupil_chron[m] - last_valid)
            if dist < threshold:
                screen_chron[m] = old_label
                phase_chron[m] = 'A'
                last_valid = pupil_chron[m].copy()
                m += 1
            else:
                break   # saccade detected

        # Phase B: saccade - discard frames where eye is actively moving
        while m < sac_end:
            if np.all(pupil_chron[m] == -1):
                saccade_mask_chron[m] = True    # blink during saccade: discard
                phase_chron[m] = 'B'
                m += 1
                continue
            dist = np.linalg.norm(pupil_chron[m] - last_valid)
            if dist >= threshold:
                saccade_mask_chron[m] = True
                phase_chron[m] = 'B'
                last_valid = pupil_chron[m].copy()
                m += 1
            else:
                break   # large movement ended, settling begins

        # Settling: stability window — discard until stability_window consecutive frames are below threshold
        stable_count = 0
        while m < sac_end and stable_count < stability_window:
            if np.all(pupil_chron[m] == -1):
                saccade_mask_chron[m] = True   # blink resets window
                phase_chron[m] = 'C'
                stable_count = 0
                m += 1
                continue
            dist = np.linalg.norm(pupil_chron[m] - last_valid)
            if dist >= threshold:
                saccade_mask_chron[m] = True   # spike: discard and reset window
                phase_chron[m] = 'C'
                last_valid = pupil_chron[m].copy()
                stable_count = 0
                m += 1
            else:
                saccade_mask_chron[m] = True   # within stability window: discard
                phase_chron[m] = 'C'
                last_valid = pupil_chron[m].copy()
                stable_count += 1
                m += 1
        # Post-saccade (kept) frames begin at m (eye settled)

    # Remaining saccade-section frames are kept on the current target: post-saccade
    # fixation (and the initial fixation before the first target change).
    for i in range(sac_start, sac_end):
        if phase_chron[i] == 'none':
            phase_chron[i] = 'D'

    return screen_chron[::-1], saccade_mask_chron[::-1], phase_chron[::-1]


def build_valid_mask(blink_mask, screen_coords, skip_frames,
                     saccade_mask=None, skip_label_changes=True, post_blink_skip_frames=1,
                     alignment_gaps=None, max_alignment_gap_us=None):
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

    # ev_eye: drop frames whose nearest Tobii sample is too far in time (bad label).
    alignment_removed = 0
    if alignment_gaps is not None and max_alignment_gap_us is not None:
        gap_ok = alignment_gaps[::-1] <= max_alignment_gap_us
        alignment_removed = int(np.sum(valid & ~gap_ok))
        valid &= gap_ok

    basic_removed = np.sum(blink_mask_chron | post_blink_chron | np.all(screen_chron == 0, axis=1))
    removed_total = n - np.sum(valid)
    print(f"Frames removed (basic filter): {basic_removed}")
    print(f"Frames removed (temporal skip): {removed_total - basic_removed - alignment_removed}")
    print(f"Frames removed (alignment gap): {alignment_removed}")
    print(f"Frames removed (total): {removed_total} / {n}")

    # Return mask in the original (reversed) array order
    return valid[::-1]



def template_tracking_stage(events_np, frame_list, ellipses,
                            screen_coords, valid_mask, config: TemplateTrackingConfig):
    frame_list_chron = frame_list[::-1]
    ellipses_chron = ellipses[::-1]
    screen_chron = screen_coords[::-1]
    valid_chron = valid_mask[::-1]
    frame_ts = np.array([f.timestamp for f in frame_list_chron], dtype=np.int64)

    first_valid_idx = None
    for i in range(len(valid_chron)):
        if valid_chron[i] and ellipses_chron[i] is not None:
            first_valid_idx = i
            break
    if first_valid_idx is None:
        print("Template tracking: no valid frame with ellipse found, skipping.")
        return []

    first_valid_ts = frame_ts[first_valid_idx]
    template_ellipse = ellipses_chron[first_valid_idx]
    boundary_Q = sample_ellipse_boundary(template_ellipse, config.num_boundary)
    center = np.array(template_ellipse[0], dtype=np.float64)
    axes = template_ellipse[1]
    angle = template_ellipse[2]
    gamma_bar = np.mean(np.linalg.norm(boundary_Q - center, axis=1))

    start_ev = int(np.searchsorted(events_np[:, 3], first_valid_ts, side='right'))

    event_samples = []
    candidate_buf = []
    current_frame_ptr = first_valid_idx

    # Option B filtering state: anchor = last frame-detected center; in_blink = current
    # frame has no detection (eyelid closed / detection failed).
    anchor_center = center.copy()
    in_blink = False
    rej_blink = rej_residual = rej_drift = 0

    for i in range(start_ev, len(events_np)):
        polarity, row, col, ts = events_np[i]

        while (current_frame_ptr + 1 < len(frame_list_chron) and
               frame_ts[current_frame_ptr + 1] <= ts):
            current_frame_ptr += 1
            candidate_ell = ellipses_chron[current_frame_ptr]
            if candidate_ell is not None:
                template_ellipse = candidate_ell
                boundary_Q = sample_ellipse_boundary(template_ellipse, config.num_boundary)
                center = np.array(template_ellipse[0], dtype=np.float64)
                axes = template_ellipse[1]
                angle = template_ellipse[2]
                gamma_bar = np.mean(np.linalg.norm(boundary_Q - center, axis=1))
                anchor_center = center.copy()
                in_blink = False
                candidate_buf = []
            else:
                in_blink = True   # failed detection / blink: template is stale

        # Blink exclusion: don't track through a blink until the next valid frame re-anchors.
        if config.enable_filter and in_blink:
            candidate_buf = []
            continue

        dist = np.sqrt((col - center[0]) ** 2 + (row - center[1]) ** 2)
        if config.lambda1 * gamma_bar < dist < config.lambda2 * gamma_bar:
            candidate_buf.append((col, row, ts))

        if len(candidate_buf) < config.num_events:
            continue

        pts = np.array([(c[0], c[1]) for c in candidate_buf], dtype=np.float64)
        T, residual = points_to_edge_matching(pts, boundary_Q,
                                              max_iter=config.max_icp_iter,
                                              convergence=config.convergence)

        if config.enable_filter:
            # Residual gate (E-Gaze): candidate events must lie on the pupil ring.
            if residual > config.max_residual_ratio * gamma_bar:
                rej_residual += 1
                candidate_buf = []
                continue
            # Drift bound (Angelopoulos): center can't wander far from the frame anchor.
            new_center = center - T
            if np.linalg.norm(new_center - anchor_center) > config.max_drift_ratio * gamma_bar:
                rej_drift += 1
                candidate_buf = []
                continue
            center = new_center
        else:
            center = center - T
        boundary_Q = boundary_Q - T

        t_last = candidate_buf[-1][2]
        nxt = int(np.searchsorted(frame_ts, t_last, side='right'))
        if nxt < len(frame_list_chron):
            new_ellipse = ((float(center[0]), float(center[1])), axes, angle)
            event_samples.append({
                'ellipse':      new_ellipse,
                'screen_coord': screen_chron[nxt],
                'timestamp':    int(t_last),
            })

        candidate_buf = []

    if config.enable_filter:
        print(f"Template tracking: {len(event_samples)} event samples extracted "
              f"(rejected — residual: {rej_residual}, drift: {rej_drift}; blink batches skipped).")
    else:
        print(f"Template tracking: {len(event_samples)} event samples extracted (filtering off).")
    return event_samples


def label_events_from_tobii(event_samples, gaze_records):
    """
    Assign each event sample the gaze label of its nearest Tobii sample in time, and
    record the time gap. Replaces the coarse next-frame label set in template_tracking_stage.

    gaze_records: (M, 3) array of [davis_us, x_norm(col), y_norm(row)] (from EvEyeDataset).
    Mirrors the frame-to-Tobii nearest-neighbor matching in loaders.py. Mutates and returns
    event_samples; each gains 'gap_us' and an updated 'screen_coord' = [row(y), col(x)].
    """
    if not event_samples or gaze_records is None or len(gaze_records) == 0:
        return event_samples

    ev_ts   = np.array([s['timestamp'] for s in event_samples], dtype=np.int64)
    gaze_ts = gaze_records[:, 0].astype(np.int64)

    idx      = np.clip(np.searchsorted(gaze_ts, ev_ts), 0, len(gaze_records) - 1)
    prev_idx = np.maximum(idx - 1, 0)
    best     = np.where(np.abs(gaze_ts[prev_idx] - ev_ts) <
                        np.abs(gaze_ts[idx]      - ev_ts), prev_idx, idx)
    gaps     = np.abs(gaze_ts[best] - ev_ts)

    for s, gi, gap in zip(event_samples, best, gaps):
        col = gaze_records[gi, 1]   # x_norm
        row = gaze_records[gi, 2]   # y_norm
        s['screen_coord'] = np.array([row, col], dtype=np.float64)
        s['gap_us'] = int(gap)

    return event_samples


def merge_frame_event_samples(ellipses, screen_coords, valid_mask, frame_list, event_samples):
    """
    Combine valid frame samples and event samples into a single chronologically sorted list.
    Each entry: {'ellipse': ..., 'screen_coord': ..., 'timestamp': ...}
    """
    frame_list_chron = frame_list[::-1]
    ellipses_chron   = ellipses[::-1]
    screen_chron     = screen_coords[::-1]
    valid_chron      = valid_mask[::-1]

    frame_samples = [
        {'ellipse': e, 'screen_coord': sc, 'timestamp': f.timestamp, 'source': 'frame'}
        for f, e, sc, v in zip(frame_list_chron, ellipses_chron, screen_chron, valid_chron)
        if v and e is not None
    ]
    for s in event_samples:
        s.setdefault('source', 'event')

    combined = sorted(frame_samples + event_samples, key=lambda x: x['timestamp'])
    print(f"Combined samples: {len(frame_samples)} frame + {len(event_samples)} event = {len(combined)} total.")
    return combined


def pupil_extraction_stage(eye_dataset: EyeDataset, frame_config: FrameDetectionConfig):
    pupil_centers, ellipses = extract_pupil_centers(eye_dataset.frame_list, config=frame_config)
    screen_coords = np.array([(frame.row, frame.col) for frame in eye_dataset.frame_list], dtype=np.float64)
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
    dataset = getattr(opt, 'dataset', 'ebveye')
    motion = getattr(opt, 'motion',  'saccadic')
    frame_config = get_frame_detection_config(opt.subject, opt.eye, dataset)
    gaze_config = get_gaze_config(opt.subject)

    print(f'Collecting data of the {opt.eye} eye of subject {opt.subject}')
    print('Loading data from ' + opt.data_dir)

    if dataset == 'ev_eye':
        eye_dataset = EvEyeDataset(
            opt.data_dir, opt.subject, motion=motion, mode='np',
        )
        eye_key = opt.eye  # 'left' or 'right'
        with timer("Collection"):
            eye_dataset.collect_data(eye=eye_key)
    else:
        eye_dataset = EyeDataset(opt.data_dir, opt.subject, mode='stack')
        eye_key = 0 if opt.eye == 'left' else 1
        with timer("Collection"):
            eye_dataset.collect_data(eye=eye_key, motion=motion)

    with timer("Pupil extraction"):
        pupil_centers, ellipses, screen_coords = pupil_extraction_stage(eye_dataset, frame_config)

    with timer("Noise flagging"):
        blink_mask = noise_flagging_stage(pupil_centers)

    saccade_mask = None
    screen_coords_original = screen_coords.copy()
    phase_chron_storage = None
    if opt.relabel and motion == 'saccadic' and dataset != 'ev_eye':
        with timer("Relabeling"):
            screen_coords, saccade_mask, phase_chron_storage = relabeling_stage(
                pupil_centers, screen_coords, gaze_config)

    # skip_label_changes only makes sense for ebveye, where target jumps between
    # discrete fixation points. For ev_eye, Tobii labels are continuous floats —
    # every frame looks like a "change", which would invalidate everything.
    skip_label_changes = (dataset == 'ebveye') and (motion == 'saccadic') and not opt.relabel
    valid_mask = build_valid_mask(
        blink_mask, screen_coords,
        skip_frames=gaze_config.saccade_skip_frames,
        saccade_mask=saccade_mask,
        skip_label_changes=skip_label_changes,
        post_blink_skip_frames=gaze_config.post_blink_skip_frames,
        alignment_gaps=getattr(eye_dataset, 'alignment_gaps', None),
        max_alignment_gap_us=gaze_config.max_alignment_gap_us,
    )

    if opt.relabel_diag and opt.relabel:
        from data.visualization import plot_relabeling_diagnostic
        plot_relabeling_diagnostic(
            pupil_centers[::-1], screen_coords_original[::-1],
            phase_chron_storage[::-1], blink_mask[::-1],
            threshold=gaze_config.relabel_diff_threshold,
        )

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

    # Event ellipses are only needed for the LSTM, the event diagnostics, or a regressor
    # run that includes events. Skip the (slow) extraction for a frame-only regressor.
    skip_events = (getattr(opt, 'frame_only', False) and opt.model == 'regressor'
                   and not getattr(opt, 'event_diag', False)
                   and not getattr(opt, 'events_eval', False))
    if skip_events:
        event_samples = []
        print("Frame-only mode: skipping event extraction.")
    else:
        events_np = eye_dataset.load_events_sorted(eye_key)
        with timer("Event extraction"):
            tt_config = TemplateTrackingConfig()
            tt_config.enable_filter = not getattr(opt, 'no_event_filter', False)
            event_samples = template_tracking_stage(
                events_np, eye_dataset.frame_list,
                ellipses, screen_coords, valid_mask, tt_config,
            )
            # ev_eye: label each event by its nearest Tobii sample in time (instead of the
            # coarse next-frame label) and attach the per-event alignment gap.
            if getattr(eye_dataset, 'gaze_records', None) is not None:
                event_samples = label_events_from_tobii(event_samples, eye_dataset.gaze_records)

    if getattr(opt, 'event_diag', False):
        from data.visualization import (plot_event_ellipse_diagnostic,
                                        plot_combined_pupil_trajectory,
                                        plot_event_frame_deviation_hist)
        plot_event_ellipse_diagnostic(eye_dataset.frame_list, ellipses, event_samples)
        plot_event_frame_deviation_hist(eye_dataset.frame_list, ellipses, event_samples)
        combined_diag = merge_frame_event_samples(
            ellipses, screen_coords, valid_mask, eye_dataset.frame_list, event_samples,
        )
        plot_combined_pupil_trajectory(combined_diag)

    with timer("Model training"):
        if opt.model == 'regressor':
            frame_timestamps = np.array([f.timestamp for f in eye_dataset.frame_list],
                                        dtype=np.int64)
            if getattr(opt, 'events_eval', False):
                run_regressor_events_eval(pupil_centers, screen_coords, valid_mask, gaze_config,
                                          opt, event_samples=event_samples,
                                          frame_timestamps=frame_timestamps)
            else:
                reg_event_samples = None if getattr(opt, 'frame_only', False) else event_samples
                run_regressor(pupil_centers, screen_coords, valid_mask, gaze_config, opt,
                              event_samples=reg_event_samples, frame_timestamps=frame_timestamps)
        elif opt.model == 'lstm':
            combined = merge_frame_event_samples(
                ellipses, screen_coords, valid_mask, eye_dataset.frame_list, event_samples,
            )
            run_lstm_combined(combined, gaze_config, opt)
