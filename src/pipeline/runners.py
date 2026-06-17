import numpy as np

from data.visualization import plot_gaze_predictions
from models.polynomial import GazeEstimator
from models.lstm import LSTMGazeEstimator, build_lstm_sequences, build_lstm_sequences_combined
from config import GazeConfig, LSTMConfig
from processing.normalization import compute_pupil_stats, normalize_pupils


def gaze_clip_bounds(gaze_config, normalized):
    """Per-axis (lo, hi) bounds for valid gaze labels, in the same units as the labels.

    Column 0 = vertical (row), column 1 = horizontal (col). ev_eye labels are normalized
    to [0,1]; ebveye labels are screen pixels. Used to clip regressor predictions so
    polynomial extrapolation can't produce off-screen outliers.
    """
    lo = np.array([0.0, 0.0])
    if normalized:
        hi = np.array([1.0, 1.0])
    else:
        hi = np.array([float(gaze_config.screen_height_px), float(gaze_config.screen_width_px)])
    return lo, hi


def errors_to_degrees(err_v, err_h, gaze_config, normalized):
    """Convert per-axis label-unit errors to degrees of visual angle.

    Label/error column 0 = vertical (row, height/screen_fov_y_deg), column 1 =
    horizontal (col, width/screen_fov_x_deg). Linear FoV approximation, matching
    fov_filter_mask. normalized=True for ev_eye ([0,1] labels), False for ebveye (px).
    """
    if normalized:
        return (err_v * gaze_config.screen_fov_y_deg,
                err_h * gaze_config.screen_fov_x_deg)
    return (err_v / (gaze_config.screen_height_px / gaze_config.screen_fov_y_deg),
            err_h / (gaze_config.screen_width_px  / gaze_config.screen_fov_x_deg))


def _gaze_unit_dirs(coords, gaze_config, normalized):
    """Map screen points to unit gaze direction vectors via a flat-screen pinhole model.

    coords: (N,2) with col 0 = vertical (row/y), col 1 = horizontal (col/x). The full
    screen subtends the configured FoV, so half-width/distance = tan(fov/2).
    """
    coords = np.asarray(coords, dtype=np.float64)
    v = coords[:, 0]
    h = coords[:, 1]
    if not normalized:
        h = h / gaze_config.screen_width_px
        v = v / gaze_config.screen_height_px
    tx = (h - 0.5) * 2.0 * np.tan(np.deg2rad(gaze_config.screen_fov_x_deg) / 2.0)
    ty = (v - 0.5) * 2.0 * np.tan(np.deg2rad(gaze_config.screen_fov_y_deg) / 2.0)
    dirs = np.stack([tx, ty, np.ones_like(tx)], axis=1)
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def angular_dod(pred, gt, gaze_config, normalized):
    """Difference of Direction: mean & median angular gaze error (degrees) between the
    predicted and ground-truth gaze direction vectors. This is the metric reported by
    EV-Eye/E-Gaze (single combined angle), unlike the per-axis errors_to_degrees.

    pred/gt: (N,2), col 0 = vertical (row/y), col 1 = horizontal (col/x).
    normalized=True for ev_eye ([0,1] labels), False for ebveye (px). Returns (mean, median).
    """
    up = _gaze_unit_dirs(pred, gaze_config, normalized)
    ug = _gaze_unit_dirs(gt,   gaze_config, normalized)
    dots = np.clip(np.sum(up * ug, axis=1), -1.0, 1.0)
    ang = np.degrees(np.arccos(dots))
    return float(np.mean(ang)), float(np.median(ang))


def fov_filter_mask(screen_coords, fov_width_deg, fov_height_deg, gaze_config, center=None):
    px_per_deg_x = gaze_config.screen_width_px / gaze_config.screen_fov_x_deg
    px_per_deg_y = gaze_config.screen_height_px / gaze_config.screen_fov_y_deg
    half_w_px = (fov_width_deg / 2) * px_per_deg_x
    half_h_px = (fov_height_deg / 2) * px_per_deg_y
    if center is None:
        center_row = gaze_config.screen_height_px / 2
        center_col = gaze_config.screen_width_px / 2
    else:
        center_row, center_col = center
    rows = screen_coords[:, 0]
    cols = screen_coords[:, 1]
    return (np.abs(rows - center_row) <= half_h_px) & (np.abs(cols - center_col) <= half_w_px)


def _fov_rect(fov, fov_center, gaze_config):
    """Return (row_min, row_max, col_min, col_max) in pixels for the FoV window, or None."""
    if fov is None:
        return None
    fov_w_deg, fov_h_deg = fov
    px_per_deg_x = gaze_config.screen_width_px / gaze_config.screen_fov_x_deg
    px_per_deg_y = gaze_config.screen_height_px / gaze_config.screen_fov_y_deg
    half_w = (fov_w_deg / 2) * px_per_deg_x
    half_h = (fov_h_deg / 2) * px_per_deg_y
    if fov_center is None:
        cr = gaze_config.screen_height_px / 2
        cc = gaze_config.screen_width_px / 2
    else:
        cr, cc = fov_center
    return (cr - half_h, cr + half_h, cc - half_w, cc + half_w)


def split_by_label(pupil_centers, screen_coords, val_ratio, rng=None):
    """
    For each unique label (screen coordinate), assign val_ratio of its frames to
    validation and the rest to training, sampling randomly within each label group.
    Suitable for ebveye saccadic data where many frames share the exact same label.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    train_idx, val_idx = [], []
    labels = [tuple(r) for r in screen_coords]
    unique_labels = set(labels)

    for label in unique_labels:
        idx = np.where((screen_coords == label).all(axis=1))[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_ratio))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    return (pupil_centers[train_idx], pupil_centers[val_idx],
            screen_coords[train_idx], screen_coords[val_idx])


def split_randomly(pupil_centers, screen_coords, train_ratio, val_ratio, rng=None):
    """
    Shuffle all (pupil_center, screen_coord) pairs together (seed 42), then
    take the first train_ratio as training and the next val_ratio as validation.
    Suitable for ev_eye where gaze labels are continuous floats with no repeated values.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    idx = np.arange(len(pupil_centers))
    rng.shuffle(idx)
    n_train = int(len(idx) * train_ratio)
    n_val   = int(len(idx) * val_ratio)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    return (pupil_centers[train_idx], pupil_centers[val_idx],
            screen_coords[train_idx], screen_coords[val_idx])


def split_by_time_blocks(pupil_centers, screen_coords, timestamps, val_ratio, n_blocks):
    """
    Leakage-free split for ev_eye: partition samples into n_blocks contiguous time
    blocks and assign whole blocks to validation vs training. Evenly-spaced eval blocks
    are chosen so both sets span the full session. Because temporally-adjacent (near
    duplicate) frame/event samples fall in the same block, none are split across
    train/val — unlike split_randomly, which leaks them and inflates accuracy.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64)
    t0, t1 = timestamps.min(), timestamps.max()
    span = max(t1 - t0, 1.0)
    block_id = np.clip(((timestamps - t0) / span * n_blocks).astype(int), 0, n_blocks - 1)

    n_val_blocks = max(1, min(n_blocks - 1, round(n_blocks * val_ratio)))
    val_blocks = set(np.linspace(0, n_blocks - 1, n_val_blocks).round().astype(int).tolist())

    val_sel = np.array([b in val_blocks for b in block_id])
    train_sel = ~val_sel
    return (pupil_centers[train_sel], pupil_centers[val_sel],
            screen_coords[train_sel], screen_coords[val_sel])


def run_regressor(pupil_centers, screen_coords, valid_mask, gaze_config: GazeConfig, opt,
                  event_samples=None, frame_timestamps=None):
    dataset = getattr(opt, 'dataset', 'ebveye')

    pupil_centers = np.round(pupil_centers[valid_mask], 2)
    screen_coords = np.round(screen_coords[valid_mask], 2)
    # Per-sample timestamps (parallel to pupil/screen) for the leakage-free block split.
    timestamps = (np.asarray(frame_timestamps, dtype=np.int64)[valid_mask]
                  if frame_timestamps is not None else None)

    if event_samples:
        # Extract (cx, cy) from each event ellipse — same (col, row) format as frame centers
        ev_centers = np.array([[s['ellipse'][0][0], s['ellipse'][0][1]]
                               for s in event_samples], dtype=np.float32)
        ev_labels  = np.array([s['screen_coord'] for s in event_samples], dtype=np.float32)
        ev_ts      = np.array([s['timestamp'] for s in event_samples], dtype=np.int64)
        # Drop zero-coord labels (transition/invalid frames) and, when events were labeled
        # from Tobii (ev_eye), badly-aligned events (nearest Tobii sample too far in time).
        valid_ev = ~np.all(ev_labels == 0, axis=1)
        if event_samples[0].get('gap_us') is not None:
            ev_gaps = np.array([s.get('gap_us', 0) for s in event_samples])
            valid_ev &= ev_gaps <= gaze_config.max_alignment_gap_us
        ev_centers = ev_centers[valid_ev]
        ev_labels  = ev_labels[valid_ev]
        pupil_centers = np.vstack([pupil_centers, ev_centers])
        screen_coords = np.vstack([screen_coords, ev_labels])
        if timestamps is not None:
            timestamps = np.concatenate([timestamps, ev_ts[valid_ev]])
        print(f"Added {valid_ev.sum()} event ellipses → total samples: {len(pupil_centers)}")

    if opt.fov is not None:
        fov_w, fov_h = opt.fov
        fov_mask = fov_filter_mask(screen_coords, fov_w, fov_h, gaze_config, center=opt.fov_center)
        pupil_centers = pupil_centers[fov_mask]
        screen_coords = screen_coords[fov_mask]
        if timestamps is not None:
            timestamps = timestamps[fov_mask]

    # Z-score pupil coordinates by this subject's own stats, mirroring the per-subject
    # normalization the cross-subject path applies (cross_subject_regressor.run_fold);
    # single-subject is the N=1 case. Keeps both regressor paths on one normalization.
    mean, std = compute_pupil_stats(pupil_centers)
    pupil_centers = normalize_pupils(pupil_centers, mean, std)

    if dataset == 'ev_eye' and timestamps is not None:
        pupil_train, pupil_val, screen_train, screen_val = split_by_time_blocks(
            pupil_centers, screen_coords, timestamps,
            val_ratio=gaze_config.val_ratio, n_blocks=gaze_config.n_time_blocks,
        )
    elif dataset == 'ev_eye':
        pupil_train, pupil_val, screen_train, screen_val = split_randomly(
            pupil_centers, screen_coords,
            train_ratio=gaze_config.train_ratio, val_ratio=gaze_config.val_ratio,
        )
    else:
        pupil_train, pupil_val, screen_train, screen_val = split_by_label(
            pupil_centers, screen_coords, val_ratio=gaze_config.val_ratio,
        )

    print(f"Training set size: {len(pupil_train)}")
    print(f"Validation set size: {len(pupil_val)}")

    normalized = dataset == 'ev_eye'
    clip_bounds = gaze_clip_bounds(gaze_config, normalized)
    for deg in gaze_config.poly_degrees:
        print(f"\n--- Degree {deg} ---")
        gaze_estimator = GazeEstimator(degree=deg, clip_bounds=clip_bounds)
        gaze_estimator.fit(pupil_train, screen_train)

        eval_pupil, eval_screen = pupil_val, screen_val

        val_metrics = gaze_estimator.evaluate(eval_pupil, eval_screen)
        dod_mean, dod_med = angular_dod(gaze_estimator.predict(eval_pupil), eval_screen,
                                        gaze_config, normalized)
        print(f"Validation MSE:        {val_metrics['mse']:.5f} pixels²")
        print(f"Validation RMSE: {val_metrics['rmse']:.5f} pixels")
        print(f"Validation Mean Error: {val_metrics['mean_error']:.5f} pixels")
        print(f"DoD: mean={dod_mean:.2f}°  median={dod_med:.2f}°")

        if opt.ge_plots:
            val_pred = gaze_estimator.predict(eval_pupil)
            plot_gaze_predictions(val_pred, eval_screen, title=f'Degree {deg} — validation set',
                                  fov_rect=_fov_rect(opt.fov, opt.fov_center, gaze_config))


def run_regressor_events_eval(pupil_centers, screen_coords, valid_mask, gaze_config, opt,
                              event_samples=None, frame_timestamps=None):
    """Calibrate the polynomial on frame pupil centers, then evaluate gaze on the
    high-frequency event-tracked centers — EV-Eye's high-frequency gaze protocol.

    The gaze model is unchanged: one polynomial maps a pupil center to a screen point.
    The only difference from run_regressor is *what feeds it*. We fit on frame centers
    (clean, 25 Hz) and then score the fitted polynomial on the event-derived centers
    (noisy, high-frequency) — which is how events contribute to gaze: they supply extra
    pupil centers between frames, not a separate model.

    A leakage-free temporal block split keeps calibration and evaluation disjoint in time:
    time blocks are assigned to calibration vs. evaluation; frames in calibration blocks fit
    the polynomial, while both events and frames in the evaluation blocks are scored with it.
    Reporting the frame-eval and event-eval DoD side by side isolates the accuracy cost of
    using the event stream vs. the frame stream under one shared calibration.
    """
    dataset = getattr(opt, 'dataset', 'ebveye')
    normalized = dataset == 'ev_eye'

    if not event_samples:
        print("events_eval: no event samples available — nothing to evaluate "
              "(did you also pass --frame_only?).")
        return

    # Frame samples — the calibration source.
    frame_pupils = np.round(pupil_centers[valid_mask], 2)
    frame_screens = np.round(screen_coords[valid_mask], 2)
    frame_ts = np.asarray(frame_timestamps, dtype=np.int64)[valid_mask]

    # Event samples — the high-frequency evaluation source. Drop zero-coord (transition)
    # labels and, for ev_eye, events whose nearest Tobii sample is too far in time.
    ev_centers = np.array([[s['ellipse'][0][0], s['ellipse'][0][1]] for s in event_samples],
                          dtype=np.float64)
    ev_labels = np.array([s['screen_coord'] for s in event_samples], dtype=np.float64)
    ev_ts = np.array([s['timestamp'] for s in event_samples], dtype=np.int64)
    valid_ev = ~np.all(ev_labels == 0, axis=1)
    if event_samples[0].get('gap_us') is not None:
        ev_gaps = np.array([s.get('gap_us', 0) for s in event_samples])
        valid_ev &= ev_gaps <= gaze_config.max_alignment_gap_us
    ev_centers, ev_labels, ev_ts = ev_centers[valid_ev], ev_labels[valid_ev], ev_ts[valid_ev]

    if opt.fov is not None:
        fov_w, fov_h = opt.fov
        fm = fov_filter_mask(frame_screens, fov_w, fov_h, gaze_config, center=opt.fov_center)
        frame_pupils, frame_screens, frame_ts = frame_pupils[fm], frame_screens[fm], frame_ts[fm]
        em = fov_filter_mask(ev_labels, fov_w, fov_h, gaze_config, center=opt.fov_center)
        ev_centers, ev_labels, ev_ts = ev_centers[em], ev_labels[em], ev_ts[em]

    # Leakage-free temporal block split, shared between the two streams so calibration
    # frames and evaluation events never fall in the same block (hence never overlap in time).
    n_blocks = gaze_config.n_time_blocks
    t0 = int(min(frame_ts.min(), ev_ts.min()))
    t1 = int(max(frame_ts.max(), ev_ts.max()))
    span = max(t1 - t0, 1)

    def block_ids(ts):
        return np.clip(((ts - t0) / span * n_blocks).astype(int), 0, n_blocks - 1)

    n_val_blocks = max(1, min(n_blocks - 1, round(n_blocks * gaze_config.val_ratio)))
    val_blocks = np.linspace(0, n_blocks - 1, n_val_blocks).round().astype(int)

    frame_train = ~np.isin(block_ids(frame_ts), val_blocks)
    frame_eval = np.isin(block_ids(frame_ts), val_blocks)
    ev_eval = np.isin(block_ids(ev_ts), val_blocks)

    if frame_train.sum() == 0 or ev_eval.sum() == 0:
        print("events_eval: empty calibration or event-eval split — aborting.")
        return

    # Normalize every stream with the calibration-frame stats (one shared input space).
    mean, std = compute_pupil_stats(frame_pupils[frame_train])
    ftrain = normalize_pupils(frame_pupils[frame_train], mean, std)
    feval = normalize_pupils(frame_pupils[frame_eval], mean, std)
    eveval = normalize_pupils(ev_centers[ev_eval], mean, std)
    ftrain_y, feval_y, eveval_y = (frame_screens[frame_train], frame_screens[frame_eval],
                                   ev_labels[ev_eval])

    # Stale-frame baseline: predict each eval event from the most-recent frame's pupil
    # center (i.e. *no* event tracking — hold the last frame center). Comparing this to the
    # event-tracked DoD answers whether the ICP tracking actually improves over doing nothing:
    #   stale ≈ tracked  → tracking adds little; the error is intrinsic to scoring at event
    #                       timestamps against 100 Hz Tobii labels (and saccade oversampling).
    #   stale ≫ tracked  → tracking is following the pupil; the residual is the tracker's
    #                       accuracy ceiling and is what's worth improving.
    fts_sorted_idx = np.argsort(frame_ts)
    fts_sorted = frame_ts[fts_sorted_idx]
    fp_sorted = frame_pupils[fts_sorted_idx]
    fs_sorted = frame_screens[fts_sorted_idx]
    ev_eval_ts = ev_ts[ev_eval]
    stale_idx = np.clip(np.searchsorted(fts_sorted, ev_eval_ts, side='right') - 1,
                        0, len(fts_sorted) - 1)
    stale_eval = normalize_pupils(fp_sorted[stale_idx], mean, std)

    # Pure label-vs-label diagnostic (no model): angular distance between each event's Tobii
    # label and its nearest frame's Tobii label, plus the timestamp gap to that frame. If this
    # is large for temporally-close pairs, the event labels are mis-assigned (not a tracker issue).
    lbl_mean, lbl_med = angular_dod(fs_sorted[stale_idx], eveval_y, gaze_config, normalized)
    stale_gap_us = np.abs(ev_eval_ts - fts_sorted[stale_idx])
    print(f"  [diag] event-label vs nearest-frame-label: mean={lbl_mean:.2f}°  median={lbl_med:.2f}°  "
          f"| frame-gap median={np.median(stale_gap_us)/1000:.1f} ms  95th={np.percentile(stale_gap_us,95)/1000:.1f} ms")
    print(f"  [diag] unique stale frames: {len(np.unique(stale_idx))} serving {len(stale_idx)} events")

    # Effective event update rate over the eval blocks (instantaneous, from median spacing).
    rate_str = ""
    ev_ts_eval = np.sort(ev_eval_ts)
    if len(ev_ts_eval) > 1:
        med_dt_us = float(np.median(np.diff(ev_ts_eval)))
        if med_dt_us > 0:
            rate_str = f"  (median spacing {med_dt_us / 1000:.2f} ms ≈ {1e6 / med_dt_us:.0f} Hz)"
    print(f"Calibration frames: {len(ftrain)}  |  eval frames: {len(feval)}  |  "
          f"eval events: {len(eveval)}{rate_str}")

    # Fixation vs saccade split: angular travel of the Tobii label over a ±1-frame window.
    # During a fixation the eye is still, so the tracker should match a fresh frame detection;
    # if event-tracked DoD on stable samples ≈ frame DoD, the tracker is accurate and the bulk
    # error is saccade/label-limited (a 100 Hz-reference limitation, not a fixable tracker bug).
    win_us = 40000
    o = np.argsort(ev_eval_ts)
    ts_s, lab_s = ev_eval_ts[o], eveval_y[o]
    j_a = np.clip(np.searchsorted(ts_s, ts_s + win_us), 0, len(ts_s) - 1)
    j_b = np.clip(np.searchsorted(ts_s, ts_s - win_us), 0, len(ts_s) - 1)
    d_a = _gaze_unit_dirs(lab_s[j_a], gaze_config, normalized)
    d_b = _gaze_unit_dirs(lab_s[j_b], gaze_config, normalized)
    travel = np.degrees(np.arccos(np.clip(np.sum(d_a * d_b, axis=1), -1.0, 1.0)))
    stable = np.empty(len(eveval), dtype=bool)
    stable[o] = travel < 3.0   # <3° gaze travel over ~80 ms → fixation / slow pursuit
    print(f"  eval events: {stable.sum()} stable (fixation) | {(~stable).sum()} moving (saccade)")

    clip_bounds = gaze_clip_bounds(gaze_config, normalized)
    for deg in gaze_config.poly_degrees:
        print(f"\n--- Degree {deg} ---")
        estimator = GazeEstimator(degree=deg, clip_bounds=clip_bounds)
        estimator.fit(ftrain, ftrain_y)

        ev_pred = estimator.predict(eveval)
        f_mean, f_med = angular_dod(estimator.predict(feval), feval_y, gaze_config, normalized)
        e_mean, e_med = angular_dod(ev_pred, eveval_y, gaze_config, normalized)
        s_mean, s_med = angular_dod(estimator.predict(stale_eval), eveval_y, gaze_config, normalized)
        st_mean, st_med = angular_dod(ev_pred[stable], eveval_y[stable], gaze_config, normalized)
        mv_mean, mv_med = angular_dod(ev_pred[~stable], eveval_y[~stable], gaze_config, normalized)
        stale_pred = estimator.predict(stale_eval)
        ss_mean, ss_med = angular_dod(stale_pred[stable], eveval_y[stable], gaze_config, normalized)
        # Polynomial error on the stale frames vs their OWN frame labels (fixation subset):
        # isolates whether these anchor frames are simply hard for the regressor.
        sl_mean, sl_med = angular_dod(stale_pred[stable], fs_sorted[stale_idx][stable],
                                      gaze_config, normalized)
        print(f"  [diag deg{deg}] stale-frame vs OWN frame-label @ fixation: "
              f"mean={sl_mean:.2f}°  median={sl_med:.2f}°")
        # Per-event anchor residual = poly error of the anchor frame vs its own label.
        ua = _gaze_unit_dirs(stale_pred, gaze_config, normalized)
        ub = _gaze_unit_dirs(fs_sorted[stale_idx], gaze_config, normalized)
        anchor_resid = np.degrees(np.arccos(np.clip(np.sum(ua * ub, axis=1), -1.0, 1.0)))
        good = anchor_resid < 5.0
        if good.sum():
            g_mean, g_med = angular_dod(ev_pred[good], eveval_y[good], gaze_config, normalized)
            print(f"  [diag deg{deg}] events anchored to GOOD frames (<5° anchor resid): "
                  f"mean={g_mean:.2f}°  median={g_med:.2f}°  (n={good.sum()}/{len(good)})")
        print(f"  frames     (25 Hz)    : DoD mean={f_mean:.2f}°  median={f_med:.2f}°  (n={len(feval)})")
        print(f"  events     (hi-freq)  : DoD mean={e_mean:.2f}°  median={e_med:.2f}°  (n={len(eveval)})")
        print(f"  stale-frame (baseline): DoD mean={s_mean:.2f}°  median={s_med:.2f}°  (no event tracking)")
        print(f"  events @ fixation     : DoD mean={st_mean:.2f}°  median={st_med:.2f}°  (n={stable.sum()})")
        print(f"  stale  @ fixation     : DoD mean={ss_mean:.2f}°  median={ss_med:.2f}°  (last frame, eye still)")
        print(f"  events @ saccade      : DoD mean={mv_mean:.2f}°  median={mv_med:.2f}°  (n={(~stable).sum()})")

        if opt.ge_plots:
            plot_gaze_predictions(estimator.predict(eveval), eveval_y,
                                  title=f'Events eval — Degree {deg}',
                                  fov_rect=_fov_rect(opt.fov, opt.fov_center, gaze_config))


def run_lstm(ellipses, screen_coords, valid_mask, gaze_config, opt):
    lstm_config = LSTMConfig()

    X, y = build_lstm_sequences(
        ellipses, screen_coords, valid_mask, seq_len=lstm_config.seq_len
    )
    print(f"Total windows: {len(X)}  (shape {X.shape})")

    if opt.fov is not None:
        fov_w, fov_h = opt.fov
        fov_mask = fov_filter_mask(y, fov_w, fov_h, gaze_config, center=opt.fov_center)
        X, y = X[fov_mask], y[fov_mask]

    n = len(X)
    n_train = int(n * gaze_config.train_ratio)
    n_val   = int(n * gaze_config.val_ratio)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    print(f"Training set: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    lstm_estimator = LSTMGazeEstimator(lstm_config)
    lstm_estimator.fit(X_train, y_train, X_val, y_val)

    eval_X, eval_y = X_val, y_val

    val_metrics = lstm_estimator.evaluate(eval_X, eval_y)
    print(f"Validation MSE:        {val_metrics['mse']:.5f} pixels²")
    print(f"Validation RMSE:       {val_metrics['rmse']:.5f} pixels")
    print(f"Validation Mean Error: {val_metrics['mean_error']:.5f} pixels")

    if opt.ge_plots:
        val_pred = lstm_estimator.predict(eval_X)
        plot_gaze_predictions(val_pred, eval_y, title='LSTM — validation set',
                              fov_rect=_fov_rect(opt.fov, opt.fov_center, gaze_config))


def run_lstm_combined(combined_samples, gaze_config, opt):
    lstm_config = LSTMConfig()

    X, y = build_lstm_sequences_combined(combined_samples, seq_len=lstm_config.seq_len)
    print(f"Total windows (frame+event): {len(X)}  (shape {X.shape})")

    if opt.fov is not None:
        fov_w, fov_h = opt.fov
        fov_mask = fov_filter_mask(y, fov_w, fov_h, gaze_config, center=opt.fov_center)
        X, y = X[fov_mask], y[fov_mask]

    n = len(X)
    n_train = int(n * gaze_config.train_ratio)
    n_val   = int(n * gaze_config.val_ratio)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    print(f"Training set: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    lstm_estimator = LSTMGazeEstimator(lstm_config)
    lstm_estimator.fit(X_train, y_train, X_val, y_val)

    val_metrics = lstm_estimator.evaluate(X_val, y_val)
    print(f"Validation MSE:        {val_metrics['mse']:.5f} pixels²")
    print(f"Validation RMSE:       {val_metrics['rmse']:.5f} pixels")
    print(f"Validation Mean Error: {val_metrics['mean_error']:.5f} pixels")

    if opt.ge_plots:
        val_pred = lstm_estimator.predict(X_val)
        plot_gaze_predictions(val_pred, y_val, title='LSTM (frame+event) — validation set',
                              fov_rect=_fov_rect(opt.fov, opt.fov_center, gaze_config))