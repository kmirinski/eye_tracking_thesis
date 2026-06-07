import numpy as np

from data.visualization import plot_gaze_predictions
from models.polynomial import GazeEstimator
from models.lstm import LSTMGazeEstimator, build_lstm_sequences, build_lstm_sequences_combined
from config import GazeConfig, LSTMConfig
from processing.normalization import compute_pupil_stats, normalize_pupils


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
    for deg in gaze_config.poly_degrees:
        print(f"\n--- Degree {deg} ---")
        gaze_estimator = GazeEstimator(degree=deg)
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