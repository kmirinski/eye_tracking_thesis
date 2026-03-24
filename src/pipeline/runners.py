import numpy as np

from data.visualization import plot_gaze_predictions
from models.polynomial import GazeEstimator
from models.lstm import LSTMGazeEstimator, build_lstm_sequences
from config import GazeConfig, LSTMConfig


def fov_filter_mask(screen_coords, fov_width_deg, fov_height_deg, gaze_config):
    px_per_deg_x = gaze_config.screen_width_px / gaze_config.screen_fov_x_deg
    px_per_deg_y = gaze_config.screen_height_px / gaze_config.screen_fov_y_deg
    half_w_px = (fov_width_deg / 2) * px_per_deg_x
    half_h_px = (fov_height_deg / 2) * px_per_deg_y
    center_row = gaze_config.screen_height_px / 2
    center_col = gaze_config.screen_width_px / 2
    rows = screen_coords[:, 0]
    cols = screen_coords[:, 1]
    return (np.abs(rows - center_row) <= half_h_px) & (np.abs(cols - center_col) <= half_w_px)


def _fov_rect(opt, gaze_config):
    """Return (row_min, row_max, col_min, col_max) in pixels for the FoV window, or None."""
    if opt.fov is None:
        return None
    fov_w_deg, fov_h_deg = opt.fov
    px_per_deg_x = gaze_config.screen_width_px / gaze_config.screen_fov_x_deg
    px_per_deg_y = gaze_config.screen_height_px / gaze_config.screen_fov_y_deg
    half_w = (fov_w_deg / 2) * px_per_deg_x
    half_h = (fov_h_deg / 2) * px_per_deg_y
    cr = gaze_config.screen_height_px / 2
    cc = gaze_config.screen_width_px / 2
    return (cr - half_h, cr + half_h, cc - half_w, cc + half_w)


def split_by_label(pupil_centers, screen_coords, val_ratio, rng=None):
    """
    For each unique label (screen coordinate), assign 80% of its frames to
    training and 20% to validation, sampling randomly within each label group.
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


def run_regressor(pupil_centers, screen_coords, valid_mask, gaze_config: GazeConfig, opt):

    pupil_centers = np.round(pupil_centers[valid_mask], 2)
    screen_coords = np.round(screen_coords[valid_mask], 2)

    pupil_train, pupil_val, screen_train, screen_val = split_by_label(
        pupil_centers, screen_coords, val_ratio=gaze_config.val_ratio
    )

    print(f"Training set size: {len(pupil_train)}")
    print(f"Validation set size: {len(pupil_val)}")

    for deg in gaze_config.poly_degrees:
        print(f"\n--- Degree {deg} ---")
        gaze_estimator = GazeEstimator(degree=deg)
        gaze_estimator.fit(pupil_train, screen_train)

        if opt.fov is not None:
            fov_w, fov_h = opt.fov
            fov_mask = fov_filter_mask(screen_val, fov_w, fov_h, gaze_config)
            eval_pupil, eval_screen = pupil_val[fov_mask], screen_val[fov_mask]
        else:
            eval_pupil, eval_screen = pupil_val, screen_val

        val_metrics = gaze_estimator.evaluate(eval_pupil, eval_screen)
        print(f"Validation RMSE: {val_metrics['rmse']:.2f} pixels")
        print(f"Validation Mean Error: {val_metrics['mean_error']:.2f} pixels")

        if opt.ge_plots:
            val_pred = gaze_estimator.predict(eval_pupil)
            plot_gaze_predictions(val_pred, eval_screen, title=f'Degree {deg} — validation set',
                                  fov_rect=_fov_rect(opt, gaze_config))


def run_lstm(ellipses, screen_coords, valid_mask, gaze_config, opt):
    lstm_config = LSTMConfig()

    X, y = build_lstm_sequences(
        ellipses, screen_coords, valid_mask, seq_len=lstm_config.seq_len
    )
    print(f"Total windows: {len(X)}  (shape {X.shape})")

    n = len(X)
    n_train = int(n * gaze_config.train_ratio)
    n_val   = int(n * gaze_config.val_ratio)

    X_train, y_train = X[:n_train],               y[:n_train]
    X_val,   y_val   = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test,  y_test  = X[n_train + n_val:],        y[n_train + n_val:]

    print(f"Training set: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    lstm_estimator = LSTMGazeEstimator(lstm_config)
    lstm_estimator.fit(X_train, y_train, X_val, y_val)

    if opt.fov is not None:
        fov_w, fov_h = opt.fov
        fov_mask = fov_filter_mask(y_val, fov_w, fov_h, gaze_config)
        eval_X, eval_y = X_val[fov_mask], y_val[fov_mask]
    else:
        eval_X, eval_y = X_val, y_val

    val_metrics = lstm_estimator.evaluate(eval_X, eval_y)
    print(f"Validation RMSE:       {val_metrics['rmse']:.2f} pixels")
    print(f"Validation Mean Error: {val_metrics['mean_error']:.2f} pixels")

    if opt.ge_plots:
        val_pred = lstm_estimator.predict(eval_X)
        plot_gaze_predictions(val_pred, eval_y, title='LSTM — validation set',
                              fov_rect=_fov_rect(opt, gaze_config))