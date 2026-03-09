import numpy as np
from sklearn.model_selection import train_test_split

from data.visualization import plot_gaze_predictions
from models.polynomial import GazeEstimator
from models.lstm import LSTMGazeEstimator, build_lstm_sequences
from config import GazeConfig, LSTMConfig


def run_regressor(pupil_centers, screen_coords, valid_mask, gaze_config: GazeConfig, opt):

    pupil_centers = np.round(pupil_centers[valid_mask], 2)
    screen_coords = np.round(screen_coords[valid_mask], 2)

    pupil_train, pupil_val, screen_train, screen_val = train_test_split(
        pupil_centers, screen_coords, test_size=gaze_config.val_ratio, random_state=42
    )

    print(f"Training set size: {len(pupil_train)}")
    print(f"Validation set size: {len(pupil_val)}")

    for deg in gaze_config.poly_degrees:
        print(f"\n--- Degree {deg} ---")
        gaze_estimator = GazeEstimator(degree=deg)
        gaze_estimator.fit(pupil_train, screen_train)

        val_metrics = gaze_estimator.evaluate(pupil_val, screen_val)
        print(f"Validation RMSE: {val_metrics['rmse']:.2f} pixels")
        print(f"Validation Mean Error: {val_metrics['mean_error']:.2f} pixels")

        if opt.ge_plots:
            val_pred = gaze_estimator.predict(pupil_val)
            plot_gaze_predictions(val_pred, screen_val, title=f'Degree {deg} — validation set')


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

    val_metrics = lstm_estimator.evaluate(X_val, y_val)
    print(f"Validation RMSE:       {val_metrics['rmse']:.2f} pixels")
    print(f"Validation Mean Error: {val_metrics['mean_error']:.2f} pixels")

    if opt.ge_plots:
        val_pred = lstm_estimator.predict(X_val)
        plot_gaze_predictions(val_pred, y_val, title='LSTM — validation set')