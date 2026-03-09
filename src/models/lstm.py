import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from config import LSTMConfig


def ellipse_to_21d(ellipse) -> np.ndarray:
    """
    Encode a single OpenCV ellipse as a 21-dimensional vector.

    The encoding follows the outer-product method from the paper:
      1. Build [cx, cy, w, h, angle, 1]  (6 elements; the '1' preserves 1st-order terms)
      2. Compute outer product M = v ⊗ vᵀ  (6×6)
      3. Extract lower triangle (diagonal + below) → 21 elements

    Args:
        ellipse: OpenCV ellipse tuple ((cx, cy), (w, h), angle)

    Returns:
        np.ndarray of shape (21,)
    """
    (cx, cy), (w, h), angle = ellipse
    v = np.array([cx, cy, w, h, angle, 1.0], dtype=np.float32)
    M = np.outer(v, v)
    row_idx, col_idx = np.tril_indices(6)
    return M[row_idx, col_idx]


def build_lstm_sequences(ellipses, screen_coords, valid_mask, seq_len=10):
    """
    Build sliding-window sequences from chronologically ordered ellipse data.

    A window is accepted only when ALL seq_len frames in it are valid
    (non-None ellipse AND valid_mask is True), guaranteeing temporal continuity.
    The label for each window is the screen coordinate of its last frame.

    Args:
        ellipses:     list of N ellipse tuples (or None for failed detections),
                      in reversed/stack order as produced by extract_pupil_centers
        screen_coords: np.ndarray (N, 2), same order as ellipses
        valid_mask:   bool np.ndarray (N,), same order as ellipses
        seq_len:      number of frames per window

    Returns:
        X: np.ndarray (M, seq_len, 21)
        y: np.ndarray (M, 2)
    """
    n = len(ellipses)

    # Flip to chronological order (stack order is reversed)
    ellipses_chron = ellipses[::-1]
    screen_chron   = screen_coords[::-1]
    valid_chron    = valid_mask[::-1]

    # Pre-compute 21D features; None where the frame is invalid
    features = []
    for i in range(n):
        if valid_chron[i] and ellipses_chron[i] is not None:
            features.append(ellipse_to_21d(ellipses_chron[i]))
        else:
            features.append(None)

    X, y = [], []
    for i in range(seq_len - 1, n):
        window_slice = range(i - seq_len + 1, i + 1)
        if all(features[j] is not None for j in window_slice):
            X.append(np.stack([features[j] for j in window_slice]))
            y.append(screen_chron[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class LSTMGazeEstimator:

    def __init__(self, config: LSTMConfig = None):
        self.config = config or LSTMConfig()
        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted = False

    def _build_model(self):
        cfg = self.config
        l1 = regularizers.L1(cfg.l1_reg)

        inputs = keras.Input(shape=(cfg.seq_len, 21))
        x = layers.LSTM(
            cfg.lstm_units,
            activation='tanh',
            recurrent_activation='sigmoid',
            kernel_regularizer=l1,
        )(inputs)
        for units in cfg.dense_units:
            x = layers.Dense(
                units,
                activation='relu',
                kernel_initializer='he_uniform',
                kernel_regularizer=l1,
            )(x)
        outputs = layers.Dense(2, kernel_regularizer=l1)(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(cfg.learning_rate),
            loss='mse',
        )
        return model

    def _scale(self, X):
        """Reshape (N, seq_len, 21) → (N*seq_len, 21), scale, reshape back."""
        n, s, f = X.shape
        return self.scaler.transform(X.reshape(-1, f)).reshape(n, s, f)

    def fit(self, X_train, y_train, X_val, y_val):
        """
        Fit the scaler on training data, build and train the Keras model.

        Args:
            X_train: (N_train, seq_len, 21)
            y_train: (N_train, 2)
            X_val:   (N_val, seq_len, 21)
            y_val:   (N_val, 2)
        """
        cfg = self.config
        n, s, f = X_train.shape

        # Fit scaler on training data only
        self.scaler.fit(X_train.reshape(-1, f))
        X_train_s = self._scale(X_train)
        X_val_s   = self._scale(X_val)

        self.model = self._build_model()
        self.model.summary()

        self.model.fit(
            X_train_s, y_train,
            validation_data=(X_val_s, y_val),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            verbose=1,
        )

        self.is_fitted = True

        train_pred = self.predict(X_train)
        train_rmse = np.sqrt(np.mean(np.sum((train_pred - y_train) ** 2, axis=1)))
        print(f"LSTM Training RMSE: {train_rmse:.2f} pixels")

        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        X_s = self._scale(X)
        return self.model.predict(X_s, verbose=0)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        y = np.array(y)
        errors = predictions - y
        euclidean = np.sqrt(np.sum(errors ** 2, axis=1))
        return {
            'rmse':         np.sqrt(np.mean(np.sum(errors ** 2, axis=1))),
            'mean_error':   np.mean(euclidean),
            'std_error':    np.std(euclidean),
            'max_error':    np.max(euclidean),
            'median_error': np.median(euclidean),
        }
