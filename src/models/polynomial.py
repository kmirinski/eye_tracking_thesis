import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class GazeEstimator:

    def __init__(self, degree=5, clip_bounds=None):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree)

        self.regressor_x = LinearRegression()
        self.regressor_y = LinearRegression()

        # Optional per-axis (lo, hi) bounds the predictions are clipped to. High-degree
        # polynomials extrapolate wildly on inputs outside the training hull; clipping to
        # the valid gaze range (screen px for ebveye, [0,1] for ev_eye) removes the
        # nonsensical off-screen outliers that otherwise dominate the mean error / DoD.
        self.clip_bounds = clip_bounds

        self.is_fitted = False

    def fit(self, pupil_centers, screen_coords):
        X_poly = self.poly.fit_transform(pupil_centers)
        print(f"Number of polynomial features: {X_poly.shape[1]}")

        self.regressor_x.fit(X_poly, screen_coords[:, 0])
        self.regressor_y.fit(X_poly, screen_coords[:, 1])

        self.is_fitted = True

        train_pred = self.predict(pupil_centers)
        train_error = np.mean(np.sqrt(np.sum((train_pred - screen_coords)**2, axis=1)))
        print(f"Training Distance Error: {train_error:.5f} pixels")

        return self


    def predict(self, pupil_centers):
        if not self.is_fitted:
            raise RuntimeError("Regressor must be fitted before prediction. Call fit() first.")

        if pupil_centers.ndim == 1:
            pupil_centers = pupil_centers.reshape(1, -1)

        X_poly = self.poly.transform(pupil_centers)

        x_s = self.regressor_x.predict(X_poly)
        y_s = self.regressor_y.predict(X_poly)

        predictions = np.column_stack([x_s, y_s])
        if self.clip_bounds is not None:
            lo, hi = self.clip_bounds
            predictions = np.clip(predictions, lo, hi)
        return predictions
    

    def evaluate(self, pupil_centers, screen_coords):
        predictions = self.predict(pupil_centers)
        screen_coords = np.array(screen_coords)

        errors = predictions - screen_coords
        # Distance Error: Euclidean distance (px) between predicted and true gaze point.
        euclidean_errors = np.sqrt(np.sum(errors ** 2, axis=1))

        return {
            'mean_error': np.mean(euclidean_errors),
            'median_error': np.median(euclidean_errors),
        }
    
    def get_coefficients(self):
        if not self.is_fitted:
            raise RuntimeError("Regressor must be fitted first.")
        
        return {
            'theta_x': self.regressor_x.coef_,
            'intercept_x': self.regressor_x.intercept_,
            'theta_y': self.regressor_y.coef_,
            'intercept_y': self.regressor_y.intercept_,
            'feature_names': self.poly.get_feature_names_out(['x_c', 'y_c'])
        }