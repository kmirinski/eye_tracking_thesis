import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class GazeEstimator:

    def __init__(self, degree=5):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree)

        self.regressor_x = LinearRegression()
        self.regressor_y = LinearRegression()

        self.is_fitted = False

    def fit(self, pupil_centers, screen_coords):
        # pupil_centers = np.array(pupil_centers)
        # screen_coords = np.array(screen_coords)

        N = pupil_centers.shape[0]
        min_samples = (self.degree + 1) * (self.degree * 2) // 2

        if N < min_samples:
            print(f"Warning: You have fewer samples ({N}) than coefficients ({min_samples}). "
                  f"Consider collecting more data or reducing polynomial degree.")
        
        X_poly = self.poly.fit_transform(pupil_centers)
        print(f"Number of polynomial features: {X_poly.shape[1]}")

        self.regressor_x.fit(X_poly, screen_coords[:, 0])
        self.regressor_y.fit(X_poly, screen_coords[:, 1])

        self.is_fitted = True

        train_pred = self.predict(pupil_centers)
        train_error = np.sqrt(np.mean(np.sum((train_pred - screen_coords)**2, axis=1)))
        print(f"Training RMSE: {train_error:.2f} pixels")

        return self

    
    def predict(self, pupil_centers):
        if not self.is_fitted:
            raise RuntimeError("Regressor must be fitted before prediction. Call fit() first.")

        pupil_centers = np.array(pupil_centers)          
        if pupil_centers.ndim == 1:
            pupil_centers = pupil_centers.reshape(1, -1)
        
        X_poly = self.poly.transform(pupil_centers)

        x_s = self.regressor_x.predict(X_poly)
        y_s = self.regressor_y.predict(X_poly)

        return np.column_stack([x_s, y_s])
    

    def evaluate(self, pupil_centers, screen_coords):
        predictions = self.predict(pupil_centers)
        screen_coords = np.array(screen_coords)

        errors = predictions - screen_coords
        euclidean_errors = np.sqrt(np.sum(errors ** 2, axis = 1))

        metrics = {
            'rmse': np.sqrt(np.mean(euclidean_errors**2)),
            'mean_error': np.mean(euclidean_errors),
            'std_error': np.std(euclidean_errors),
            'max_error': np.max(euclidean_errors),
            'median_error': np.median(euclidean_errors)
        }
        
        return metrics
    
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