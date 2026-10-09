"""
backend/detection.py - Core Poisoning Detection Engine for SENTRY
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


class PoisoningDetector:
    """
    SENTRY Poisoning Detector engine.
    Detects dataset poisoning via anomaly detection and label-flip analysis.
    """

    def __init__(self, contamination: float = 0.1, random_state: int = 42, n_neighbors: int = 20):
        self.contamination = contamination
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
        self.lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)

    def scale_features(self, X: np.ndarray) -> np.ndarray:
        """
        Standardizes features by removing the mean and scaling to unit variance.
        """
        return self.scaler.fit_transform(X)

    def detect_outliers_isolation_forest(self, X: np.ndarray):
        """
        Runs Isolation Forest anomaly detection on feature matrix X.
        Returns:
            predictions: ndarray of shape (n_samples,) where -1 is outlier, 1 is inlier.
            scores: ndarray of shape (n_samples,) with anomaly scores.
        """
        predictions = self.iso_forest.fit_predict(X)
        scores = self.iso_forest.score_samples(X)
        return predictions, scores

    def detect_outliers_lof(self, X: np.ndarray):
        """
        Runs Local Outlier Factor anomaly detection on feature matrix X.
        Returns:
            predictions: ndarray of shape (n_samples,) where -1 is outlier, 1 is inlier.
            scores: ndarray of shape (n_samples,) with negative outlier factors.
        """
        predictions = self.lof.fit_predict(X)
        scores = self.lof.negative_outlier_factor_
        return predictions, scores
