"""
backend/detection.py - Core Poisoning Detection Engine for SENTRY
"""

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
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

    def detect_label_flip_attacks(self, X: np.ndarray, y: np.ndarray, confidence_threshold: float = 0.6):
        """
        Detects potential label-flip attacks by training a classifier and flagging
        samples with low confidence or mismatched labels.
        Returns:
            suspicious: boolean ndarray indicating suspicious samples
            confidence: ndarray of maximum predicted class probabilities
            predictions: ndarray of model predictions
        """
        model = RandomForestClassifier(n_estimators=50, random_state=self.random_state, oob_score=True)
        model.fit(X, y)

        if hasattr(model, 'oob_decision_function_') and model.oob_decision_function_ is not None:
            probs = model.oob_decision_function_
            if np.isnan(probs).any():
                in_sample_probs = model.predict_proba(X)
                probs = np.where(np.isnan(probs), in_sample_probs, probs)
            preds = np.argmax(probs, axis=1)
        else:
            probs = model.predict_proba(X)
            preds = model.predict(X)

        confidence = np.max(probs, axis=1)
        suspicious = (confidence < confidence_threshold) | (preds != y)
        return suspicious, confidence, preds

    def compute_poisoning_score(self, X: np.ndarray, y: np.ndarray):
        """
        Combines anomaly detection (Isolation Forest + LOF) and label-flip
        detection into a single 0-100 poisoning score, plus per-attack scores.
        Returns:
            poisoning_score: float 0-100
            anomaly_score: float 0-100
            label_flip_score: float 0-100
            suspicious_indices: list of int row indices flagged by any detector
        """
        n_samples = len(X)

        # Scale features for anomaly detectors
        X_scaled = self.scale_features(X)

        # --- Anomaly detection ---
        iso_preds, iso_scores = self.detect_outliers_isolation_forest(X_scaled)
        lof_preds, lof_scores = self.detect_outliers_lof(X_scaled)

        # Combine: samples flagged by BOTH detectors are strong anomalies
        both_flagged = (iso_preds == -1) & (lof_preds == -1)
        either_flagged = (iso_preds == -1) | (lof_preds == -1)

        # Subtract expected baseline: contamination param forces ~contamination fraction
        # to be flagged even on clean data. Only excess flags indicate real poisoning.
        expected_flag_rate = self.contamination
        strong_ratio = max(0, (both_flagged.sum() / n_samples) - expected_flag_rate * 0.5)
        weak_ratio = max(0, (either_flagged.sum() / n_samples) - expected_flag_rate)

        # Anomaly score: scale so 20% excess flagging = score ~100
        anomaly_score = min((strong_ratio * 300 + weak_ratio * 200), 100.0)

        # --- Label-flip detection ---
        label_flip_flags, confidence, _ = self.detect_label_flip_attacks(X_scaled, y)

        # Subtract expected baseline: even clean data has some low-confidence boundary samples
        # Typically ~10-15% on clean data with OOB predictions
        expected_flip_baseline = 0.12
        flip_ratio = max(0, (label_flip_flags.sum() / n_samples) - expected_flip_baseline)

        # Severity: how low is the average confidence of flagged samples
        if label_flip_flags.sum() > 0:
            avg_conf_suspicious = confidence[label_flip_flags].mean()
            severity_boost = max(0, (0.55 - avg_conf_suspicious) * 150)
        else:
            severity_boost = 0.0

        label_flip_score = min(flip_ratio * 250 + severity_boost, 100.0)

        # --- Ensemble: combine into single score ---
        # Label-flip is the stronger, more actionable signal for this tool
        poisoning_score = min(0.35 * anomaly_score + 0.65 * label_flip_score, 100.0)

        # Suspicious = flagged by any detector
        all_suspicious = either_flagged | label_flip_flags
        suspicious_indices = list(np.where(all_suspicious)[0])

        return poisoning_score, anomaly_score, label_flip_score, suspicious_indices



    @staticmethod
    def _get_recommendation(score: float) -> str:
        """Maps a poisoning score to a human-readable recommendation string."""
        if score < 5:
            return "✅ Dataset appears clean"
        elif score < 20:
            return "⚠️ Minor anomalies detected - investigate further"
        elif score < 50:
            return "🚨 Significant poisoning detected - DO NOT USE"
        else:
            return "🔴 CRITICAL: Dataset is heavily poisoned - reject immediately"

    def generate_report(self, dataset_name: str, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Main entry point for T2's dashboard. Runs the full detection pipeline
        and returns the frozen-contract dict.
        Returns:
        {
          'dataset': str,
          'total_samples': int,
          'poisoning_score': float,       # 0-100
          'suspicious_samples': int,
          'anomaly_score': float,         # 0-100
          'label_flip_score': float,      # 0-100
          'suspicious_indices': list[int],# row indices flagged
          'recommendation': str
        }
        """
        poisoning_score, anomaly_score, label_flip_score, suspicious_indices = \
            self.compute_poisoning_score(X, y)

        return {
            'dataset': dataset_name,
            'total_samples': len(X),
            'poisoning_score': round(poisoning_score, 2),
            'suspicious_samples': len(suspicious_indices),
            'anomaly_score': round(anomaly_score, 2),
            'label_flip_score': round(label_flip_score, 2),
            'suspicious_indices': suspicious_indices,
            'recommendation': self._get_recommendation(poisoning_score),
        }
