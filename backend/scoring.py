"""
backend/scoring.py - Scoring thresholds and verdict mapping for SENTRY
"""


def get_recommendation(score: float) -> str:
    """
    Maps a 0-100 poisoning score to a human-readable recommendation string.
    Thresholds match BUILD.md and BUILD_T1_backend_detection.md exactly.
    """
    if score < 5:
        return "✅ Dataset appears clean"
    elif score < 20:
        return "⚠️ Minor anomalies detected - investigate further"
    elif score < 50:
        return "🚨 Significant poisoning detected - DO NOT USE"
    else:
        return "🔴 CRITICAL: Dataset is heavily poisoned - reject immediately"


def get_verdict(score: float) -> str:
    """
    Maps a 0-100 poisoning score to the categorical verdict from BUILD.md.
    """
    if score < 5:
        return "Clean"
    elif score < 20:
        return "Minor anomalies"
    elif score < 50:
        return "Significant"
    else:
        return "Critical"
