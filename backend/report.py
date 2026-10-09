"""
backend/report.py - Plain-text audit report formatter for SENTRY
"""

from datetime import datetime
from typing import Any, Dict


def format_text_report(report: Dict[str, Any]) -> str:
    """
    Formats a detection report dictionary into a clean plain-text summary report.
    Compatible with the frozen contract schema from BUILD.md.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_name = report.get('dataset', 'Unknown')
    total_samples = report.get('total_samples', 0)
    poisoning_score = report.get('poisoning_score', 0.0)
    anomaly_score = report.get('anomaly_score', 0.0)
    label_flip_score = report.get('label_flip_score', 0.0)
    suspicious_samples = report.get('suspicious_samples', 0)
    suspicious_indices = report.get('suspicious_indices', [])
    recommendation = report.get('recommendation', 'N/A')

    if suspicious_indices:
        flagged_preview = ", ".join(map(str, suspicious_indices[:30]))
        if len(suspicious_indices) > 30:
            flagged_preview += f" ... (+{len(suspicious_indices) - 30} more)"
    else:
        flagged_preview = "None"

    return f"""======================================================================
SENTRY: DATA POISONING DETECTION AUDIT REPORT
======================================================================
Generated:           {timestamp}
Dataset Name:        {dataset_name}
Total Samples:       {total_samples}

----------------------------------------------------------------------
DETECTION METRICS
----------------------------------------------------------------------
Poisoning Score:     {poisoning_score:.2f}%
Anomaly Score:       {anomaly_score:.2f}%
Label-Flip Score:    {label_flip_score:.2f}%
Suspicious Samples:  {suspicious_samples} / {total_samples} ({(suspicious_samples / max(1, total_samples) * 100):.1f}%)

----------------------------------------------------------------------
VERDICT & RECOMMENDATION
----------------------------------------------------------------------
Recommendation:      {recommendation}

----------------------------------------------------------------------
SUSPICIOUS ROW INDICES (0-indexed)
----------------------------------------------------------------------
{flagged_preview}
======================================================================
"""


# Alias matching frontend nomenclature
generate_text_report = format_text_report
