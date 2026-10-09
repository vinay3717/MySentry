"""
data/schema_mapper.py - Flexible Dataset Ingestion Layer for SENTRY
Transforms arbitrary real-world CSVs into the frozen contract schema
expected by PoisoningDetector: Feature_1..Feature_N + Label (0/1).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

COMMON_LABEL_NAMES = {'label', 'target', 'class', 'outcome', 'quality', 'y'}
COMMON_ID_NAMES = {'id', 'index', 'unnamed: 0'}


def suggest_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Task 1: Auto-detects schema candidates with best-guess defaults:
    - label_guess: case-insensitive match on common label names or last column.
    - id_cols: identifier/index columns to exclude.
    - feature_cols: remaining columns proposed as features.
    """
    if df.empty or len(df.columns) == 0:
        return {'label_guess': None, 'id_cols': [], 'feature_cols': []}

    cols_lower = {str(c).lower().strip(): c for c in df.columns}
    label_guess = next((cols_lower[c] for c in COMMON_LABEL_NAMES if c in cols_lower), df.columns[-1])
    id_cols = [cols_lower[c] for c in COMMON_ID_NAMES if c in cols_lower]
    feature_cols = [c for c in df.columns if c != label_guess and c not in id_cols]

    return {
        'label_guess': label_guess,
        'id_cols': id_cols,
        'feature_cols': feature_cols
    }


def binarize_label(series: pd.Series, threshold: Optional[float] = None) -> pd.Series:
    """
    Task 3: Converts arbitrary target columns to binary 0/1 labels.
    - If already binary (2 unique values), converts directly to 0/1 integers.
    - If continuous/multi-class with threshold provided, splits by threshold (>= threshold -> 1).
    - Otherwise defaults to median split.
    """
    clean_series = series.dropna()
    unique_vals = clean_series.nunique()

    if unique_vals == 2:
        vals = sorted(clean_series.unique())
        val_map = {vals[0]: 0, vals[1]: 1}
        return series.map(val_map).fillna(0).astype(int)

    if threshold is not None:
        return (series >= threshold).astype(int)

    # Default fallback: median split
    median_val = series.median()
    return (series >= median_val).astype(int)


def get_non_numeric_columns(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    """
    Identifies non-numeric feature columns that cannot be fed directly to ML estimators.
    """
    return [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]


def reshape_to_contract(
    df: pd.DataFrame,
    label_col: str,
    exclude_cols: Optional[List[str]] = None,
    binarize_threshold: Optional[float] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Task 4: Reshapes and renames any tabular CSV to match the exact frozen contract:
    Feature_1..Feature_N + Label (binary 0/1).
    Returns:
        (reshaped_df, excluded_non_numeric_cols)
    """
    if exclude_cols is None:
        exclude_cols = []

    feature_cols = [c for c in df.columns if c != label_col and c not in exclude_cols]
    non_numeric_dropped = get_non_numeric_columns(df, feature_cols)

    numeric_features = df[feature_cols].select_dtypes(include='number')
    renamed = numeric_features.rename(
        columns={c: f'Feature_{i+1}' for i, c in enumerate(numeric_features.columns)}
    )
    renamed['Label'] = binarize_label(df[label_col], binarize_threshold)

    return renamed, non_numeric_dropped


def validate_dataset_mappable(
    df: pd.DataFrame,
    label_col: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Task 5: Validates whether a dataset can be mapped to SENTRY's detection pipeline.
    Provides clear, actionable error feedback instead of blunt rejections.
    """
    if df.empty:
        return False, "Uploaded CSV is empty (0 rows)."

    if len(df.columns) < 2:
        return False, "Dataset must contain at least 2 columns (1 feature + 1 label candidate)."

    target_label = label_col or df.columns[-1]
    candidate_features = [c for c in df.columns if c != target_label]
    numeric_count = len(df[candidate_features].select_dtypes(include='number').columns)

    if numeric_count == 0:
        return False, "No numeric feature columns found. SENTRY requires numeric tabular features for outlier & decision-boundary analysis."

    if df[target_label].nunique() < 2:
        return False, f"Selected label column '{target_label}' contains only 1 unique class. Binary classification requires at least 2 classes."

    return True, "Valid"
