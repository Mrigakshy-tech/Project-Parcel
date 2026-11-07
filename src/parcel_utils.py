"""Small, testable helpers for the Project Parcel demonstration.

Connect these functions to the fitted pipeline and exact feature schema in the
notebook before treating them as a production inference package.
"""
from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np
import pandas as pd


def validate_input(
    frame: pd.DataFrame,
    required_features: Sequence[str],
    numeric_features: Iterable[str] = (),
) -> pd.DataFrame:
    """Validate a batch and return a copy with numeric columns coerced safely.

    Missing values are allowed where the fitted preprocessing pipeline supports
    imputation. Missing required columns, duplicate column names, empty batches,
    and infinite numeric values are rejected explicitly.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError("Input batch is empty.")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate input columns: {duplicates}")
    missing = [c for c in required_features if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    result = frame.copy()
    for col in numeric_features:
        if col not in required_features:
            continue
        result[col] = pd.to_numeric(result[col], errors="coerce")
        values = result[col].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any():
            raise ValueError(f"Feature '{col}' contains infinite values.")
    return result


def predict_risk(
    model,
    frame: pd.DataFrame,
    required_features: Sequence[str],
    numeric_features: Iterable[str] = (),
) -> pd.DataFrame:
    """Return late-risk scores and binary predictions from a fitted classifier."""
    clean = validate_input(frame, required_features, numeric_features)
    X = clean.loc[:, list(required_features)]
    if not hasattr(model, "predict_proba"):
        raise TypeError("Model must provide predict_proba().")
    probabilities = model.predict_proba(X)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Model did not return a positive-class probability.")
    risk = np.asarray(probabilities[:, 1], dtype=float)
    if not np.isfinite(risk).all() or ((risk < 0) | (risk > 1)).any():
        raise ValueError("Model returned invalid probability values.")
    return pd.DataFrame(
        {"late_risk": risk, "predicted_late": (risk >= 0.5).astype(int)},
        index=frame.index,
    )


def population_stability_index(reference, current, bins: int = 10) -> float:
    """Calculate PSI for numeric samples using reference quantile bins."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        raise ValueError("Reference and current samples must contain finite values.")
    if bins < 2:
        raise ValueError("bins must be at least 2.")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        # Constant / near-constant reference: use a single split around its value.
        center = float(ref[0])
        spread = max(abs(center) * 1e-6, 1e-6)
        edges = np.array([-np.inf, center - spread, center + spread, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    eps = 1e-6
    ref_pct = np.maximum(ref_counts / ref_counts.sum(), eps)
    cur_pct = np.maximum(cur_counts / cur_counts.sum(), eps)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
