"""Classification + probability-calibration metrics.

All metrics are computed from the held-out predictions passed in -- nothing is
hard-coded, and every helper returns ``None``/``nan`` gracefully when a split is
too small or single-class.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.ml.config import MIN_POSITIVES_PER_ACTION_FOR_METRICS
from app.ml.features.schema import LABEL


def _safe(fn, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except (ValueError, ZeroDivisionError):
        return None


def classification_metrics(
    y_true, y_prob, threshold: float = 0.5
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    n_pos = int(y_true.sum())
    out = {
        "n": int(len(y_true)),
        "n_positive": n_pos,
        "n_negative": int(len(y_true) - n_pos),
        "positive_rate": round(n_pos / len(y_true), 6) if len(y_true) else None,
        "threshold": threshold,
        "roc_auc": _safe(roc_auc_score, y_true, y_prob),
        "pr_auc": _safe(average_precision_score, y_true, y_prob),
        "log_loss": _safe(log_loss, y_true, y_prob, labels=[0, 1]),
        "brier": _safe(brier_score_loss, y_true, y_prob),
        "precision": _safe(precision_score, y_true, y_pred, zero_division=0),
        "recall": _safe(recall_score, y_true, y_pred, zero_division=0),
        "f1": _safe(f1_score, y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        out["confusion_matrix"] = {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }
    else:
        out["confusion_matrix"] = None

    # The model is used for probability ranking, not 0.5-thresholded
    # classification; report precision/recall at a few operating points.
    base_rate = float(y_true.mean()) if len(y_true) else 0.0
    sweep = []
    for thr in sorted({0.1, 0.2, 0.3, round(base_rate, 3), _top_k_threshold(y_prob, 0.1)}):
        yp = (y_prob >= thr).astype(int)
        sweep.append(
            {
                "threshold": round(float(thr), 4),
                "flagged": int(yp.sum()),
                "precision": _safe(precision_score, y_true, yp, zero_division=0),
                "recall": _safe(recall_score, y_true, yp, zero_division=0),
                "f1": _safe(f1_score, y_true, yp, zero_division=0),
            }
        )
    out["threshold_sweep"] = sweep
    return out


def _top_k_threshold(y_prob: np.ndarray, frac: float) -> float:
    if y_prob.size == 0:
        return 1.0
    k = max(1, int(len(y_prob) * frac))
    return float(np.sort(y_prob)[::-1][k - 1])


def reliability_table(y_true, y_prob, n_bins: int = 10) -> list[dict]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if not mask.any():
            continue
        rows.append(
            {
                "bin": f"[{lo:.1f},{hi:.1f})",
                "n": int(mask.sum()),
                "mean_predicted": round(float(y_prob[mask].mean()), 6),
                "observed_rate": round(float(y_true[mask].mean()), 6),
            }
        )
    return rows


def calibration_metrics(y_true, y_prob, n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    table = reliability_table(y_true, y_prob, n_bins)
    n = len(y_true)
    ece = sum(
        (row["n"] / n) * abs(row["mean_predicted"] - row["observed_rate"])
        for row in table
    ) if n else None
    mce = max(
        (abs(row["mean_predicted"] - row["observed_rate"]) for row in table),
        default=None,
    )
    return {
        "expected_calibration_error": round(ece, 6) if ece is not None else None,
        "max_calibration_error": round(mce, 6) if mce is not None else None,
        "brier": _safe(brier_score_loss, y_true, y_prob),
        "mean_predicted": round(float(y_prob.mean()), 6) if n else None,
        "observed_rate": round(float(y_true.mean()), 6) if n else None,
        "reliability_table": table,
    }


def per_action_metrics(frame: pd.DataFrame, y_prob) -> dict:
    frame = frame.reset_index(drop=True)
    y_prob = np.asarray(y_prob, dtype=float)
    out: dict[str, dict] = {}
    for action, idx in frame.groupby("action").groups.items():
        idx = list(idx)
        yt = frame.loc[idx, LABEL].astype(int).to_numpy()
        yp = y_prob[idx]
        n_pos = int(yt.sum())
        entry = {
            "n": len(idx),
            "n_positive": n_pos,
            "observed_rate": round(float(yt.mean()), 6) if len(idx) else None,
            "mean_predicted": round(float(yp.mean()), 6) if len(idx) else None,
        }
        if n_pos >= MIN_POSITIVES_PER_ACTION_FOR_METRICS and n_pos < len(idx):
            entry["roc_auc"] = _safe(roc_auc_score, yt, yp)
            entry["pr_auc"] = _safe(average_precision_score, yt, yp)
            entry["brier"] = _safe(brier_score_loss, yt, yp)
        else:
            entry["note"] = (
                f"too few positives ({n_pos}) for reliable per-action AUC/Brier"
            )
        out[str(action)] = entry
    return out


def evaluate_predictions(
    frame: pd.DataFrame, y_prob, threshold: float = 0.5
) -> dict:
    y_true = frame[LABEL].astype(int).to_numpy()
    return {
        "classification": classification_metrics(y_true, y_prob, threshold),
        "calibration": calibration_metrics(y_true, y_prob),
        "per_action": per_action_metrics(frame, y_prob),
    }
