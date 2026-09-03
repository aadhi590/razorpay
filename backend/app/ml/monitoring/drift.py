"""Monitoring readiness: a training-time feature baseline, a PSI drift check
against it, and an append-only prediction log.

This is deliberately lightweight -- it establishes the hooks a real deployment
needs (compare live feature/prediction distributions to training, alert on
drift) without a metrics backend.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.config import ARTIFACT_DIR
from app.ml.features.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES

_PSI_BINS = 10
PREDICTION_LOG_PATH = ARTIFACT_DIR / "prediction_log.jsonl"


def build_feature_baseline(frame: pd.DataFrame) -> dict:
    baseline: dict = {"numeric": {}, "categorical": {}, "n_rows": int(len(frame))}
    for col in NUMERIC_FEATURES:
        s = pd.to_numeric(frame[col], errors="coerce")
        q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
        baseline["numeric"][col] = {
            "mean": _f(s.mean()),
            "std": _f(s.std()),
            "min": _f(s.min()),
            "max": _f(s.max()),
            "p05": _f(q.get(0.05)),
            "p25": _f(q.get(0.25)),
            "p50": _f(q.get(0.5)),
            "p75": _f(q.get(0.75)),
            "p95": _f(q.get(0.95)),
            "null_fraction": _f(s.isna().mean()),
            # histogram edges (on non-null values) for PSI
            "hist_edges": _quantile_edges(s.dropna().to_numpy()),
        }
    for col in CATEGORICAL_FEATURES:
        vc = frame[col].astype("string").fillna("__MISSING__").value_counts(normalize=True)
        baseline["categorical"][col] = {str(k): float(v) for k, v in vc.items()}
    return baseline


def _f(x) -> float | None:
    try:
        v = float(x)
        return None if np.isnan(v) else round(v, 6)
    except (TypeError, ValueError):
        return None


def _quantile_edges(values: np.ndarray, n_bins: int = _PSI_BINS) -> list[float]:
    if values.size == 0:
        return []
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(values, qs))
    if edges.size < 2:
        edges = np.array([values.min(), values.max() + 1e-9])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return [float(e) for e in edges]


def population_stability_index(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI over two discrete distributions (proportions). <0.1 stable,
    0.1-0.25 moderate shift, >0.25 significant shift."""
    expected = np.clip(np.asarray(expected, dtype=float), 1e-6, None)
    actual = np.clip(np.asarray(actual, dtype=float), 1e-6, None)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_report(baseline: dict, frame: pd.DataFrame) -> dict:
    out: dict = {"numeric": {}, "categorical": {}, "generated_at": _now()}
    for col, stats in baseline.get("numeric", {}).items():
        edges = stats.get("hist_edges")
        s = pd.to_numeric(frame[col], errors="coerce").dropna().to_numpy()
        if not edges or len(edges) < 2 or s.size == 0:
            out["numeric"][col] = {"psi": None, "note": "insufficient data"}
            continue
        # training data was quantile-binned -> expected proportions ~ uniform.
        n_bins = len(edges) - 1
        expected = np.full(n_bins, 1.0 / n_bins)
        actual, _ = np.histogram(s, bins=np.asarray(edges))
        psi = population_stability_index(expected, actual)
        out["numeric"][col] = {
            "psi": round(psi, 6),
            "status": _psi_status(psi),
            "live_mean": _f(s.mean()),
            "baseline_mean": stats.get("mean"),
        }
    for col, ref in baseline.get("categorical", {}).items():
        live = (
            frame[col].astype("string").fillna("__MISSING__")
            .value_counts(normalize=True).to_dict()
        )
        cats = sorted(set(ref) | set(map(str, live)))
        exp = np.array([ref.get(c, 0.0) for c in cats])
        act = np.array([live.get(c, 0.0) for c in cats])
        psi = population_stability_index(exp, act)
        unseen = [c for c in map(str, live) if c not in ref]
        out["categorical"][col] = {
            "psi": round(psi, 6),
            "status": _psi_status(psi),
            "unseen_categories": unseen,
        }
    return out


def _psi_status(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "moderate_shift"
    return "significant_shift"


def log_prediction(record: dict, path: Path | None = None) -> None:
    path = Path(path) if path is not None else PREDICTION_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"logged_at": _now(), **record}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
