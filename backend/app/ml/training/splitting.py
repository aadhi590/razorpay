"""Grouped, chronological train / validation / test split.

Why not a random split: rows are recovery *attempts* through time, and multiple
rows share a recovery event. A random split would (a) let a model peek at "the
future" and (b) put attempt 1 of an event in train and attempt 3 in test.

Strategy:
  1. Assign every recovery event a timestamp = its earliest ``as_of``.
  2. Order events by that timestamp.
  3. Oldest 60% of events -> train, next 20% -> validation, newest 20% -> test.
  4. Every row follows its event, so no event crosses a boundary.

Customers may still appear in more than one split (a real, expected situation --
you do see repeat customers over time). That does not leak the label: every
customer-history feature for a row is computed only from data before that row's
own ``as_of``. Customer overlap is reported for transparency.

Synthetic-data caveat: ``as_of`` here derives from ``payment.failed_at``, which
the generator samples ~uniformly over a year. The ordering is therefore the
simulator's sampling order, not real business-time evolution. A production split
must use true event time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.ml.config import SPLIT_FRACTIONS
from app.ml.features.schema import LABEL


@dataclass
class SplitResult:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    report: dict = field(default_factory=dict)

    def frames(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (
            df.iloc[self.train_idx].reset_index(drop=True),
            df.iloc[self.val_idx].reset_index(drop=True),
            df.iloc[self.test_idx].reset_index(drop=True),
        )


def grouped_chronological_split(
    df: pd.DataFrame,
    fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
) -> SplitResult:
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1.0, got {fractions}")

    event_time = (
        df.groupby("recovery_event_id")["as_of"].min().sort_values()
    )
    events = event_time.index.to_numpy()
    n = len(events)
    n_train = int(round(n * fractions[0]))
    n_val = int(round(n * fractions[1]))

    train_events = set(events[:n_train])
    val_events = set(events[n_train:n_train + n_val])
    test_events = set(events[n_train + n_val:])

    ev = df["recovery_event_id"]
    train_idx = np.where(ev.isin(train_events))[0]
    val_idx = np.where(ev.isin(val_events))[0]
    test_idx = np.where(ev.isin(test_events))[0]

    def _cust(idx: np.ndarray) -> set:
        return set(df.iloc[idx]["customer_id"].tolist())

    ct, cv, cte = _cust(train_idx), _cust(val_idx), _cust(test_idx)

    def _span(idx: np.ndarray) -> dict:
        s = df.iloc[idx]
        return {
            "rows": int(len(s)),
            "events": int(s["recovery_event_id"].nunique()),
            "customers": int(s["customer_id"].nunique()),
            "positives": int(s[LABEL].sum()),
            "positive_rate": round(float(s[LABEL].mean()), 6) if len(s) else None,
            "as_of_min": str(s["as_of"].min()),
            "as_of_max": str(s["as_of"].max()),
        }

    report = {
        "strategy": "grouped(recovery_event) + chronological(min as_of)",
        "fractions": list(fractions),
        "train": _span(train_idx),
        "validation": _span(val_idx),
        "test": _span(test_idx),
        "event_overlap": {
            "train_val": len(train_events & val_events),
            "train_test": len(train_events & test_events),
            "val_test": len(val_events & test_events),
        },
        "customer_overlap": {
            "train_val": len(ct & cv),
            "train_test": len(ct & cte),
            "val_test": len(cv & cte),
            "note": (
                "customer overlap is expected and does not leak the label; "
                "history features are point-in-time per row"
            ),
        },
    }
    return SplitResult(train_idx, val_idx, test_idx, report)
