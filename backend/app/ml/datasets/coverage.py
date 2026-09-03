"""Action coverage & overlap analysis.

Answers: does the observed data actually support a per-action model, or are some
actions too thin to trust? Also reports how often two actions were both observed
on the *same* recovery event (the only within-event "overlap" the current data
provides -- true counterfactual overlap does not exist here).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.ml.config import ACTIONS, MIN_EXAMPLES_PER_ACTION
from app.ml.features.schema import LABEL


@dataclass
class ActionCoverage:
    per_action: dict = field(default_factory=dict)
    within_event_action_pairs: dict = field(default_factory=dict)
    insufficient_actions: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "per_action": self.per_action,
            "within_event_action_pairs": self.within_event_action_pairs,
            "insufficient_actions": self.insufficient_actions,
            "notes": self.notes,
        }


def action_coverage(frame: pd.DataFrame) -> ActionCoverage:
    cov = ActionCoverage()

    for action in ACTIONS:
        sub = frame[frame["action"] == action]
        n = int(len(sub))
        pos = int(sub[LABEL].sum())
        cov.per_action[action] = {
            "examples": n,
            "recovered": pos,
            "not_recovered": n - pos,
            "recovery_rate": round(pos / n, 6) if n else None,
            "distinct_events": int(sub["recovery_event_id"].nunique()),
            "sufficient": n >= MIN_EXAMPLES_PER_ACTION,
        }
        if n < MIN_EXAMPLES_PER_ACTION:
            cov.insufficient_actions.append(action)

    by_event = frame.groupby("recovery_event_id")["action"].agg(set)
    pair_counts: dict[str, int] = {}
    for actions in by_event:
        ordered = sorted(actions)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                key = f"{ordered[i]}+{ordered[j]}"
                pair_counts[key] = pair_counts.get(key, 0) + 1
    cov.within_event_action_pairs = dict(
        sorted(pair_counts.items(), key=lambda kv: -kv[1])
    )

    cov.notes.append(
        "No fabricated counterfactuals: rows exist only for actions actually "
        "executed on an event."
    )
    cov.notes.append(
        "Control events (no intervention) are excluded from this dataset; they "
        "are reserved for future uplift modelling."
    )
    if cov.insufficient_actions:
        cov.notes.append(
            "Insufficient data for: "
            + ", ".join(cov.insufficient_actions)
            + f" (< {MIN_EXAMPLES_PER_ACTION} examples) -- per-action estimates "
            "for these actions are not reliable."
        )
    return cov
