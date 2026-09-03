"""Uplift training driver: DB -> dataset -> validate -> split -> compare
learners -> select champion on held-out policy value -> evaluate on test ->
artifact + report.

Invoked only by ``app.scripts.train_uplift_model`` (CLI). Never at import or API
startup.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.features.schema import ALL_FEATURES
from app.ml.training.splitting import grouped_chronological_split
from app.ml.uplift import config as ucfg
from app.ml.uplift.dataset.builder import UpliftDataset, build_uplift_dataset
from app.ml.uplift.estimators.learners import build_estimator
from app.ml.uplift.evaluation.metrics import evaluate_uplift
from app.ml.uplift.models.artifact import UpliftArtifact, save_uplift_artifact
from app.ml.uplift.validation import validate_propensity, validate_uplift_frame

# (learner_type, base classifier) candidates compared on the validation split.
_CANDIDATES: list[tuple[str, str]] = [
    ("t_learner", "hist_gb"),
    ("s_learner", "hist_gb"),
    ("s_learner", "logreg"),
]

_SYNTHETIC_WARNING = (
    "Trained on synthetic data from app/scripts/generate_data.py "
    "--randomized-assignment, whose outcome process is a known closed form. "
    "Every uplift / Qini / policy-value number is a SYNTHETIC BENCHMARK, not "
    "real-world causal evidence. The control arm is small (tens of positives), "
    "so treat bootstrap intervals, not point estimates, as the result."
)


@dataclass
class UpliftTrainingRun:
    artifact_path: Path
    report_path: Path
    artifact: UpliftArtifact
    report: dict = field(default_factory=dict)


def _attach_predictions(estimator, frame: pd.DataFrame) -> pd.DataFrame:
    actions = list(ucfg.TREATMENT_ACTIONS)
    out = frame.reset_index(drop=True).copy()
    base = estimator.predict_baseline(out[ALL_FEATURES])
    cols = []
    for a in actions:
        u = estimator.predict_action(out[ALL_FEATURES], a) - base
        out[f"uplift_{a}"] = u
        cols.append(u)
    U = np.column_stack(cols)
    out["baseline_probability"] = base
    out["tau_hat"] = U.max(axis=1)
    out["best_action"] = [actions[i] for i in U.argmax(axis=1)]
    return out


def _champion_score(evaluation: dict) -> float:
    """Primary selection metric: held-out policy-value gain of the uplift policy
    over random action assignment (the business objective). Falls back to the
    Qini coefficient when the policy-value estimate is unavailable."""
    pv = evaluation.get("policy_value", {}).get("uplift_policy", {})
    gain = pv.get("gain_vs_random_action")
    if gain is not None:
        return float(gain)
    return float(evaluation.get("qini", {}).get("qini_coefficient", 0.0) or 0.0) * 0.01


def run_uplift_training(
    conn,
    *,
    out_dir: Path | None = None,
    fast: bool = False,
) -> UpliftTrainingRun:
    out_dir = Path(out_dir) if out_dir is not None else ucfg.UPLIFT_ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if fast:
        # fewer bootstrap resamples for the Qini CI (still enough for a range)
        ucfg.QINI_BOOTSTRAP_SAMPLES = 60

    # 1. dataset --------------------------------------------------
    dataset: UpliftDataset = build_uplift_dataset(conn)
    frame = dataset.frame

    # 2. split (event-grouped, chronological) -- one row per event ---
    split = grouped_chronological_split(frame, fractions=ucfg.SPLIT_FRACTIONS)
    train_df, val_df, test_df = split.frames(frame)

    # 3. validation (FAILS LOUDLY on leakage / integrity) ---------
    propensity = validate_propensity(frame)
    integrity = validate_uplift_frame(
        frame, split_event_overlap=split.report["event_overlap"]
    )

    candidates = _CANDIDATES if not fast else _CANDIDATES[:2]

    # 4. compare learners on the validation split -----------------
    comparison: dict[str, dict] = {}
    for learner_type, base in candidates:
        name = f"{learner_type}:{base}"
        est = build_estimator(learner_type, base=base)
        est.fit(train_df)
        val_pred = _attach_predictions(est, val_df)
        val_eval = evaluate_uplift(val_pred)
        comparison[name] = {
            "learner_type": learner_type,
            "base_algorithm": base,
            "validation": {
                "qini_coefficient": val_eval["qini"]["qini_coefficient"],
                "policy_value": val_eval["policy_value"],
                "uplift_at_k": val_eval["uplift_at_k"],
            },
            "selection_score": _champion_score(val_eval),
        }

    champion_name = max(comparison, key=lambda n: comparison[n]["selection_score"])
    champion_spec = dict(zip(("learner_type", "base"), champion_name.split(":")))
    champion_reason = (
        f"selected '{champion_name}': highest held-out policy-value gain over "
        f"random action assignment "
        f"({comparison[champion_name]['selection_score']:+.5f} recovery-rate "
        f"points); ties broken toward the Qini coefficient. "
        f"Scores: "
        + ", ".join(
            f"{n}={comparison[n]['selection_score']:+.5f}" for n in comparison
        )
    )

    # 5. refit champion on train+val, evaluate on untouched test ---
    trainval = pd.concat([train_df, val_df], ignore_index=True)
    champion = build_estimator(champion_spec["learner_type"], base=champion_spec["base"])
    champion.fit(trainval)

    test_pred = _attach_predictions(champion, test_df)
    test_eval = evaluate_uplift(test_pred)

    # 6. artifact -----------------------------------------------
    limitations = _limitations(dataset, propensity, test_eval)
    artifact = UpliftArtifact(
        estimator=champion,
        learner_type=champion_spec["learner_type"],
        base_algorithm=champion_spec["base"],
        champion_reason=champion_reason,
        dataset={
            "dataset_version": dataset.dataset_version,
            "feature_version": dataset.feature_version,
            **dataset.stats,
        },
        split=split.report,
        propensity_diagnostics=propensity.as_dict(),
        integrity=integrity.as_dict(),
        model_comparison=comparison,
        evaluation={"test": test_eval},
        estimator_metadata=(
            champion.metadata.__dict__ if getattr(champion, "metadata", None) else {}
        ),
        config={
            "random_state": ucfg.RANDOM_STATE,
            "split_fractions": list(ucfg.SPLIT_FRACTIONS),
            "candidates": [f"{lt}:{b}" for lt, b in candidates],
            "champion": champion_name,
            "fast_mode": fast,
            "selection_metric": "heldout_policy_value_gain_vs_random_action",
        },
        limitations=limitations,
    )
    artifact_path = save_uplift_artifact(
        artifact, out_dir / ucfg.UPLIFT_ARTIFACT_FILENAME
    )

    report = {
        "model_name": ucfg.UPLIFT_MODEL_NAME,
        "model_version": ucfg.UPLIFT_MODEL_VERSION,
        "created_at": artifact.created_at,
        "synthetic_benchmark": True,
        "synthetic_warning": _SYNTHETIC_WARNING,
        "dataset": artifact.dataset,
        "split": {k: split.report[k] for k in ("strategy", "fractions", "train", "validation", "test", "event_overlap")},
        "propensity_diagnostics": artifact.propensity_diagnostics,
        "integrity": artifact.integrity,
        "model_comparison": comparison,
        "champion": champion_name,
        "champion_reason": champion_reason,
        "estimator_metadata": artifact.estimator_metadata,
        "test_evaluation": test_eval,
        "config": artifact.config,
        "limitations": limitations,
    }
    report_path = out_dir / f"{ucfg.UPLIFT_MODEL_NAME}_{ucfg.UPLIFT_MODEL_VERSION}.report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / f"{ucfg.UPLIFT_MODEL_NAME}_{ucfg.UPLIFT_MODEL_VERSION}.report.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    return UpliftTrainingRun(
        artifact_path=artifact_path,
        report_path=report_path,
        artifact=artifact,
        report=report,
    )


def _limitations(dataset, propensity, test_eval) -> list[str]:
    s = dataset.stats
    lims = [
        "SYNTHETIC DATA. Results are a benchmark against a known generator, not "
        "evidence of a real causal effect.",
        f"Control arm is small: {s['n_control']} rows / "
        f"{s['per_arm']['none']['positives']} natural recoveries. mu_0 and every "
        "uplift number built on it are high-variance.",
        "Treatment rows carry a UNIFORMLY RANDOM first action, not the policy's "
        "choice, so Qini / uplift@k estimate the value of targeting WHO to treat "
        "with a random action -- a lower bound on the value of also choosing WHAT.",
        "First-decision only: escalation attempts 2-3 are excluded, so the model "
        "says nothing about sequential recovery strategy.",
        "Decision time is the recovery event's creation time for both arms; the "
        "generator's reconstructed payment schedule and ~uniform failure-time "
        "sampling still apply (see app/ml/README.md).",
        "No hidden-confounding adjustment beyond the logged decision context; "
        "valid here only because assignment is known-randomized.",
    ]
    if not test_eval["statistical_reliability"]["reliable"]:
        lims.append(
            "Held-out control positives are in the low tens: the champion was "
            "selected on a noisy validation signal -- the S/T-learner choice is "
            "not statistically decisive."
        )
    return lims


def _render_markdown(r: dict) -> str:
    d = r["dataset"]
    te = r["test_evaluation"]
    pv = te["policy_value"]
    lines = [
        f"# {r['model_name']} {r['model_version']} -- uplift training report",
        "",
        f"_Created: {r['created_at']}_",
        "",
        "> **SYNTHETIC BENCHMARK.** " + r["synthetic_warning"],
        "",
        "## Dataset (one row per first recovery decision)",
        f"- rows: {d['n_rows']}  |  events: {d['n_recovery_events']}  |  customers: {d['n_customers']}",
        f"- control: {d['n_control']} rows ({d['per_arm']['none']['positives']} recovered, "
        f"rate {d['control_recovery_rate']})",
        f"- treatment: {d['n_treatment']} rows, rate {d['treatment_recovery_rate']}  "
        f"(naive lift {d['naive_observed_lift']})",
        "",
        "### Per-arm",
        "| arm | rows | positives | recovery rate | mean propensity |",
        "|---|---|---|---|---|",
    ]
    for arm, m in d["per_arm"].items():
        lines.append(
            f"| {arm} | {m['rows']} | {m['positives']} | {m['recovery_rate']} | {m['mean_propensity']} |"
        )
    ipw = r["propensity_diagnostics"]["diagnostics"].get("ipw", {})
    lines += [
        "",
        "## Propensity / overlap",
        f"- realized control fraction: {r['propensity_diagnostics']['diagnostics']['control_fraction']['realized']} "
        f"(design {r['propensity_diagnostics']['diagnostics']['control_fraction']['design']})",
        f"- stabilized-IPW effective sample size: {ipw.get('effective_sample_size')} / {ipw.get('n_rows')} "
        f"({ipw.get('ess_fraction')})",
        f"- recommendation: {r['propensity_diagnostics']['diagnostics'].get('weighting_recommendation', '')}",
        "",
        "## Learner comparison (validation policy-value gain vs random action)",
        "| learner | selection score |",
        "|---|---|",
    ]
    for n, m in r["model_comparison"].items():
        lines.append(f"| {n} | {m['selection_score']:+.5f} |")
    lines += [
        "",
        f"**Champion: {r['champion']}** -- {r['champion_reason']}",
        "",
        "## Held-out test evaluation",
        f"- n={te['n_heldout']} (treated {te['n_treated']}, control {te['n_control']}; "
        f"control positives {te['positives_control']})",
        f"- Qini coefficient: {te['qini']['qini_coefficient']}  "
        f"(bootstrap 95% CI {te['qini']['bootstrap_ci'].get('ci_95')})",
        f"- total incremental responders (Qini @100%): {te['qini']['total_incremental_responders']}",
        "",
        "### Policy value (self-normalized IPW on held-out treated)",
        f"- treat-none (control rate): {pv['treat_none_control_rate']}",
        f"- random action: {pv['random_action_value']}",
        f"- best single action ({pv['best_marginal']['action']}): {pv['best_marginal']['value']}",
        f"- **uplift policy: {pv['uplift_policy']['value']}**  "
        f"(gain vs random {pv['uplift_policy']['gain_vs_random_action']}, "
        f"vs treat-none {pv['uplift_policy']['gain_vs_treat_none']})",
        f"- recommended-action mix: {pv['recommended_action_distribution']}",
        "",
        "### Observed lift by action (held-out)",
        "| action | n | recovery rate | uplift vs control | z |",
        "|---|---|---|---|---|",
    ]
    for a, m in te["observed_lift_by_action"].items():
        if a == "control":
            lines.append(f"| control | {m['n']} | {m['recovery_rate']} | - | - |")
        else:
            lines.append(
                f"| {a} | {m['n']} | {m['recovery_rate']} | {m['observed_uplift_vs_control']} | {m.get('uplift_z')} |"
            )
    lines += [
        "",
        f"- reliability: {te['statistical_reliability']['note']}",
        "",
        "## Known limitations",
    ]
    lines += [f"- {x}" for x in r["limitations"]]
    return "\n".join(lines) + "\n"
