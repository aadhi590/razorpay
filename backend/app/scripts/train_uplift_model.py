"""CLI: train the uplift / causal-recovery model from the current database.

    python -m app.scripts.train_uplift_model
    python -m app.scripts.train_uplift_model --fast        # quick, for tests/CI
    python -m app.scripts.train_uplift_model --out artifacts/ml

Prerequisite: a database populated with randomized-assignment data --
    python -m app.scripts.generate_data --reset --customers 6000 --seed 42 \
        --randomized-assignment

Writes:
    artifacts/ml/uplift_recovery_uplift_v1.joblib
    artifacts/ml/uplift_recovery_latest.joblib
    artifacts/ml/uplift_recovery_uplift_v1.report.{json,md}

Training NEVER runs implicitly -- only via this command. The predictive
recovery-response model (app/ml/) is untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from app.database import engine
from app.ml.uplift.registry import reset_default_uplift_artifact
from app.ml.uplift.training import run_uplift_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the uplift recovery model.")
    parser.add_argument("--out", type=str, default=None, help="Output directory for artifacts.")
    parser.add_argument("--fast", action="store_true", help="Fast config -- for CI/tests.")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else None
    with engine.connect() as conn:
        run = run_uplift_training(conn, out_dir=out_dir, fast=args.fast)

    reset_default_uplift_artifact()
    r = run.report
    te = r["test_evaluation"]
    pv = te["policy_value"]["uplift_policy"]
    print("\n=== uplift training complete (SYNTHETIC BENCHMARK) ===")
    print(f"artifact : {run.artifact_path}")
    print(f"report   : {run.report_path}")
    print(f"dataset  : {r['dataset']['n_rows']} rows "
          f"({r['dataset']['n_control']} control / {r['dataset']['n_treatment']} treatment), "
          f"{r['dataset']['n_positive']} positive")
    print(f"champion : {r['champion']}  -- {r['champion_reason']}")
    print(f"test Qini: {te['qini']['qini_coefficient']}  "
          f"CI {te['qini']['bootstrap_ci'].get('ci_95')}")
    print(f"policy   : uplift={pv['value']}  "
          f"gain vs random={pv['gain_vs_random_action']}  "
          f"vs treat-none={pv['gain_vs_treat_none']}")
    print(f"reliable : {te['statistical_reliability']['reliable']} "
          f"-- {te['statistical_reliability']['note']}")


if __name__ == "__main__":
    main()
