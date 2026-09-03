from __future__ import annotations

import pytest
from sqlalchemy import text

from app.database import engine


@pytest.fixture(scope="session")
def _uplift_counts() -> dict:
    with engine.connect() as c:
        return {
            "control": c.execute(
                text("select count(*) from recovery_events where is_control")
            ).scalar()
            or 0,
            "randomized": c.execute(
                text(
                    "select count(*) from agent_events "
                    "where event_type='intervention_decision' "
                    "and (input_context -> 'assignment' ->> 'chosen_action') is not null"
                )
            ).scalar()
            or 0,
        }


@pytest.fixture
def require_uplift_data(_uplift_counts):
    if _uplift_counts["control"] < 200 or _uplift_counts["randomized"] < 500:
        pytest.skip(
            "needs randomized-assignment data: python -m app.scripts.generate_data "
            "--reset --customers 6000 --seed 42 --randomized-assignment"
        )


@pytest.fixture(scope="session")
def uplift_dataset(_uplift_counts):
    """Built once per session -- the point-in-time feature SQL is expensive."""
    if _uplift_counts["control"] < 200 or _uplift_counts["randomized"] < 500:
        pytest.skip("no randomized-assignment data")
    from app.ml.uplift.dataset.builder import build_uplift_dataset

    with engine.connect() as c:
        return build_uplift_dataset(c)


@pytest.fixture(scope="session")
def uplift_split(uplift_dataset):
    from app.ml.uplift.config import SPLIT_FRACTIONS
    from app.ml.training.splitting import grouped_chronological_split

    split = grouped_chronological_split(uplift_dataset.frame, fractions=SPLIT_FRACTIONS)
    tr, va, te = split.frames(uplift_dataset.frame)
    return {"split": split, "train": tr, "val": va, "test": te}


@pytest.fixture(scope="session")
def fitted_t_learner(uplift_split):
    from app.ml.uplift.estimators.learners import TLearner

    return TLearner(base="logreg").fit(uplift_split["train"])


@pytest.fixture(scope="session")
def uplift_model(_uplift_counts, tmp_path_factory):
    """A loaded UpliftModel: the committed artifact if present, else a fast one
    trained into a temp dir."""
    from app.ml.uplift.inference.predictor import UpliftModel
    from app.ml.uplift.models.artifact import UpliftModelUnavailable

    try:
        return UpliftModel.load()
    except UpliftModelUnavailable:
        pass
    if _uplift_counts["control"] < 200:
        pytest.skip("no artifact and no randomized-assignment data")
    from app.ml.uplift.training import run_uplift_training

    out = tmp_path_factory.mktemp("uplift_artifacts")
    with engine.connect() as c:
        run = run_uplift_training(c, out_dir=out, fast=True)
    return UpliftModel.load(run.artifact_path)
