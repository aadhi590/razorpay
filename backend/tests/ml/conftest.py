"""Shared, expensive ML fixtures.

The point-in-time feature SQL is heavy (~2-3 min on a 6k-customer dataset).
These build it **once per test session** and hand out the result. Every
consumer either only reads the frame or takes a ``.copy()`` before mutating
(the data-quality / leakage tests), so a shared instance is safe -- no test
mutates the session frame in place.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

from app.database import engine


def _has_training_data() -> bool:
    with engine.connect() as c:
        return (
            c.execute(text("select count(*) from interventions")).scalar() or 0
        ) >= 100


@pytest.fixture(scope="session")
def training_dataset():
    """``build_training_dataset`` run once for the whole session."""
    if not _has_training_data():
        pytest.skip(
            "needs full-pipeline data: python -m app.scripts.generate_data "
            "--reset --customers 1000 --seed 42"
        )
    from app.ml.datasets.builder import build_training_dataset

    with engine.connect() as c:
        return build_training_dataset(c)


@pytest.fixture(scope="session")
def training_frame(training_dataset):
    """The session training frame. Treat as read-only; ``.copy()`` before
    mutating."""
    return training_dataset.frame


@pytest.fixture(scope="session")
def fast_training_run(tmp_path_factory):
    """``run_training(fast=True)`` executed once; both artifact tests share it."""
    if not _has_training_data():
        pytest.skip("no training data")
    from app.ml.training.train import run_training

    out = tmp_path_factory.mktemp("ml_fast_training")
    with engine.connect() as c:
        return run_training(c, out_dir=out, fast=True)
