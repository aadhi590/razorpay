"""Unit tests for the generator's opt-in randomized-assignment helper."""
from __future__ import annotations

import random
from collections import Counter

import pytest

from app.scripts.generate_data import choose_randomized_action


def test_uniform_over_eligible_and_correct_propensity():
    rng = random.Random(20260902)
    eligible = ["retry", "sms_nudge", "whatsapp_nudge", "method_switch_prompt"]
    counts: Counter[str] = Counter()
    for _ in range(40_000):
        action, propensity = choose_randomized_action(rng, list(eligible))
        assert action in eligible
        assert propensity == 1 / len(eligible)
        counts[action] += 1
    for a in eligible:
        assert abs(counts[a] / 40_000 - 0.25) < 0.02


def test_propensity_tracks_shrinking_eligible_set():
    rng = random.Random(1)
    _, p3 = choose_randomized_action(rng, ["retry", "sms_nudge", "whatsapp_nudge"])
    _, p2 = choose_randomized_action(rng, ["retry", "sms_nudge"])
    assert p3 == pytest.approx(1 / 3)
    assert p2 == pytest.approx(1 / 2)


def test_empty_eligible_set_raises():
    with pytest.raises(ValueError):
        choose_randomized_action(random.Random(0), [])


def test_deterministic_for_a_given_seed():
    def run() -> list[str]:
        rng = random.Random(777)
        pool = ["retry", "sms_nudge", "whatsapp_nudge", "method_switch_prompt"]
        return [choose_randomized_action(rng, list(pool))[0] for _ in range(100)]

    assert run() == run()
