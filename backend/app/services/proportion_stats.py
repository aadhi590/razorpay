"""Confidence interval for the difference of two independent proportions.

Pure stdlib ``math`` -- no new dependency (the same minimal-dependency posture
used for the Gemini and Razorpay adapters).

Method: **Newcombe's method** (a.k.a. the "hybrid score" / MOVER interval),
which builds the interval for ``p_treated - p_control`` from the two individual
**Wilson score** intervals. Newcombe (1998) showed this has good coverage even
for small samples and proportions near 0 or 1 -- exactly the regime this metric
lives in (small control groups, low recovery rates). It is not a normal /Wald
approximation and it is not a p-value.

Reference: Newcombe RG, "Interval estimation for the difference between
independent proportions: comparison of eleven methods", Statistics in Medicine
1998; 17:873-890 (method 10).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# z for a two-sided 95% interval
_Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Interval:
    low: float
    high: float

    def as_list(self) -> list[float]:
        return [round(self.low, 6), round(self.high, 6)]

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0


def wilson_interval(successes: int, n: int, z: float = _Z_95) -> Interval:
    """Wilson score interval for a single proportion ``successes / n``."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))) / denom
    return Interval(max(0.0, centre - half), min(1.0, centre + half))


def newcombe_difference_interval(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    z: float = _Z_95,
) -> Interval:
    """95% CI (by default) for ``p_a - p_b`` via Newcombe's hybrid-score method.

    ``a`` = treated group, ``b`` = control group, so a positive interval means
    treatment recovered proportionally more than control.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError("both groups must be non-empty")
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    ci_a = wilson_interval(successes_a, n_a, z)
    ci_b = wilson_interval(successes_b, n_b, z)
    diff = p_a - p_b
    low = diff - math.sqrt((p_a - ci_a.low) ** 2 + (ci_b.high - p_b) ** 2)
    high = diff + math.sqrt((ci_a.high - p_a) ** 2 + (p_b - ci_b.low) ** 2)
    return Interval(low, high)
