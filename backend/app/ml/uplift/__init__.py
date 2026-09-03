"""Uplift / causal recovery intelligence.

Estimates *incremental* recovery caused by an intervention relative to no
intervention -- ``E[Y(a) - Y(0) | X]`` -- rather than the raw predictive
``P(Y=1 | X, a)`` that ``app/ml/`` produces. Both layers coexist.

Sub-packages
------------
* ``dataset``    -- point-in-time uplift dataset (one row per eligible decision)
* ``features``   -- re-export of the shared point-in-time feature contract
* ``validation`` -- propensity / overlap / positivity + leakage / split integrity
* ``estimators`` -- S-learner and T-learner (sklearn-compatible, no new deps)
* ``models``     -- the versioned ``UpliftArtifact``
* ``evaluation`` -- Qini / AUUC / uplift@k / policy value / observed lift
* ``inference``  -- ``UpliftModel``: baseline + per-action uplift + economics
* ``registry``   -- load/save the latest uplift artifact
* ``training``   -- the offline training driver (CLI only)
"""
