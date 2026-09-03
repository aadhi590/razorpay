from __future__ import annotations

from pydantic import BaseModel


class ActionUpliftSchema(BaseModel):
    action: str
    treatment_probability: float          # mu_a(X) = P(recover | X, action), calibrated
    uplift: float                         # mu_a(X) - mu_0(X)
    cost_paise: int
    incremental_expected_revenue_paise: float  # uplift * payment_amount
    net_incremental_value_paise: float         # incremental revenue - action cost
    rank: int                            # 1 = best net incremental value


class UpliftScoresResponse(BaseModel):
    recovery_event_id: int
    available: bool
    model_version: str | None
    as_of: str
    baseline_probability: float | None    # mu_0(X) = P(recover | no intervention)
    amount_paise: int | None
    untried_actions: list[str]
    actions: list[ActionUpliftSchema]     # ranked best-first by net incremental value
    recommended_action: str | None        # best action with positive net incremental value; None if none
    note: str


class UpliftModelInfoResponse(BaseModel):
    available: bool
    model_name: str | None = None
    model_version: str | None = None
    learner_type: str | None = None
    base_algorithm: str | None = None
    feature_version: str | None = None
    created_at: str | None = None
    champion_reason: str | None = None
    dataset: dict | None = None
    propensity_diagnostics: dict | None = None
    test_evaluation: dict | None = None
    limitations: list[str] | None = None
    synthetic_benchmark: bool = True
    detail: str | None = None
