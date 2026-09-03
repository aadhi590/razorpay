from app.routes.customers import router as customers_router
from app.routes.subscriptions import router as subscriptions_router
from app.routes.payments import router as payments_router
from app.routes.recovery_events import router as recovery_events_router
from app.routes.interventions import router as interventions_router
from app.routes.outcomes import router as outcomes_router
from app.routes.experiments import router as experiments_router
from app.routes.agent_events import router as agent_events_router
from app.routes.audit_logs import router as audit_logs_router
from app.routes.orchestrator import router as orchestrator_router
from app.routes.analytics import router as analytics_router
from app.routes.ml import router as ml_router
from app.routes.uplift import router as uplift_router
from app.routes.agent import router as agent_router
from app.routes.webhooks import router as webhooks_router
from app.routes.razorpay_inspect import router as razorpay_inspect_router
from app.routes.voice import router as voice_router

__all__ = [
    "customers_router",
    "subscriptions_router",
    "payments_router",
    "recovery_events_router",
    "interventions_router",
    "outcomes_router",
    "experiments_router",
    "agent_events_router",
    "audit_logs_router",
    "orchestrator_router",
    "analytics_router",
    "ml_router",
    "uplift_router",
    "agent_router",
    "webhooks_router",
    "razorpay_inspect_router",
    "voice_router",
]
