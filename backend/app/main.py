from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text  # type: ignore[reportMissingImports]

from app.config import settings
from app.database import engine
from app.services.recovery_scheduler import scheduler
from app.routes import (
    customers_router,
    subscriptions_router,
    payments_router,
    recovery_events_router,
    interventions_router,
    outcomes_router,
    experiments_router,
    agent_events_router,
    audit_logs_router,
    orchestrator_router,
    analytics_router,
    ml_router,
    uplift_router,
    agent_router,
    webhooks_router,
    razorpay_inspect_router,
    voice_router,
    scheduler_router,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # The Recovery Scheduler is a daemon thread. start() is a no-op unless
    # SCHEDULER_ENABLED is true, so importing/serving the app is unchanged when
    # the feature is off.
    scheduler.reload_config()
    scheduler.start()
    try:
        yield
    finally:
        scheduler.stop()


app = FastAPI(
    title="Razorpay AI Recovery Orchestrator",
    version="0.1.0",
    lifespan=lifespan,
)

# Browser access for the read-only dashboard frontend. Only the exact origins
# in CORS_ALLOW_ORIGINS are permitted; credentials are not allowed (the API uses
# no cookies / auth headers). When the setting is empty, no CORS layer is added
# and behaviour is identical to before the frontend existed.
_cors_origins = settings.cors_allow_origins_list
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

app.include_router(customers_router)
app.include_router(subscriptions_router)
app.include_router(payments_router)
app.include_router(recovery_events_router)
app.include_router(interventions_router)
app.include_router(outcomes_router)
app.include_router(experiments_router)
app.include_router(agent_events_router)
app.include_router(audit_logs_router)
app.include_router(orchestrator_router)
app.include_router(analytics_router)
app.include_router(ml_router)
app.include_router(uplift_router)
app.include_router(agent_router)
app.include_router(webhooks_router)
app.include_router(razorpay_inspect_router)
app.include_router(voice_router)
app.include_router(scheduler_router)


@app.get("/")
def root():
    return {
        "message": "Razorpay AI Recovery Orchestrator API is running"
    }


@app.get("/health")
def health_check():
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "database": result.scalar(),
        }