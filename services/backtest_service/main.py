# Warning suppression MUST happen before all other imports,
# so that TF_CPP_MIN_LOG_LEVEL takes effect.
from services.warnings_config import configure_warnings
configure_warnings()

from fastapi import FastAPI
from services.logging_config import setup_service_logger
from services.backtest_service.routes import router

logger = setup_service_logger("backtest_service")

app = FastAPI(title="Backtest Service", version="0.1.0")
app.include_router(router)

@app.on_event("startup")
def startup():
    logger.info("Backtest Service started")