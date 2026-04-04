from fastapi import FastAPI
from services.logging_config import setup_service_logger
from services.model_service.routes import router

logger = setup_service_logger("model_service")

app = FastAPI(title="Model Service", version="0.1.0")
app.include_router(router)

@app.on_event("startup")
def startup():
    logger.info("Model Service started")
