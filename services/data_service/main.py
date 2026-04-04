from fastapi import FastAPI
from services.logging_config import setup_service_logger
from services.data_service.routes import router

logger = setup_service_logger("data_service")

app = FastAPI(title="Data Service", version="0.1.0")
app.include_router(router)

@app.on_event("startup")
def startup():
    logger.info("Data Service started")