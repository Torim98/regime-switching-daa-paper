from fastapi import FastAPI
from services.backtest_service.routes import router

app = FastAPI(title="Backtest Service", version="0.1.0")
app.include_router(router)