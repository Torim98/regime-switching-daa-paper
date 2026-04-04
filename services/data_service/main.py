from fastapi import FastAPI
from services.data_service.routes import router

app = FastAPI(title="Data Service", version="0.1.0")
app.include_router(router)