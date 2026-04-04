from fastapi import FastAPI
from services.model_service.routes import router

app = FastAPI(title="Model Service", version="0.1.0")
app.include_router(router)