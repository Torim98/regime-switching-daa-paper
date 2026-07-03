"""Dashboard Service: interactive control hub + visualization.

Port 8004. Accesses data/, assets/, logs/ read-only and allows
write access to config/config.yaml (with .bak backup).
The existing pipeline (data/model/backtest services) remains
untouched.
"""
from services.warnings_config import configure_warnings
configure_warnings()

from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from services.logging_config import setup_service_logger
from services.dashboard_service.routes import router as html_router
from services.dashboard_service.hub_api import router as hub_router
from services.dashboard_service.config_api import router as config_router
from services.dashboard_service.data_adapters import router as data_router
from services.dashboard_service.websockets import router as ws_router

logger = setup_service_logger("dashboard_service")

app = FastAPI(title="Dashboard Service", version="0.1.0")

# Static files (CSS, JS)
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# Router
app.include_router(html_router)
app.include_router(hub_router)
app.include_router(config_router)
app.include_router(data_router)
app.include_router(ws_router)


@app.middleware("http")
async def add_chart_cache_headers(request: Request, call_next):
    """Browser cache for chart JSON: 5 minutes private cache.
    Combined with the server-side mtime cache in data_adapters.py, this yields
    latency well below 10 ms on rereads (no Plotly re-render, no network hit)."""
    response = await call_next(request)
    if request.url.path.startswith("/api/chart/"):
        response.headers.setdefault("Cache-Control", "private, max-age=300")
    return response


@app.on_event("startup")
def startup():
    logger.info("Dashboard Service started on :8004")
