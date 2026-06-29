"""Dashboard Service — interaktives Control Hub + Visualisierung.

Port 8004. Greift read-only auf data/, assets/, logs/ zu und erlaubt
Write-Access auf config/config.yaml (mit .bak-Backup).
Die bestehende Pipeline (data/model/backtest-Services) bleibt
unangetastet.
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
    """Browser-Cache für Chart-JSON: 5 Minuten private Cache.
    Kombiniert mit dem serverseitigen mtime-Cache in data_adapters.py ergibt
    das bei Rereads ≪ 10 ms Latenz — kein Plotly-Re-Render, kein Netzwerk-Hit."""
    response = await call_next(request)
    if request.url.path.startswith("/api/chart/"):
        response.headers.setdefault("Cache-Control", "private, max-age=300")
    return response


@app.on_event("startup")
def startup():
    logger.info("Dashboard Service started on :8004")
