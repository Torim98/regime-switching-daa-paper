"""HTML-Seiten-Routes des Dashboards.

Jede Route rendert ein Jinja-Template. Die eigentlichen Daten werden
clientseitig via HTMX/fetch gegen /api/... geladen (siehe data_adapters.py,
hub_api.py, config_api.py).
"""
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from config.config_loader import PipelineConfig

router = APIRouter(tags=["html"])

_templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))


def _nav_context(active: str) -> dict:
    """Gemeinsamer Kontext für alle Seiten (Nav-Highlight, Build-Info)."""
    cfg = PipelineConfig()
    return {
        "active": active,
        "build_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "end_date": cfg.data.end_date,
        "end_date_frozen": getattr(cfg.data, "end_date_is_frozen", False),
        "walk_forward_enabled": cfg.walk_forward.enabled,
        "fast_mode_enabled": cfg.fast_mode.enabled,
    }


@router.get("/", response_class=HTMLResponse)
def page_index(request: Request):
    return templates.TemplateResponse(request, "index.html", _nav_context("index"))


@router.get("/eda", response_class=HTMLResponse)
def page_eda(request: Request):
    return templates.TemplateResponse(request, "eda.html", _nav_context("eda"))


@router.get("/models", response_class=HTMLResponse)
def page_models(request: Request):
    return templates.TemplateResponse(request, "models.html", _nav_context("models"))


@router.get("/backtest", response_class=HTMLResponse)
def page_backtest(request: Request):
    return templates.TemplateResponse(request, "backtest.html", _nav_context("backtest"))


@router.get("/evaluation", response_class=HTMLResponse)
def page_evaluation(request: Request):
    return templates.TemplateResponse(request, "evaluation.html", _nav_context("evaluation"))


@router.get("/hub", response_class=HTMLResponse)
def page_hub(request: Request):
    return templates.TemplateResponse(request, "hub.html", _nav_context("hub"))


@router.get("/config", response_class=HTMLResponse)
def page_config(request: Request):
    return templates.TemplateResponse(request, "config_editor.html", _nav_context("config"))


@router.get("/logs", response_class=HTMLResponse)
def page_logs(request: Request):
    return templates.TemplateResponse(request, "logs.html", _nav_context("logs"))
