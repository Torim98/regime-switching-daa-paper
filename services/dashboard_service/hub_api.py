"""Control-Hub-Proxy: Ruft die FastAPI-Endpoints der drei Pipeline-Services
(data/model/backtest) per httpx auf, damit die UI Pipeline-Stages auslösen kann.

Alle Proxy-Calls haben lange Timeouts, weil z.B. /models/train-all bei
Walk-Forward sehr lange läuft. Die UI zeigt währenddessen einen Spinner
und das Live-Log (separater WebSocket).
"""
from typing import Any, Dict
import os
import httpx
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/hub", tags=["hub"])

# Service-URLs — in Docker-Compose via Service-Namen erreichbar,
# lokal via localhost fallback.
_SERVICES = {
    "data":     os.environ.get("DATA_SERVICE_URL",     "http://data-service:8001"),
    "model":    os.environ.get("MODEL_SERVICE_URL",    "http://model-service:8002"),
    "backtest": os.environ.get("BACKTEST_SERVICE_URL", "http://backtest-service:8003"),
}

# Längere Timeouts: WF + Train-All können > 1 h dauern
_TIMEOUT = httpx.Timeout(connect=10.0, read=28800.0, write=30.0, pool=10.0)


# ---------------------------------------------------------------------------
# Katalog aller verfügbaren Endpoints — wird vom Frontend gelesen,
# um dynamisch Buttons/Forms zu rendern.
# ---------------------------------------------------------------------------

_CATALOG = [
    {
        "service": "data", "port": 8001,
        "endpoints": [
            {
                "id": "data.ingest", "method": "POST", "path": "/data/ingest",
                "label": "Daten-Ingestion starten",
                "description": "yfinance → Preprocessing → Feature-Engineering → EDA → Plots.",
                "params": [], "danger": False,
            },
            {
                "id": "data.features", "method": "GET", "path": "/data/features",
                "label": "Feature-DataFrame abrufen",
                "description": "Voller Feature-Engineered-DF als JSON (split-orient).",
                "params": [], "danger": False,
            },
            {
                "id": "data.label_analysis", "method": "POST", "path": "/data/label-analysis",
                "label": "Label-Analyse",
                "description": "Konkordanz + Switch-Stats der Regime-Labeler (MSM, HMM, PagSoss, P2T, LundeT, NBER).",
                "params": [], "danger": False,
            },
        ],
    },
    {
        "service": "model", "port": 8002,
        "endpoints": [
            {
                "id": "model.status", "method": "GET", "path": "/models/status",
                "label": "Modell-Persistenz-Status",
                "description": "Welche Modelle liegen in models/ auf der Platte?",
                "params": [], "danger": False,
            },
            {
                "id": "model.train_one", "method": "POST", "path": "/models/train/{model_name}",
                "label": "Einzelnes Modell trainieren",
                "description": "Nur bei walk_forward.enabled=false. Sonst HTTP 400.",
                "params": [
                    {"name": "model_name", "in": "path", "type": "select",
                     "options": ["msm", "hmm", "lstm", "transformer"]},
                ],
                "danger": True,
            },
            {
                "id": "model.train_all", "method": "POST", "path": "/models/train-all",
                "label": "Alle Modelle trainieren",
                "description": "Single-Split oder Walk-Forward-Engine (je nach Config).",
                "params": [], "danger": True,
            },
            {
                "id": "model.optimize_one", "method": "POST", "path": "/models/optimize/{model_name}",
                "label": "Optuna-HPO für ein Modell",
                "description": "Erfordert walk_forward.enabled=true. Persistiert in optuna_studies.db.",
                "params": [
                    {"name": "model_name", "in": "path", "type": "select",
                     "options": ["MSM", "HMM", "LSTM", "Transformer"]},
                    {"name": "n_trials",        "in": "query", "type": "int", "default": 50},
                    {"name": "every_nth_fold",  "in": "query", "type": "int", "default": 2},
                ],
                "danger": True,
            },
            {
                "id": "model.optimize_all", "method": "POST", "path": "/models/optimize-all",
                "label": "Optuna-HPO für alle Modelle",
                "description": "Sequentiell MSM → HMM → LSTM → Transformer.",
                "params": [
                    {"name": "n_trials",        "in": "query", "type": "int", "default": 50},
                    {"name": "every_nth_fold",  "in": "query", "type": "int", "default": 2},
                ],
                "danger": True,
            },
        ],
    },
    {
        "service": "backtest", "port": 8003,
        "endpoints": [
            {
                "id": "backtest.run", "method": "POST", "path": "/backtest/run",
                "label": "Backtest + SORR ausführen",
                "description": "Equity, TxCosts, annualisierte Metriken, Krisen-Perf, Rolling-Sharpe, DD.",
                "params": [], "danger": False,
            },
            {
                "id": "backtest.evaluate", "method": "POST", "path": "/backtest/evaluate",
                "label": "Evaluation + MCS",
                "description": "Block-Bootstrap-MCS + Klassifikation + ROC/PR + Churning + H1/H2-Tests.",
                "params": [], "danger": False,
            },
            {
                "id": "backtest.report", "method": "POST", "path": "/backtest/report",
                "label": "statistics.md generieren",
                "description": "Finales Markdown-Report unter docs/statistics.md.",
                "params": [], "danger": False,
            },
            {
                "id": "backtest.results", "method": "GET", "path": "/backtest/results",
                "label": "Evaluation-Tabelle abrufen",
                "description": "Evaluation-MD als JSON. Setzt /backtest/evaluate voraus.",
                "params": [], "danger": False,
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Routen
# ---------------------------------------------------------------------------

@router.get("/catalog")
def catalog() -> Dict[str, Any]:
    """Liefert den Endpoint-Katalog zum dynamischen UI-Rendering."""
    return {"services": _CATALOG}


@router.get("/health")
async def health() -> Dict[str, Any]:
    """Ping-Check auf alle drei Services (OpenAPI-JSON als Marker)."""
    out: Dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for svc, base in _SERVICES.items():
            try:
                r = await client.get(f"{base}/openapi.json")
                out[svc] = {"url": base, "up": r.status_code == 200, "status": r.status_code}
            except Exception as e:
                out[svc] = {"url": base, "up": False, "error": str(e)}
    return out


@router.post("/call")
async def hub_call(
    service: str = Query(..., pattern="^(data|model|backtest)$"),
    path: str = Query(..., description="Pfad wie /data/ingest"),
    method: str = Query("POST", pattern="^(GET|POST)$"),
    query: str = Query("", description="JSON-Query-Params, optional"),
):
    """Generischer Proxy: UI gibt Service + Pfad + Methode + optional Query-JSON an.

    Beispiele (vom Frontend):
      - POST /api/hub/call?service=data&path=/data/ingest&method=POST
      - POST /api/hub/call?service=model&path=/models/optimize/MSM&method=POST&query={"n_trials":20}
    """
    import json
    base = _SERVICES.get(service)
    if base is None:
        raise HTTPException(400, f"Unbekannter Service: {service}")

    url = base.rstrip("/") + path
    try:
        params = json.loads(query) if query else {}
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Query-JSON ungültig: {e}")

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        try:
            if method == "GET":
                r = await client.get(url, params=params)
            else:
                r = await client.post(url, params=params)
        except httpx.ConnectError as e:
            raise HTTPException(502, f"Verbindung zu {service}-Service fehlgeschlagen: {e}")
        except httpx.ReadTimeout as e:
            raise HTTPException(504, f"Timeout bei {url}: {e}")

    try:
        body = r.json()
    except ValueError:
        body = {"text": r.text}

    return {"status_code": r.status_code, "ok": r.is_success, "body": body}
