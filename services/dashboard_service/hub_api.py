"""Control hub proxy: calls the FastAPI endpoints of the three pipeline
services (data/model/backtest) via httpx so that the UI can trigger pipeline stages.

All proxy calls have long timeouts because e.g. /models/train-all takes
very long in walk-forward mode. Meanwhile, the UI shows a spinner
and the live log (separate WebSocket).
"""
from typing import Any, Dict
import os
import httpx
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/hub", tags=["hub"])

# Service URLs: reachable via service names in Docker Compose,
# locally via localhost fallback.
_SERVICES = {
    "data":     os.environ.get("DATA_SERVICE_URL",     "http://data-service:8001"),
    "model":    os.environ.get("MODEL_SERVICE_URL",    "http://model-service:8002"),
    "backtest": os.environ.get("BACKTEST_SERVICE_URL", "http://backtest-service:8003"),
}

# Longer timeouts: WF + train-all can take > 1 h
_TIMEOUT = httpx.Timeout(connect=10.0, read=28800.0, write=30.0, pool=10.0)


# ---------------------------------------------------------------------------
# Catalog of all available endpoints; read by the frontend to render
# buttons/forms dynamically.
# ---------------------------------------------------------------------------

_CATALOG = [
    {
        "service": "data", "port": 8001,
        "endpoints": [
            {
                "id": "data.ingest", "method": "POST", "path": "/data/ingest",
                "label": "Start data ingestion",
                "description": "yfinance → preprocessing → feature engineering → EDA → plots.",
                "params": [], "danger": False,
            },
            {
                "id": "data.features", "method": "GET", "path": "/data/features",
                "label": "Fetch feature DataFrame",
                "description": "Full feature-engineered DF as JSON (split orient).",
                "params": [], "danger": False,
            },
            {
                "id": "data.label_analysis", "method": "POST", "path": "/data/label-analysis",
                "label": "Label analysis",
                "description": "Concordance + switch stats of the regime labelers (MSM, HMM, PagSoss, P2T, LundeT, NBER).",
                "params": [], "danger": False,
            },
        ],
    },
    {
        "service": "model", "port": 8002,
        "endpoints": [
            {
                "id": "model.status", "method": "GET", "path": "/models/status",
                "label": "Model persistence status",
                "description": "Which models are stored on disk in models/?",
                "params": [], "danger": False,
            },
            {
                "id": "model.train_one", "method": "POST", "path": "/models/train/{model_name}",
                "label": "Train a single model",
                "description": "Only with walk_forward.enabled=false. Otherwise HTTP 400.",
                "params": [
                    {"name": "model_name", "in": "path", "type": "select",
                     "options": ["msm", "hmm", "hmm_uni", "lstm", "transformer"]},
                ],
                "danger": True,
            },
            {
                "id": "model.train_all", "method": "POST", "path": "/models/train-all",
                "label": "Train all models",
                "description": "Single split or walk-forward engine (depending on the config).",
                "params": [], "danger": True,
            },
            {
                "id": "model.optimize_one", "method": "POST", "path": "/models/optimize/{model_name}",
                "label": "Optuna HPO for one model",
                "description": "Requires walk_forward.enabled=true. Sampler (grid/TPE), trial budget, objective metric and tune_until come from config.yaml (optimization.*). Persisted in optuna_studies.db.",
                "params": [
                    {"name": "model_name", "in": "path", "type": "select",
                     "options": ["MSM", "HMM", "HMM_Uni", "LSTM", "Transformer"]},
                ],
                "danger": True,
                # Stop button: graceful cancel, resumes from the same study on re-run.
                "stop_path": "/models/optimize-stop",
            },
            {
                "id": "model.optimize_all", "method": "POST", "path": "/models/optimize-all",
                "label": "Optuna HPO for all models",
                "description": "Sequentially MSM → HMM → HMM_Uni → LSTM → Transformer. Sampler (grid/TPE), trial budget, objective metric and tune_until come from config.yaml (optimization.*). Persisted in optuna_studies.db.",
                "params": [],
                "danger": True,
                # Stop button: graceful cancel, resumes from the same study on re-run.
                "stop_path": "/models/optimize-stop",
            },
            {
                "id": "model.hpo_analysis", "method": "POST", "path": "/models/hpo-analysis",
                "label": "HPO analysis reports",
                "description": "Post-HPO reports from the Optuna studies. scope=cheap: convergence + edge-of-range review + objective sensitivity (seconds). scope=full: also DSR, PBO and multi-seed reeval (re-trains DL on the GPU, slow). Writes Markdown assets shown in statistics.md and the Models page.",
                "params": [
                    {"name": "scope", "in": "query", "type": "select",
                     "options": ["cheap", "full"]},
                ],
                "danger": False,
            },
        ],
    },
    {
        "service": "backtest", "port": 8003,
        "endpoints": [
            {
                "id": "backtest.run", "method": "POST", "path": "/backtest/run",
                "label": "Run backtest + SORR",
                "description": "Equity, tx costs, annualized metrics, crisis performance, rolling Sharpe, DD.",
                "params": [], "danger": False,
            },
            {
                "id": "backtest.evaluate", "method": "POST", "path": "/backtest/evaluate",
                "label": "Evaluation + MCS",
                "description": "Bootstrap MCS (configurable: block or stationary) + classification + ROC/PR + churning + H1/H2 tests.",
                "params": [], "danger": False,
            },
            {
                "id": "backtest.bootstrap-robustness", "method": "POST",
                "path": "/backtest/bootstrap-robustness",
                "label": "Bootstrap robustness (block vs. stationary)",
                "description": "Issue #7: re-runs the MCS with the block and the stationary bootstrap (same seed, no re-training) and writes assets/bootstrap_robustness.md comparing depletion rate (Wilson CI) and median terminal capital. Requires /backtest/run. Optional n_paths overrides the config for a quick check.",
                "params": [
                    {"name": "n_paths", "in": "query", "type": "number"},
                ],
                "danger": False,
            },
            {
                "id": "backtest.report", "method": "POST", "path": "/backtest/report",
                "label": "Generate statistics.md",
                "description": "Final Markdown report at docs/statistics.md.",
                "params": [], "danger": False,
            },
            {
                "id": "backtest.results", "method": "GET", "path": "/backtest/results",
                "label": "Fetch evaluation table",
                "description": "Evaluation MD as JSON. Requires /backtest/evaluate.",
                "params": [], "danger": False,
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/catalog")
def catalog() -> Dict[str, Any]:
    """Returns the endpoint catalog for dynamic UI rendering."""
    return {"services": _CATALOG}


@router.get("/health")
async def health() -> Dict[str, Any]:
    """Ping check on all three services (OpenAPI JSON as marker)."""
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
    path: str = Query(..., description="Path such as /data/ingest"),
    method: str = Query("POST", pattern="^(GET|POST)$"),
    query: str = Query("", description="JSON query params, optional"),
):
    """Generic proxy: the UI provides service + path + method + optional query JSON.

    Examples (from the frontend):
      - POST /api/hub/call?service=data&path=/data/ingest&method=POST
      - POST /api/hub/call?service=model&path=/models/optimize/MSM&method=POST
    """
    import json
    base = _SERVICES.get(service)
    if base is None:
        raise HTTPException(400, f"Unknown service: {service}")

    url = base.rstrip("/") + path
    try:
        params = json.loads(query) if query else {}
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid query JSON: {e}")

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        try:
            if method == "GET":
                r = await client.get(url, params=params)
            else:
                r = await client.post(url, params=params)
        except httpx.ConnectError as e:
            raise HTTPException(502, f"Connection to the {service} service failed: {e}")
        except httpx.ReadTimeout as e:
            raise HTTPException(504, f"Timeout at {url}: {e}")

    try:
        body = r.json()
    except ValueError:
        body = {"text": r.text}

    return {"status_code": r.status_code, "ok": r.is_success, "body": body}
