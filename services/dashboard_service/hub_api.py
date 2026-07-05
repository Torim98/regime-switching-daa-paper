"""Control hub proxy: calls the FastAPI endpoints of the three pipeline
services (data/model/backtest) via httpx so that the UI can trigger pipeline stages.

All proxy calls have long timeouts because e.g. /models/train-all takes
very long in walk-forward mode. Meanwhile, the UI shows a spinner
and the live log (separate WebSocket).
"""
from typing import Any, Dict, List
import os
import time
import threading
import logging
from pathlib import Path

import httpx
from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel

from config.config_loader import PipelineConfig

router = APIRouter(prefix="/api/hub", tags=["hub"])

logger = logging.getLogger("dashboard_service")

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
                "description": "Which models are trained? (models/ files, or the walk-forward cache when walk_forward is on)",
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
            {
                "id": "model.seed_sensitivity", "method": "POST", "path": "/models/seed-sensitivity",
                "label": "Seed sensitivity (retraining stability)",
                "description": "Re-runs every model on the production config and the full walk-forward fold set, varying only its random source (EM init for HMM/HMM_Uni, global RNG for LSTM/Transformer; MSM is the zero-variance control). Reports mean/std/CV of the OOS metrics to quantify the jumping DL performance between retrainings. Re-trains DL on the GPU per seed (slow). Writes assets/seed_sensitivity.md, shown on the Models page. models='dl' restricts to LSTM+Transformer for a faster check.",
                "params": [
                    {"name": "seeds", "in": "query", "type": "int", "default": 5},
                    {"name": "models", "in": "query", "type": "select",
                     "options": ["all", "dl"]},
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
                    {"name": "n_paths", "in": "query", "type": "int", "default": 10000},
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


# ===========================================================================
# Full-pipeline orchestrator
# ---------------------------------------------------------------------------
# A single "run everything once" entry point that walks the canonical endpoint
# sequence (data -> model -> backtest) so that ONE run produces every asset the
# paper needs. Optional steps (HPO, seed sensitivity, label analysis, bootstrap
# robustness, the jupyter/ figure notebooks) are toggled per run, and previous
# pipeline artifacts (e.g. the walk-forward cache) can be deleted beforehand.
#
# Runs in a background thread with a polled status endpoint: a full walk-forward
# run takes hours, so the UI must not depend on one blocking request. One uvicorn
# worker per service, so a module-level job state is shared (same pattern as the
# HPO cancel Event in the model service).
# ===========================================================================

# Canonical, ordered step catalog. `default` marks the steps that belong in the
# minimal "all assets" run; `core` steps are the ones downstream steps depend on
# (unchecking them is allowed but flagged in the UI). Params mirror the query
# parameters of the underlying endpoints.
_PIPELINE_STEPS: List[Dict[str, Any]] = [
    {
        "key": "ingest", "label": "1. Data ingestion",
        "service": "data", "method": "POST", "path": "/data/ingest",
        "default": True, "core": True,
        "desc": "yfinance -> preprocessing -> feature engineering -> EDA + quality + feature plots.",
        "params": [],
    },
    {
        "key": "optimize", "label": "2. HPO (optimize-all)",
        "service": "model", "method": "POST", "path": "/models/optimize-all",
        "default": False, "core": False, "stop_path": "/models/optimize-stop",
        "desc": "Optuna HPO for all 5 models (walk-forward inner CV). Slow (GPU). Resumes from optuna_studies.db.",
        "params": [],
    },
    {
        "key": "hpo_analysis", "label": "3. HPO analysis reports",
        "service": "model", "method": "POST", "path": "/models/hpo-analysis",
        "default": False, "core": False,
        "desc": "Post-HPO reports (convergence, edge-of-range, objective sensitivity; scope=full adds DSR/PBO/multi-seed).",
        "params": [{"name": "scope", "type": "select", "options": ["cheap", "full"], "default": "cheap"}],
    },
    {
        "key": "train", "label": "4. Train all models",
        "service": "model", "method": "POST", "path": "/models/train-all",
        "default": True, "core": True,
        "desc": "Walk-forward (or single-split) training of MSM/HMM/HMM_Uni/LSTM/Transformer; writes test_df + regime plots + WF schema.",
        "params": [],
    },
    {
        "key": "seed_sensitivity", "label": "5. Seed sensitivity",
        "service": "model", "method": "POST", "path": "/models/seed-sensitivity",
        "default": False, "core": False,
        "desc": "Retraining stability of the production config (GPU, slow). Writes seed_sensitivity.md.",
        "params": [
            {"name": "seeds", "type": "int", "default": 5},
            {"name": "models", "type": "select", "options": ["all", "dl"], "default": "all"},
        ],
    },
    {
        "key": "label_analysis", "label": "6. Label analysis",
        "service": "data", "method": "POST", "path": "/data/label-analysis",
        "default": True, "core": False,
        "desc": "Concordance of alternative regime labelers vs. MSM/HMM. Needs trained models (test_df).",
        "params": [],
    },
    {
        "key": "backtest_run", "label": "7. Backtest + SORR",
        "service": "backtest", "method": "POST", "path": "/backtest/run",
        "default": True, "core": True,
        "desc": "Equity, transaction costs, annualized metrics, crisis performance, SORR scenarios.",
        "params": [],
    },
    {
        "key": "bootstrap_robustness", "label": "8. Bootstrap robustness",
        "service": "backtest", "method": "POST", "path": "/backtest/bootstrap-robustness",
        "default": False, "core": False,
        "desc": "Block vs. stationary bootstrap comparison (same seed, no retraining). Writes bootstrap_robustness.md.",
        "params": [{"name": "n_paths", "type": "int", "default": 10000}],
    },
    {
        "key": "evaluate", "label": "9. Evaluation + MCS",
        "service": "backtest", "method": "POST", "path": "/backtest/evaluate",
        "default": True, "core": True,
        "desc": "Monte-Carlo, classification vs. NBER, extended evaluation, H1/H2 tests; also writes statistics.md.",
        "params": [],
    },
    {
        "key": "notebooks", "label": "10. Paper figure notebooks",
        "service": "_local", "method": "", "path": "",
        "default": False, "core": False,
        "desc": "Executes jupyter/*.ipynb (standalone paper figures). Needs nbclient in the dashboard image.",
        "params": [],
    },
    {
        "key": "report", "label": "11. Regenerate statistics.md",
        "service": "backtest", "method": "POST", "path": "/backtest/report",
        "default": True, "core": True,
        "desc": "Final consolidation of ALL assets into docs/statistics.md (run last so bootstrap/label/seed assets are included).",
        "params": [],
    },
]

# Deletable pre-run artifacts. The dashboard container mounts data/, assets/,
# docs/ and models/, so it can remove these directly.
_CLEAN_TARGETS: List[Dict[str, Any]] = [
    {
        "key": "wf_cache", "label": "Walk-forward cache (wf_cache.parquet)", "default": True,
        "desc": "Essential for a true retraining: otherwise a cache hit skips train-all entirely.",
    },
    {
        "key": "derived_data", "label": "Derived data (test_df, backtesting_*, mcs_data)", "default": True,
        "desc": "Stale model/backtest outputs from a previous run.",
    },
    {
        "key": "assets", "label": "All generated assets + docs/statistics.md", "default": False,
        "desc": "Full wipe of assets/. Also removes notebook figures; keep the Notebooks step on to regenerate them.",
    },
    {
        "key": "optuna_db", "label": "Optuna studies DB (HPO history)", "default": False,
        "desc": "Destroys all HPO results; only when re-running HPO from scratch.",
    },
]

_STEP_BY_KEY = {s["key"]: s for s in _PIPELINE_STEPS}

# Job state (single job at a time). Guarded by _PIPELINE_LOCK for start/read.
_PIPELINE_LOCK = threading.Lock()
_PIPELINE_STOP = threading.Event()
_JOB: Dict[str, Any] = {"state": "idle"}


class PipelineRunRequest(BaseModel):
    # step key -> include? (omitted keys fall back to the step's `default`)
    steps: Dict[str, bool] = {}
    # step key -> {param name: value}
    params: Dict[str, Dict[str, Any]] = {}
    # clean target key -> include?
    clean: Dict[str, bool] = {}


def _clean_artifacts(cfg: PipelineConfig, opts: Dict[str, bool]) -> List[str]:
    """Delete the selected pre-run artifacts; returns the removed paths."""
    deleted: List[str] = []

    def _rm(path: Path) -> None:
        if path.exists() and path.is_file():
            path.unlink()
            deleted.append(str(path))

    if opts.get("wf_cache"):
        cache = Path(cfg.data_path("walk_forward_cache"))
        _rm(cache)
        _rm(Path(str(cache) + ".fingerprint"))

    if opts.get("derived_data"):
        for key in ("test_data", "backtesting_results", "backtesting_costs",
                    "backtesting_sorr", "mcs_data"):
            _rm(Path(cfg.data_path(key)))

    if opts.get("assets"):
        assets_dir = cfg._base_dir / "assets"
        if assets_dir.exists():
            for f in sorted(assets_dir.iterdir()):
                _rm(f)
        _rm(Path(cfg.asset_path("statistics_output")))  # docs/statistics.md

    if opts.get("optuna_db"):
        _rm(Path(cfg.model_path("optuna_db")))

    return deleted


def _run_notebooks(cfg: PipelineConfig, timeout_s: int = 1800) -> Dict[str, Any]:
    """Execute every jupyter/*.ipynb in place so its savefig() calls refresh the
    paper figures under assets/. Each notebook runs with its own directory as the
    working directory, so the notebooks' `../assets/` paths resolve correctly.

    Requires nbclient/nbformat/ipykernel in the image; a clear error is raised
    (and surfaced per step) when they are missing instead of crashing the job.
    """
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as e:
        raise RuntimeError(
            "Notebook execution needs nbclient/nbformat/ipykernel in the "
            "dashboard image. Rebuild it (docker compose build dashboard-service) "
            f"after adding the dashboard extra. Import error: {e}"
        )

    nb_dir = cfg._base_dir / "jupyter"
    if not nb_dir.exists():
        raise RuntimeError(
            f"jupyter/ not found at {nb_dir}. Mount ./jupyter into the dashboard "
            "container (see docker-compose.yml)."
        )

    notebooks = sorted(p for p in nb_dir.glob("*.ipynb")
                       if ".ipynb_checkpoints" not in p.parts)
    results: Dict[str, str] = {}
    for nb_path in notebooks:
        logger.info(f"[pipeline] executing notebook: {nb_path.name}")
        nb = nbformat.read(str(nb_path), as_version=4)
        client = NotebookClient(
            nb, timeout=timeout_s, kernel_name="python3",
            resources={"metadata": {"path": str(nb_dir)}},
        )
        client.execute()
        results[nb_path.name] = "ok"
    return {"executed": results, "count": len(notebooks)}


def _proxy_call(service: str, path: str, method: str, query: Dict[str, Any]):
    """Blocking service call used inside the orchestrator thread."""
    base = _SERVICES[service]
    url = base.rstrip("/") + path
    with httpx.Client(timeout=_TIMEOUT) as client:
        if method == "GET":
            r = client.get(url, params=query)
        else:
            r = client.post(url, params=query)
    try:
        body = r.json()
    except ValueError:
        body = {"text": r.text}
    return r.status_code, r.is_success, body


def _selected_steps(req: PipelineRunRequest) -> List[Dict[str, Any]]:
    """Ordered steps included in this run (request override, else step default)."""
    return [s for s in _PIPELINE_STEPS
            if req.steps.get(s["key"], s["default"])]


def _build_query(step: Dict[str, Any], req: PipelineRunRequest) -> Dict[str, Any]:
    """Resolve a step's query params from the request (falling back to defaults)."""
    supplied = req.params.get(step["key"], {})
    query: Dict[str, Any] = {}
    for p in step["params"]:
        val = supplied.get(p["name"], p.get("default"))
        if val is not None and val != "":
            query[p["name"]] = val
    return query


def _run_pipeline(req: PipelineRunRequest) -> None:
    """Worker: walk the selected steps sequentially, updating _JOB in place."""
    cfg = PipelineConfig()
    steps = _selected_steps(req)

    _JOB["steps"] = [
        {"key": s["key"], "label": s["label"], "status": "pending",
         "http_status": None, "elapsed_s": None, "error": None, "summary": None}
        for s in steps
    ]
    _JOB["state"] = "running"
    _JOB["error"] = None
    _JOB["started_at"] = time.time()
    _JOB["finished_at"] = None

    # 0. Clean (best effort; a failure here aborts before any compute).
    try:
        deleted = _clean_artifacts(cfg, req.clean)
        _JOB["clean"] = {"deleted": deleted}
        if deleted:
            logger.info(f"[pipeline] cleaned {len(deleted)} artifact(s) before run.")
    except Exception as e:  # noqa: BLE001
        logger.exception("[pipeline] clean step failed")
        _JOB["state"] = "failed"
        _JOB["error"] = f"Clean failed: {e}"
        _JOB["finished_at"] = time.time()
        return

    # Failure policy: a CORE step feeds later steps, so its failure aborts the
    # run (the rest is skipped). An OPTIONAL step (HPO, seed sensitivity, label
    # analysis, bootstrap, notebooks) is best-effort: a failure is recorded but
    # the run continues, so a single flaky extra never blocks the core paper
    # assets (statistics.md et al.).
    optional_failures: List[str] = []

    for i, step in enumerate(steps):
        if _PIPELINE_STOP.is_set():
            for js in _JOB["steps"][i:]:
                js["status"] = "skipped"
            _JOB["state"] = "stopped"
            _JOB["optional_failures"] = optional_failures
            logger.info("[pipeline] stopped by request before step "
                        f"'{step['key']}'.")
            _JOB["finished_at"] = time.time()
            return

        js = _JOB["steps"][i]
        js["status"] = "running"
        js["started_at"] = time.time()
        logger.info(f"[pipeline] >>> {step['label']} ({step['service']})")
        t0 = time.time()

        err = None
        try:
            if step["service"] == "_local" and step["key"] == "notebooks":
                js["summary"] = _run_notebooks(cfg)
                js["http_status"] = 200
            else:
                query = _build_query(step, req)
                code, ok, body = _proxy_call(
                    step["service"], step["path"], step["method"], query)
                js["http_status"] = code
                if ok:
                    js["summary"] = body.get("status", body) if isinstance(body, dict) else body
                else:
                    err = f"HTTP {code}: {body}"
        except Exception as e:  # noqa: BLE001
            logger.exception(f"[pipeline] step '{step['key']}' raised")
            err = str(e)

        js["elapsed_s"] = round(time.time() - t0, 1)

        if err is None:
            js["status"] = "ok"
            logger.info(f"[pipeline] <<< {step['label']} done in {js['elapsed_s']}s")
            continue

        # Step failed.
        js["status"] = "failed"
        js["error"] = err

        if step.get("core"):
            _JOB["state"] = "failed"
            _JOB["error"] = f"Core step '{step['key']}' failed: {err}"
            for rest in _JOB["steps"][i + 1:]:
                rest["status"] = "skipped"
            _JOB["optional_failures"] = optional_failures
            logger.error(f"[pipeline] core step '{step['key']}' failed: {err}")
            _JOB["finished_at"] = time.time()
            return

        optional_failures.append(step["key"])
        logger.warning(f"[pipeline] optional step '{step['key']}' failed, continuing: {err}")

    _JOB["optional_failures"] = optional_failures
    _JOB["state"] = "done" if not optional_failures else "done_with_errors"
    _JOB["finished_at"] = time.time()
    total = round(_JOB["finished_at"] - _JOB["started_at"], 1)
    logger.info(f"[pipeline] full run complete in {total}s "
                f"(optional failures: {optional_failures or 'none'})")


@router.get("/pipeline/plan")
def pipeline_plan() -> Dict[str, Any]:
    """Canonical step catalog + clean targets for the UI to render."""
    return {"steps": _PIPELINE_STEPS, "clean": _CLEAN_TARGETS}


@router.get("/pipeline/status")
def pipeline_status() -> Dict[str, Any]:
    """Current job snapshot (polled by the UI)."""
    return _JOB


@router.post("/pipeline/run")
def pipeline_run(req: PipelineRunRequest = Body(default=PipelineRunRequest())) -> Dict[str, Any]:
    """Start the full pipeline in a background thread. 409 while one is running."""
    with _PIPELINE_LOCK:
        if _JOB.get("state") == "running":
            raise HTTPException(409, "A pipeline run is already in progress.")

        # Validate keys early so a typo does not silently drop a step.
        unknown = [k for k in req.steps if k not in _STEP_BY_KEY]
        if unknown:
            raise HTTPException(400, f"Unknown step keys: {unknown}")

        _PIPELINE_STOP.clear()
        _JOB.clear()
        _JOB.update({"state": "running", "steps": [], "error": None,
                     "started_at": time.time(), "finished_at": None, "clean": {}})
        selected = [s["key"] for s in _selected_steps(req)]
        thread = threading.Thread(target=_run_pipeline, args=(req,), daemon=True)
        thread.start()

    return {"status": "started", "steps": selected}


@router.post("/pipeline/stop")
async def pipeline_stop() -> Dict[str, Any]:
    """Request a graceful stop: the current step finishes, no further step starts.

    If an HPO step is currently running, its own optimize-stop is forwarded so the
    long-running study halts after the current trial instead of running to the end.
    """
    _PIPELINE_STOP.set()

    # Forward optimize-stop if an HPO-style step is the one running right now.
    running = next((s for s in _JOB.get("steps", []) if s.get("status") == "running"), None)
    forwarded = False
    if running and (step := _STEP_BY_KEY.get(running["key"])) and step.get("stop_path"):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(_SERVICES[step["service"]].rstrip("/") + step["stop_path"])
            forwarded = True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[pipeline] optimize-stop forward failed: {e}")

    logger.info("[pipeline] stop requested (forwarded HPO stop: %s).", forwarded)
    return {"status": "stopping", "forwarded_hpo_stop": forwarded}
