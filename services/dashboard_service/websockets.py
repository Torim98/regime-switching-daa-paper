"""Live log streaming: WebSocket tails logs/*.log files.

File-tail approach (instead of the Docker socket): portable, safe, works
anywhere the logs/ volume is mounted.

Protocol:
  The client connects to  /ws/logs/{filename}
  The server sends text frames with one new log line each.
  Optionally, the client can pass "?tail=200" as a query param; the last
  200 lines are then delivered first, followed by live updates.
"""
import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, HTTPException

from config.config_loader import PipelineConfig

router = APIRouter(tags=["websockets"])
logger = logging.getLogger("dashboard_service")


def _logs_dir() -> Path:
    cfg = PipelineConfig()
    return cfg._base_dir / "logs"


def _safe_resolve(filename: str) -> Path:
    """Path traversal protection + existence check."""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, f"Invalid file name: {filename}")
    path = _logs_dir() / filename
    resolved = path.resolve()
    logs_root = _logs_dir().resolve()
    if logs_root not in resolved.parents and resolved != logs_root:
        raise HTTPException(400, f"Path outside logs/: {filename}")
    return path


@router.get("/api/logs/files")
def list_log_files():
    """Available log files (service and pipeline logs)."""
    logs = _logs_dir()
    if not logs.exists():
        return {"files": []}
    files = []
    for p in sorted(logs.glob("*.log")):
        files.append({
            "name": p.name,
            "size_kb": round(p.stat().st_size / 1024, 1),
            "mtime": p.stat().st_mtime,
        })
    return {"files": files}


@router.get("/api/logs/snapshot/{filename}")
def log_snapshot(filename: str, lines: int = Query(500, ge=1, le=10000)):
    """Last N lines (for the initial load without WS)."""
    path = _safe_resolve(filename)
    if not path.exists():
        raise HTTPException(404, f"Log file not found: {filename}")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(500, f"Read failed: {e}")
    tail = text.splitlines()[-lines:]
    return {"file": filename, "lines": tail}


@router.websocket("/ws/logs/{filename}")
async def ws_logs(websocket: WebSocket, filename: str, tail: int = 200):
    """WebSocket endpoint: streams new lines of a log file.

    First the last `tail` lines, then live updates every ~300 ms.
    """
    try:
        path = _safe_resolve(filename)
    except HTTPException as e:
        await websocket.close(code=4400, reason=str(e.detail))
        return

    await websocket.accept()

    try:
        # 1) Deliver the initial tail
        if path.exists():
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                for line in text.splitlines()[-tail:]:
                    await websocket.send_text(line)
                pos = path.stat().st_size
            except Exception as e:
                await websocket.send_text(f"[dashboard] Error during initial read: {e}")
                pos = 0
        else:
            await websocket.send_text(f"[dashboard] File does not exist yet: {filename}, "
                                       f"waiting for creation ...")
            pos = 0

        # 2) Live tail loop
        while True:
            await asyncio.sleep(0.3)
            if not path.exists():
                continue
            size = path.stat().st_size
            if size < pos:
                # Log was rotated / truncated
                await websocket.send_text("[dashboard] file truncated, resume from 0")
                pos = 0
            if size > pos:
                try:
                    with path.open("r", encoding="utf-8", errors="replace") as f:
                        f.seek(pos)
                        chunk = f.read()
                        pos = f.tell()
                    for line in chunk.splitlines():
                        if line:
                            await websocket.send_text(line)
                except Exception as e:
                    await websocket.send_text(f"[dashboard] Read error: {e}")
    except WebSocketDisconnect:
        logger.info(f"WS disconnect: {filename}")
    except Exception as e:
        logger.exception(f"WS error for {filename}: {e}")
        try:
            await websocket.close(code=1011, reason=f"Server error: {e}")
        except Exception:
            pass
