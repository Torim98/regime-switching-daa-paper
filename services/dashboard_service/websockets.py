"""Live-Log-Streaming: WebSocket tailt logs/*.log-Dateien.

File-Tail-Ansatz (statt Docker-Socket) — portabel, sicher, funktioniert
überall wo das logs/-Volume gemountet ist.

Protokoll:
  Client verbindet auf  /ws/logs/{filename}
  Server sendet Text-Frames mit je einer neuen Log-Zeile.
  Auf Wunsch kann der Client "?tail=200" als Query-Param mitgeben — dann
  werden zuerst die letzten 200 Zeilen ausgeliefert, danach Live-Updates.
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
    """Path-Traversal-Schutz + Existenz-Check."""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, f"Ungültiger Dateiname: {filename}")
    path = _logs_dir() / filename
    resolved = path.resolve()
    logs_root = _logs_dir().resolve()
    if logs_root not in resolved.parents and resolved != logs_root:
        raise HTTPException(400, f"Pfad außerhalb logs/: {filename}")
    return path


@router.get("/api/logs/files")
def list_log_files():
    """Verfügbare Log-Dateien (Service- und Pipeline-Logs)."""
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
    """Letzte N Zeilen (für Initial-Load ohne WS)."""
    path = _safe_resolve(filename)
    if not path.exists():
        raise HTTPException(404, f"Log-Datei nicht gefunden: {filename}")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(500, f"Lesen fehlgeschlagen: {e}")
    tail = text.splitlines()[-lines:]
    return {"file": filename, "lines": tail}


@router.websocket("/ws/logs/{filename}")
async def ws_logs(websocket: WebSocket, filename: str, tail: int = 200):
    """WebSocket-Endpoint: Streamt neue Zeilen einer Log-Datei.

    Erst letzte `tail` Zeilen, danach Live-Updates alle ~300 ms.
    """
    try:
        path = _safe_resolve(filename)
    except HTTPException as e:
        await websocket.close(code=4400, reason=str(e.detail))
        return

    await websocket.accept()

    try:
        # 1) Initial-Tail liefern
        if path.exists():
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                for line in text.splitlines()[-tail:]:
                    await websocket.send_text(line)
                pos = path.stat().st_size
            except Exception as e:
                await websocket.send_text(f"[dashboard] Fehler beim Initial-Read: {e}")
                pos = 0
        else:
            await websocket.send_text(f"[dashboard] Datei existiert noch nicht: {filename} "
                                       f"— warte auf Erstellen …")
            pos = 0

        # 2) Live-Tail-Loop
        while True:
            await asyncio.sleep(0.3)
            if not path.exists():
                continue
            size = path.stat().st_size
            if size < pos:
                # Log wurde rotiert / getruncated
                await websocket.send_text("[dashboard] Datei truncated — resume from 0")
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
                    await websocket.send_text(f"[dashboard] Read-Error: {e}")
    except WebSocketDisconnect:
        logger.info(f"WS disconnect: {filename}")
    except Exception as e:
        logger.exception(f"WS error for {filename}: {e}")
        try:
            await websocket.close(code=1011, reason=f"Server error: {e}")
        except Exception:
            pass
