"""Config-Editor-API: Liest und schreibt config/config.yaml.

Beim Speichern:
  1. YAML-Syntax-Check (yaml.safe_load)
  2. Schema-Check über PipelineConfig-Reload (wirft FileNotFoundError/KeyError,
     wenn ein kritisches Feld fehlt)
  3. Backup der alten Datei als config.yaml.YYYYMMDD-HHMMSS.bak
  4. Atomarer Swap via temp-file
"""
from datetime import datetime
from pathlib import Path
import logging
import os
import tempfile

import yaml
from fastapi import APIRouter, HTTPException, Body

from config.config_loader import PipelineConfig

router = APIRouter(prefix="/api/config", tags=["config"])
logger = logging.getLogger("dashboard_service")


def _config_path() -> Path:
    """Lokaler Pfad zur config.yaml (innerhalb Container: /app/config/config.yaml)."""
    return Path(PipelineConfig()._path)


@router.get("")
def get_config_raw():
    """Liefert die aktuelle config.yaml als reinen Text + Meta."""
    path = _config_path()
    content = path.read_text(encoding="utf-8")
    mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "path": str(path),
        "content": content,
        "mtime": mtime,
        "size_bytes": path.stat().st_size,
    }


@router.post("")
def save_config(payload: dict = Body(...)):
    """Speichert neuen YAML-Inhalt in config.yaml nach erfolgreicher Validierung.

    Payload: {"content": "<gesamter YAML-Text>"}
    """
    new_content = payload.get("content")
    if not isinstance(new_content, str) or not new_content.strip():
        raise HTTPException(400, "Payload muss 'content' (nicht-leerer String) enthalten")

    # 1) YAML-Parse-Check
    try:
        parsed = yaml.safe_load(new_content)
    except yaml.YAMLError as e:
        raise HTTPException(422, f"YAML-Syntaxfehler: {e}")

    if not isinstance(parsed, dict):
        raise HTTPException(422, "Top-Level der YAML muss ein Mapping/Dict sein")

    # 2) Minimale Struktur-Checks (Pflicht-Sections, damit Pipeline lauffähig bleibt)
    required_sections = [
        "data", "features", "portfolio", "models", "backtesting",
        "walk_forward", "evaluation", "paths", "plotting",
    ]
    missing = [s for s in required_sections if s not in parsed]
    if missing:
        raise HTTPException(422, f"Fehlende Config-Sections: {missing}")

    # 3) Backup schreiben
    path = _config_path()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_name(f"{path.stem}.{ts}.bak")
    try:
        backup_path.write_bytes(path.read_bytes())
    except Exception as e:
        raise HTTPException(500, f"Backup fehlgeschlagen: {e}")

    # 4) Atomarer Swap via tmp-file
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".cfg-", suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(new_content)
        os.replace(tmp_name, path)  # atomic rename
    except Exception as e:
        # Bei Fehler tmp-file aufräumen
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(500, f"Schreiben fehlgeschlagen: {e}")

    # 5) Reload-Check (Loader muss neue Datei lesen können)
    try:
        _ = PipelineConfig()  # wirft bei Schema-Problemen
    except Exception as e:
        # Rollback: Backup zurückspielen
        try:
            path.write_bytes(backup_path.read_bytes())
        except Exception:
            pass
        raise HTTPException(422, f"Config-Reload nach Speichern fehlgeschlagen: {e} "
                                  f"(Rollback aus {backup_path.name} durchgeführt)")

    logger.info(f"Config gespeichert. Backup: {backup_path.name}")
    return {
        "status": "ok",
        "backup": backup_path.name,
        "bytes_written": len(new_content.encode("utf-8")),
        "reloaded": True,
    }


@router.get("/backups")
def list_backups():
    """Liste aller .bak-Dateien im config/-Ordner (neueste zuerst)."""
    path = _config_path()
    backups = sorted(
        path.parent.glob(f"{path.stem}.*.bak"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return {
        "backups": [
            {
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            }
            for p in backups
        ]
    }


@router.post("/restore")
def restore_backup(payload: dict = Body(...)):
    """Restored eine bestimmte Backup-Datei als aktive config.yaml.

    Payload: {"name": "config.20260415-183000.bak"}
    """
    name = payload.get("name", "")
    if not name.endswith(".bak") or "/" in name or ".." in name:
        raise HTTPException(400, "Ungültiger Backup-Name")

    path = _config_path()
    backup = path.parent / name
    if not backup.exists():
        raise HTTPException(404, f"Backup nicht gefunden: {name}")

    # Aktuellen Zustand vorher sichern
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    pre_restore = path.with_name(f"{path.stem}.{ts}.pre-restore.bak")
    pre_restore.write_bytes(path.read_bytes())

    path.write_bytes(backup.read_bytes())
    logger.info(f"Restored {name} (vorheriger Zustand: {pre_restore.name})")
    return {"status": "ok", "restored": name, "previous_saved_as": pre_restore.name}
