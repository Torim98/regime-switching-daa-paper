"""Config editor API: reads and writes config/config.yaml.

On save:
  1. YAML syntax check (yaml.safe_load)
  2. Schema check via PipelineConfig reload (raises FileNotFoundError/KeyError
     if a critical field is missing)
  3. Backup of the old file as config.yaml.YYYYMMDD-HHMMSS.bak
  4. Atomic swap via temp file
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
    """Local path to config.yaml (inside the container: /app/config/config.yaml)."""
    return Path(PipelineConfig()._path)


@router.get("")
def get_config_raw():
    """Returns the current config.yaml as plain text + meta."""
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
    """Saves new YAML content to config.yaml after successful validation.

    Payload: {"content": "<entire YAML text>"}
    """
    new_content = payload.get("content")
    if not isinstance(new_content, str) or not new_content.strip():
        raise HTTPException(400, "Payload must contain 'content' (non-empty string)")

    # 1) YAML parse check
    try:
        parsed = yaml.safe_load(new_content)
    except yaml.YAMLError as e:
        raise HTTPException(422, f"YAML syntax error: {e}")

    if not isinstance(parsed, dict):
        raise HTTPException(422, "Top level of the YAML must be a mapping/dict")

    # 2) Minimal structure checks (required sections so the pipeline stays runnable)
    required_sections = [
        "data", "features", "portfolio", "models", "backtesting",
        "walk_forward", "evaluation", "paths", "plotting",
    ]
    missing = [s for s in required_sections if s not in parsed]
    if missing:
        raise HTTPException(422, f"Missing config sections: {missing}")

    # 3) Write the backup
    path = _config_path()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_name(f"{path.stem}.{ts}.bak")
    try:
        backup_path.write_bytes(path.read_bytes())
    except Exception as e:
        raise HTTPException(500, f"Backup failed: {e}")

    # 4) Atomic swap via tmp file
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".cfg-", suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(new_content)
        os.replace(tmp_name, path)  # atomic rename
    except Exception as e:
        # Clean up the tmp file on error
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(500, f"Write failed: {e}")

    # 5) Reload check (the loader must be able to read the new file)
    try:
        _ = PipelineConfig()  # raises on schema problems
    except Exception as e:
        # Rollback: restore from the backup
        try:
            path.write_bytes(backup_path.read_bytes())
        except Exception:
            pass
        raise HTTPException(422, f"Config reload after saving failed: {e} "
                                  f"(rollback from {backup_path.name} performed)")

    logger.info(f"Config saved. Backup: {backup_path.name}")
    return {
        "status": "ok",
        "backup": backup_path.name,
        "bytes_written": len(new_content.encode("utf-8")),
        "reloaded": True,
    }


@router.get("/backups")
def list_backups():
    """List of all .bak files in the config/ directory (newest first)."""
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
    """Restores a specific backup file as the active config.yaml.

    Payload: {"name": "config.20260415-183000.bak"}
    """
    name = payload.get("name", "")
    if not name.endswith(".bak") or "/" in name or ".." in name:
        raise HTTPException(400, "Invalid backup name")

    path = _config_path()
    backup = path.parent / name
    if not backup.exists():
        raise HTTPException(404, f"Backup not found: {name}")

    # Save the current state first
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    pre_restore = path.with_name(f"{path.stem}.{ts}.pre-restore.bak")
    pre_restore.write_bytes(path.read_bytes())

    path.write_bytes(backup.read_bytes())
    logger.info(f"Restored {name} (previous state: {pre_restore.name})")
    return {"status": "ok", "restored": name, "previous_saved_as": pre_restore.name}
