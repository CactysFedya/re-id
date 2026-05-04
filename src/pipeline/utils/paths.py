from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Optional


def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / "src").exists() and (parent / "assets").exists():
            return parent
    return cur.resolve()


def timestamp_str(dt: datetime | None = None) -> str:
    dt = dt or datetime.now()
    return dt.strftime("%d-%m-%Y_%H-%M-%S")


def make_run_dir(base_dir: Path, *, prefix: str) -> Path:
    base_dir = Path(base_dir)
    run_dir = base_dir / prefix / timestamp_str()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_path(base_dir: Path, value: str | Path | None) -> Optional[Path]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = Path(base_dir) / candidate
    return candidate.resolve()
