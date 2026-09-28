"""Shared path resolution so scripts work from any working directory."""
from pathlib import Path

_MARKERS = ("pyproject.toml", "requirements.txt", ".git")

def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in _MARKERS):
            return parent
    raise FileNotFoundError(
        "Could not locate project root (no pyproject.toml/requirements.txt/.git "
        "found in any parent directory)."
    )

PROJECT_ROOT = find_project_root()
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DB_PATH = PROJECT_ROOT / "data" / "processed" / "nba.db"