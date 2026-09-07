"""Run provenance and immutable artifact directories."""

from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
import uuid
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .errors import TransitExoplanetMLError

TRACKED_PACKAGES = (
    "transit-exoplanet-ml",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "lightgbm",
    "catboost",
    "joblib",
    "pyyaml",
    "matplotlib",
    "typer",
)


def new_run_id() -> str:
    """A sortable, unique identifier for one training or evaluation run."""
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def git_commit(repo: Path | None = None) -> dict[str, Any]:
    """Current commit and worktree cleanliness, or a reason it is unavailable."""
    cwd = str(repo or Path.cwd())
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (subprocess.SubprocessError, OSError) as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}


def dependency_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:  # pragma: no cover - optional at runtime
            versions[name] = "not-installed"
    return versions


def provenance(seed: int, run_id: str | None = None, repo: Path | None = None) -> dict[str, Any]:
    """The complete provenance block stored with every run."""
    return {
        "run_id": run_id or new_run_id(),
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "seed": seed,
        "git": git_commit(repo),
        "dependencies": dependency_versions(),
    }


class _Encoder(json.JSONEncoder):
    """Serialise NumPy scalars, arrays, and paths that appear in reports."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            value = float(o)
            return value if np.isfinite(value) else None
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, dt.datetime | dt.date):
            return o.isoformat()
        return super().default(o)


def _clean(value: Any) -> Any:
    """Normalise NumPy scalars and replace non-finite floats with null.

    JSON has no NaN or Infinity, so anything non-finite becomes ``null`` rather
    than the invalid literals ``json.dumps`` would otherwise emit.
    """
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_clean(payload), indent=2, cls=_Encoder, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_yaml(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(_clean(payload), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return path


def write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def prepare_artifact_dir(base: str | Path, run_id: str, overwrite: bool = False) -> Path:
    """Create ``base/run_id`` as an immutable destination for one run."""
    directory = Path(base) / run_id
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise TransitExoplanetMLError(
            f"Artifact directory {directory} already exists and is not empty. "
            "Runs are immutable: choose a different --artifact-dir or pass --overwrite."
        )
    directory.mkdir(parents=True, exist_ok=True)
    return directory
