"""Loading and resolution of the versioned YAML configuration."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

DEFAULT_CONFIG_RESOURCE = "config.yaml"

SUPPORTED_METRICS: tuple[str, ...] = ("precision", "average_precision")


@dataclass(frozen=True)
class Config:
    """A fully resolved run configuration.

    The raw mapping is retained verbatim so it can be persisted next to the
    model bundle exactly as it was applied.
    """

    data: dict[str, Any]
    source: Path | None = field(default=None, compare=False)

    # -- convenience accessors ------------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.data["seed"])

    @property
    def objective_metric(self) -> str:
        return str(self.data["objective"]["metric"])

    @property
    def min_recall(self) -> float:
        return float(self.data["objective"]["min_recall"])

    @property
    def weighted_metrics(self) -> bool:
        return bool(self.data["objective"].get("weighted_metrics", True))

    @property
    def accepted_candidate_multiplier(self) -> float:
        """Relative base-learner weight of accepted versus other candidates."""
        return float(
            self.data.get("weighting", {}).get("accepted_candidate_multiplier", 1.0)
        )

    @property
    def cv(self) -> dict[str, Any]:
        return self.data["cv"]

    @property
    def selection(self) -> dict[str, Any]:
        return self.data["selection"]

    @property
    def selection_candidates(self) -> tuple[str, ...]:
        configured = set(self.base_learners)
        return tuple(
            name
            for name in self.selection.get("candidates", ("stack", *self.base_learners))
            if name == "stack" or name in configured
        )

    @property
    def base_learners(self) -> tuple[str, ...]:
        """Base learners enabled by the configured ``models`` mapping."""
        return tuple(self.data["models"])

    @property
    def all_models(self) -> tuple[str, ...]:
        """Stack plus the configured base learners."""
        return ("stack", *self.base_learners)

    @property
    def meta(self) -> dict[str, Any]:
        return self.data["meta"]

    @property
    def bootstrap(self) -> dict[str, Any]:
        return self.data["bootstrap"]

    @property
    def permutation_importance(self) -> dict[str, Any]:
        return self.data["permutation_importance"]

    @property
    def calibration_bins(self) -> int:
        return int(self.data.get("calibration", {}).get("n_bins", 10))

    def model_spec(self, name: str) -> dict[str, Any]:
        try:
            return self.data["models"][name]
        except KeyError as exc:
            raise ConfigError(f"No configuration for base learner {name!r}.") from exc

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    def with_overrides(self, overrides: dict[str, Any]) -> Config:
        """Return a copy with ``overrides`` deep-merged on top."""
        merged = _deep_merge(copy.deepcopy(self.data), overrides)
        resolved = Config(data=merged, source=self.source)
        _validate(resolved)
        return resolved


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load the packaged configuration, or a user-supplied replacement."""
    resolved = Path(path) if path is not None else None
    if resolved is None:
        text = (
            resources.files(f"{__package__}.resources")
            .joinpath(DEFAULT_CONFIG_RESOURCE)
            .read_text(encoding="utf-8")
        )
    else:
        if not resolved.is_file():
            raise ConfigError(f"Config file not found: {resolved}")
        text = resolved.read_text(encoding="utf-8")

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigError("Config file must contain a YAML mapping.")

    config = Config(data=data, source=resolved)
    if overrides:
        return config.with_overrides(overrides)
    _validate(config)
    return config


SELECTION_STRATEGY_BEST = "best"
SELECTION_SOURCES = ("nested_evaluation", "crossfit")


def _validate(config: Config) -> None:
    data = config.data
    for key in ("seed", "objective", "selection", "cv", "models", "meta"):
        if key not in data:
            raise ConfigError(f"Config is missing required section {key!r}.")

    multiplier = config.accepted_candidate_multiplier
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ConfigError(
            "weighting.accepted_candidate_multiplier must be a positive finite "
            f"number; got {multiplier}."
        )

    if config.objective_metric not in SUPPORTED_METRICS:
        raise ConfigError(
            f"objective.metric must be one of {list(SUPPORTED_METRICS)}; "
            f"got {config.objective_metric!r}."
        )
    if not 0.0 < config.min_recall <= 1.0:
        raise ConfigError(f"objective.min_recall must be in (0, 1]; got {config.min_recall}.")

    selection = config.selection
    selection_metric = selection.get("metric")
    if selection_metric not in SUPPORTED_METRICS:
        raise ConfigError(
            f"selection.metric must be one of {list(SUPPORTED_METRICS)}; "
            f"got {selection_metric!r}."
        )
    configured_candidates = selection.get("candidates")
    if configured_candidates is None:
        candidates = ["stack", *config.base_learners]
    else:
        candidates = configured_candidates
    if not candidates:
        raise ConfigError("selection.candidates must list at least one model.")
    from .models import known_learners

    unknown = [name for name in candidates if name != "stack" and name not in known_learners()]
    if unknown:
        raise ConfigError(
            f"selection.candidates names unknown models {unknown}; known learners are "
            f"{sorted(known_learners())}."
        )
    active_candidates = [
        name for name in candidates if name == "stack" or name in config.base_learners
    ]
    if not active_candidates:
        raise ConfigError("selection.candidates must list at least one model.")
    strategy = selection.get("strategy")
    if strategy != SELECTION_STRATEGY_BEST and strategy not in active_candidates:
        raise ConfigError(
            f"selection.strategy must be {SELECTION_STRATEGY_BEST!r} or one of the "
            f"configured candidates {active_candidates}; got {strategy!r}."
        )
    source = selection.get("score_source")
    if source not in SELECTION_SOURCES:
        raise ConfigError(
            f"selection.score_source must be one of {list(SELECTION_SOURCES)}; got {source!r}."
        )

    cv = config.cv
    for key in ("outer_folds", "inner_folds", "min_outer_folds", "min_inner_folds"):
        if key not in cv:
            raise ConfigError(f"Config section cv is missing {key!r}.")
        if int(cv[key]) < 2:
            raise ConfigError(f"cv.{key} must be at least 2; got {cv[key]}.")
    if int(cv["min_outer_folds"]) < 3:
        raise ConfigError("cv.min_outer_folds must be at least 3 for honest reporting.")
    if int(cv["outer_folds"]) < int(cv["min_outer_folds"]):
        raise ConfigError("cv.outer_folds must not be below cv.min_outer_folds.")
    if int(cv["inner_folds"]) < int(cv["min_inner_folds"]):
        raise ConfigError("cv.inner_folds must not be below cv.min_inner_folds.")

    if not isinstance(data["models"], dict) or not data["models"]:
        raise ConfigError("Config section models must contain at least one learner.")

    from .models import known_learners

    unknown_learners = [name for name in data["models"] if name not in known_learners()]
    if unknown_learners:
        raise ConfigError(
            f"models names unknown learners {unknown_learners}; known learners are "
            f"{sorted(known_learners())}."
        )
    for name, spec in data["models"].items():
        if "grid" not in spec or not isinstance(spec["grid"], dict):
            raise ConfigError(f"models.{name}.grid must be a mapping of parameter -> values.")
        if int(spec.get("n_candidates", 0)) < 1:
            raise ConfigError(f"models.{name}.n_candidates must be at least 1.")

    meta = data["meta"]
    meta_grid = meta.get("grid")
    if meta_grid is None and "C" in meta:
        # Accept pre-grid configs as a one-candidate search for compatibility.
        meta_grid = {"C": [meta["C"]]}
    if not isinstance(meta_grid, dict) or "C" not in meta_grid:
        raise ConfigError("meta.grid.C must be a non-empty tuning grid.")
    if not meta_grid["C"]:
        raise ConfigError("meta.grid.C must be a non-empty tuning grid.")
    if int(meta.get("n_candidates", 1)) < 1:
        raise ConfigError("meta.n_candidates must be at least 1.")
    for value in meta_grid["C"]:
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ConfigError(
                "meta.grid.C values must be positive and finite; "
                f"got {value!r}."
            )
