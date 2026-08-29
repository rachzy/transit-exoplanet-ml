"""Loading and resolution of the versioned YAML configuration."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

DEFAULT_CONFIG_RESOURCE = "config.yaml"

BASE_LEARNERS: tuple[str, ...] = ("lightgbm", "extra_trees", "svm_rbf", "logistic_regression")

# Every model that can be reported and shipped. Defined here (rather than in
# `stacking`) so config validation does not import the modelling stack.
ALL_MODELS: tuple[str, ...] = ("stack", *BASE_LEARNERS, "probability_average")


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
    def min_recall(self) -> float:
        return float(self.data["objective"]["min_recall"])

    @property
    def weighted_metrics(self) -> bool:
        return bool(self.data["objective"].get("weighted_metrics", True))

    @property
    def cv(self) -> dict[str, Any]:
        return self.data["cv"]

    @property
    def selection(self) -> dict[str, Any]:
        return self.data["selection"]

    @property
    def selection_candidates(self) -> tuple[str, ...]:
        return tuple(self.selection["candidates"])

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

    if not 0.0 < config.min_recall <= 1.0:
        raise ConfigError(f"objective.min_recall must be in (0, 1]; got {config.min_recall}.")

    selection = config.selection
    candidates = selection.get("candidates") or []
    if not candidates:
        raise ConfigError("selection.candidates must list at least one model.")
    unknown = [c for c in candidates if c not in ALL_MODELS]
    if unknown:
        raise ConfigError(
            f"selection.candidates names unknown models {unknown}; known: {list(ALL_MODELS)}."
        )
    strategy = selection.get("strategy")
    if strategy != SELECTION_STRATEGY_BEST and strategy not in candidates:
        raise ConfigError(
            f"selection.strategy must be {SELECTION_STRATEGY_BEST!r} or one of the "
            f"configured candidates {list(candidates)}; got {strategy!r}."
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

    missing = [name for name in BASE_LEARNERS if name not in data["models"]]
    if missing:
        raise ConfigError(f"Config is missing base learners: {missing}.")

    for name in BASE_LEARNERS:
        spec = data["models"][name]
        if "grid" not in spec or not isinstance(spec["grid"], dict):
            raise ConfigError(f"models.{name}.grid must be a mapping of parameter -> values.")
        if int(spec.get("n_candidates", 0)) < 1:
            raise ConfigError(f"models.{name}.n_candidates must be at least 1.")
