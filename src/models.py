"""Base-learner construction and deterministic hyper-parameter candidate sets."""

from __future__ import annotations

import itertools
from math import prod
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .config import BASE_LEARNERS, Config
from .errors import ConfigError

# Above this many combinations the full product is sampled rather than enumerated.
MAX_ENUMERATED_COMBINATIONS = 20_000

_ESTIMATORS: dict[str, type[BaseEstimator]] = {
    "logistic_regression": LogisticRegression,
    "svm_rbf": SVC,
    "extra_trees": ExtraTreesClassifier,
    "lightgbm": LGBMClassifier,
}


def build_estimator(learner: str, params: dict[str, Any], config: Config) -> BaseEstimator:
    """Instantiate ``learner`` with fixed config parameters plus ``params``."""
    if learner not in _ESTIMATORS:
        raise ConfigError(f"Unknown base learner {learner!r}; known: {sorted(_ESTIMATORS)}.")
    spec = config.model_spec(learner)
    kwargs: dict[str, Any] = dict(spec.get("fixed") or {})
    kwargs.update(params)
    kwargs["random_state"] = config.seed
    estimator = _ESTIMATORS[learner](**kwargs)

    calibration = spec.get("calibration")
    if calibration:
        # SVC exposes only a decision function; probabilities come from Platt
        # scaling fitted by cross-validation inside the training fold.
        estimator = CalibratedClassifierCV(estimator, **calibration)
    return estimator


def candidate_params(learner: str, config: Config) -> list[dict[str, Any]]:
    """Deterministically derive the tuning candidates for ``learner``.

    Small grids are enumerated in full and then sub-sampled; large grids (the
    LightGBM one) are sampled without replacement. Both paths depend only on
    ``config.seed``, so a run is reproducible.
    """
    spec = config.model_spec(learner)
    grid: dict[str, list[Any]] = {k: list(v) for k, v in spec["grid"].items()}
    n_candidates = int(spec["n_candidates"])
    if not grid:
        raise ConfigError(f"models.{learner}.grid is empty.")
    for key, values in grid.items():
        if not values:
            raise ConfigError(f"models.{learner}.grid.{key} has no values.")

    keys = sorted(grid)
    values = [grid[k] for k in keys]
    total = prod(len(v) for v in values)
    rng = np.random.default_rng(config.seed)

    if total <= n_candidates:
        combos = list(itertools.product(*values))
    elif total <= MAX_ENUMERATED_COMBINATIONS:
        everything = list(itertools.product(*values))
        chosen = np.sort(rng.permutation(total)[:n_candidates])
        combos = [everything[int(i)] for i in chosen]
    else:
        seen: set[tuple[Any, ...]] = set()
        combos = []
        while len(combos) < n_candidates:
            combo = tuple(vals[int(rng.integers(len(vals)))] for vals in values)
            if combo not in seen:
                seen.add(combo)
                combos.append(combo)

    return [dict(zip(keys, combo)) for combo in combos]


def all_candidate_params(config: Config) -> dict[str, list[dict[str, Any]]]:
    return {name: candidate_params(name, config) for name in BASE_LEARNERS}


def build_meta_estimator(config: Config) -> LogisticRegression:
    """The fixed L2 logistic meta-model; it only ever sees base probabilities."""
    kwargs = dict(config.meta)
    kwargs.setdefault("random_state", config.seed)
    return LogisticRegression(**kwargs)
