"""The stacked classifier: grouped splits, tuning, cross-fitting, thresholds.

The same routine (:func:`fit_stack`) builds the stack for an outer evaluation
fold and for the shipped model, so the reported metrics describe exactly the
procedure that is saved.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from .config import Config
from .data import composite_strata, star_balanced_weights
from .errors import DataDiversityError
from .metrics import (
    ThresholdChoice,
    accepted_average_precision,
    select_threshold,
)
from .models import (
    build_estimator,
    build_meta_estimator,
    candidate_params,
    meta_candidate_params,
)
from .preprocessing import build_preprocessor

STACK = "stack"

Progress = Callable[[str], None] | None

Split = tuple[np.ndarray, np.ndarray]


def report_progress(progress: Progress, message: str) -> None:
    """Emit a progress line if the caller supplied a sink."""
    if progress is not None:
        progress(message)


# ---------------------------------------------------------------------------
# Grouped splitting
# ---------------------------------------------------------------------------
def _support_table(strata: np.ndarray, groups: np.ndarray) -> dict[int, int]:
    """Distinct groups behind each stratum -- the binding constraint on folds."""
    return {
        int(s): int(np.unique(groups[strata == s]).size) for s in np.unique(strata)
    }


def resolve_n_splits(
    strata: np.ndarray,
    groups: np.ndarray,
    desired: int,
    minimum: int,
    level: str,
) -> int:
    """Largest workable fold count in ``[minimum, desired]``.

    Folds are only reduced when class/star support forces it; if even
    ``minimum`` folds are impossible the caller gets a diagnostic naming the
    strata that are too thin.
    """
    support = _support_table(strata, groups)
    n_groups = int(np.unique(groups).size)
    for n in range(desired, minimum - 1, -1):
        if n_groups >= n and all(count >= n for count in support.values()):
            return n

    thin = {s: c for s, c in support.items() if c < minimum}
    raise DataDiversityError(
        f"Cannot build {minimum} {level} folds grouped by star: "
        f"{n_groups} stars available; distinct stars per (label, acceptance) stratum "
        f"{support}; strata below the {minimum}-star minimum: {thin}. "
        "Add stars covering the under-represented combinations of candidate_label "
        "and detection_status."
    )


def grouped_splits(
    strata: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
    shuffle: bool = True,
) -> list[Split]:
    """Star-grouped, stratum-balanced splits as concrete index arrays."""
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None
    )
    dummy = np.zeros((strata.size, 1))
    return [
        (np.asarray(tr, dtype=int), np.asarray(va, dtype=int))
        for tr, va in splitter.split(dummy, strata, groups=groups)
    ]


# ---------------------------------------------------------------------------
# Fitted components
# ---------------------------------------------------------------------------
@dataclass
class FittedBase:
    """A base learner with the preprocessing pipeline fitted alongside it."""

    learner: str
    params: dict[str, Any]
    preprocessor: Pipeline
    estimator: BaseEstimator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        classes = np.asarray(getattr(self.estimator, "classes_", [0, 1]))
        if classes.size == 1:
            # Degenerate fit fold: fall back to the single observed class.
            return np.full(len(X), float(classes[0]))
        transformed = self.preprocessor.transform(X)
        return np.asarray(self.estimator.predict_proba(transformed))[:, 1]


@dataclass
class FittedStack:
    """Every candidate model, plus a record of which one ships.

    The configured base learners and the logistic meta-model above them are
    fitted, so the stack and each configured base learner can be scored and
    compared. ``selected_model`` names
    the one that :meth:`predict_proba` actually serves; the others are retained
    so a run can be re-examined, or the selection revisited, without refitting.
    """

    bases: dict[str, FittedBase]
    meta: LogisticRegression
    thresholds: dict[str, float]
    selected_model: str
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.selected_model not in self.thresholds:
            raise ValueError(
                f"No threshold for selected model {self.selected_model!r}; "
                f"have {sorted(self.thresholds)}."
            )

    @property
    def threshold(self) -> float:
        """The recall-floor threshold belonging to the selected model."""
        return float(self.thresholds[self.selected_model])

    @property
    def base_learners(self) -> tuple[str, ...]:
        return tuple(self.bases)

    def base_matrix(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [self.bases[name].predict_proba(X) for name in self.base_learners]
        )

    def all_probabilities(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Probabilities from every candidate model, keyed by name."""
        base = self.base_matrix(X)
        out: dict[str, np.ndarray] = {
            STACK: np.asarray(self.meta.predict_proba(base))[:, 1]
        }
        for column, learner in enumerate(self.base_learners):
            out[learner] = base[:, column]
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilities from the selected model only."""
        return self.all_probabilities(X)[self.selected_model]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X) >= self.threshold

    def with_selection(self, model: str) -> FittedStack:
        """A copy of this stack serving ``model`` instead."""
        return replace(self, selected_model=model)


def fit_base(
    learner: str,
    params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    accepted: np.ndarray,
    groups: np.ndarray,
    config: Config,
) -> FittedBase:
    """Fit a base learner on all rows, emphasizing accepted candidates."""
    preprocessor = build_preprocessor(learner)
    transformed = preprocessor.fit_transform(X)
    estimator = build_estimator(learner, params, config)
    row_multipliers = np.where(
        np.asarray(accepted, dtype=bool), config.accepted_candidate_multiplier, 1.0
    )
    weights = star_balanced_weights(groups, row_multipliers=row_multipliers)
    estimator.fit(transformed, y, sample_weight=weights)
    return FittedBase(
        learner=learner, params=dict(params), preprocessor=preprocessor, estimator=estimator
    )


def fit_meta(
    P: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    config: Config,
    params: dict[str, Any] | None = None,
) -> LogisticRegression:
    """Fit the meta-model on base probabilities for accepted rows only."""
    meta = build_meta_estimator(config, params=params)
    meta.fit(P, y, sample_weight=star_balanced_weights(groups))
    return meta


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
@dataclass
class TuningResult:
    learner: str
    best_params: dict[str, Any]
    best_score: float
    candidates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "learner": self.learner,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_candidates": len(self.candidates),
            "candidates": self.candidates,
        }


def _score_fold(
    y: np.ndarray,
    prob: np.ndarray,
    accepted: np.ndarray,
    groups: np.ndarray,
    metric: str,
    min_recall: float,
    weighted: bool,
) -> float:
    """Configured tuning score restricted to accepted candidates."""
    mask = accepted
    if mask.sum() == 0:
        return float("nan")
    weights = star_balanced_weights(groups[mask]) if weighted else None
    if metric == "average_precision":
        return accepted_average_precision(y[mask], prob[mask], sample_weight=weights)
    return select_threshold(
        y[mask], prob[mask], min_recall, sample_weight=weights
    ).precision


def tune_base_learner(
    learner: str,
    X: np.ndarray,
    y: np.ndarray,
    accepted: np.ndarray,
    groups: np.ndarray,
    splits: Sequence[Split],
    config: Config,
    progress: Progress = None,
) -> TuningResult:
    """Pick hyper-parameters by mean accepted-candidate average precision."""
    grid = candidate_params(learner, config)
    weighted = config.weighted_metrics
    records: list[dict[str, Any]] = []

    for position, params in enumerate(grid):
        scores: list[float] = []
        for tr, va in splits:
            fitted = fit_base(
                learner, params, X[tr], y[tr], accepted[tr], groups[tr], config
            )
            prob = fitted.predict_proba(X[va])
            scores.append(
                _score_fold(
                    y[va],
                    prob,
                    accepted[va],
                    groups[va],
                    config.objective_metric,
                    config.min_recall,
                    weighted,
                )
            )
        valid = [s for s in scores if not np.isnan(s)]
        mean_score = float(np.mean(valid)) if valid else float("nan")
        records.append(
            {
                "rank_order": position,
                "params": params,
                "mean_score": mean_score,
                "fold_scores": [float(s) for s in scores],
            }
        )
        report_progress(
            progress,
            f"    {learner}: candidate {position + 1}/{len(grid)} -> {mean_score:.4f}",
        )

    scored = [r for r in records if not np.isnan(r["mean_score"])]
    if not scored:
        raise DataDiversityError(
            f"No hyper-parameter candidate for {learner!r} could be scored: every inner "
            "validation fold lacked accepted candidates of both classes. Add stars with "
            "accepted CONFIRMED and accepted FALSE-POSITIVE candidates."
        )
    # Ties resolve to the earlier candidate, keeping selection deterministic.
    best = max(scored, key=lambda r: (r["mean_score"], -r["rank_order"]))
    return TuningResult(
        learner=learner,
        best_params=best["params"],
        best_score=best["mean_score"],
        candidates=records,
    )


@dataclass
class MetaFold:
    """Base probabilities constructed without the meta-validation stars.

    Row indices refer to the enclosing training dataset; probability matrices
    are local to their respective rows. Training probabilities are OOF within
    ``train_rows``. Validation probabilities come from bases refitted on those
    training rows, with hyperparameters selected there as well.
    """

    train_rows: np.ndarray
    val_rows: np.ndarray
    train_probabilities: np.ndarray
    val_probabilities: np.ndarray


def tune_meta_learner(
    folds: Sequence[MetaFold],
    y: np.ndarray,
    accepted: np.ndarray,
    groups: np.ndarray,
    config: Config,
    progress: Progress = None,
) -> TuningResult:
    """Tune C using independently constructed inputs for each meta fold."""
    grid = meta_candidate_params(config)
    weighted = config.weighted_metrics
    records: list[dict[str, Any]] = []

    for position, params in enumerate(grid):
        scores: list[float] = []
        for fold in folds:
            tr, va = fold.train_rows, fold.val_rows
            train_rows = tr[accepted[tr]]
            val_rows = va[accepted[va]]
            if train_rows.size == 0 or val_rows.size == 0:
                scores.append(float("nan"))
                continue
            if np.unique(y[train_rows]).size < 2:
                scores.append(float("nan"))
                continue
            meta = fit_meta(
                fold.train_probabilities[accepted[tr]],
                y[train_rows], groups[train_rows], config, params
            )
            probability = np.asarray(
                meta.predict_proba(fold.val_probabilities[accepted[va]])
            )[:, 1]
            scores.append(
                _score_fold(
                    y[val_rows],
                    probability,
                    np.ones(val_rows.size, dtype=bool),
                    groups[val_rows],
                    config.objective_metric,
                    config.min_recall,
                    weighted,
                )
            )
        valid = [score for score in scores if not np.isnan(score)]
        mean_score = float(np.mean(valid)) if valid else float("nan")
        records.append(
            {
                "rank_order": position,
                "params": params,
                "mean_score": mean_score,
                "fold_scores": [float(score) for score in scores],
            }
        )
        report_progress(
            progress,
            f"    meta: candidate {position + 1}/{len(grid)} -> {mean_score:.4f}",
        )

    scored = [record for record in records if not np.isnan(record["mean_score"])]
    if not scored:
        raise DataDiversityError(
            "No hyper-parameter candidate for the meta-model could be scored: every "
            "inner validation fold lacked accepted candidates of both classes."
        )
    best = max(
        scored, key=lambda record: (record["mean_score"], -record["rank_order"])
    )
    return TuningResult(
        learner="meta",
        best_params=best["params"],
        best_score=best["mean_score"],
        candidates=records,
    )


# ---------------------------------------------------------------------------
# Cross-fitting
# ---------------------------------------------------------------------------
def base_oof_matrix(
    best_params: dict[str, dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    accepted: np.ndarray,
    groups: np.ndarray,
    splits: Sequence[Split],
    config: Config,
) -> np.ndarray:
    """Out-of-fold base probabilities, one column per learner."""
    learners = config.base_learners
    P = np.full((X.shape[0], len(learners)), np.nan, dtype=float)
    for tr, va in splits:
        for column, learner in enumerate(learners):
            fitted = fit_base(
                learner,
                best_params[learner],
                X[tr],
                y[tr],
                accepted[tr],
                groups[tr],
                config,
            )
            P[va, column] = fitted.predict_proba(X[va])
    if np.isnan(P).any():
        missing = int(np.isnan(P).any(axis=1).sum())
        raise DataDiversityError(
            f"{missing} rows received no out-of-fold base prediction. The grouped split "
            "did not cover every row; check star support per stratum."
        )
    return P


def prepare_meta_folds(
    X: np.ndarray,
    y: np.ndarray,
    accepted: np.ndarray,
    groups: np.ndarray,
    splits: Sequence[Split],
    config: Config,
    progress: Progress = None,
) -> list[MetaFold]:
    """Rebuild the upstream pipeline within each meta-training partition.

    A single shared OOF matrix cannot isolate meta-validation: the base models
    behind its meta-training rows may have seen the meta-validation labels.
    Both base tuning and the OOF training matrix must instead be constructed
    using only this meta fold's training stars. Cache these inputs across C
    candidates and threshold cross-fitting.
    """
    folds: list[MetaFold] = []
    for position, (tr, va) in enumerate(splits):
        report_progress(progress, f"  preparing isolated meta fold {position + 1}/{len(splits)}")
        strata = composite_strata(y[tr], accepted[tr])
        n_splits = resolve_n_splits(
            strata, groups[tr], int(config.cv["inner_folds"]),
            int(config.cv["min_inner_folds"]),
            f"base-within-meta (meta fold {position + 1})",
        )
        base_splits = grouped_splits(
            strata, groups[tr], n_splits, config.seed, bool(config.cv.get("shuffle", True))
        )
        best_params = {
            learner: tune_base_learner(
                learner, X[tr], y[tr], accepted[tr], groups[tr],
                base_splits, config, progress,
            ).best_params
            for learner in config.base_learners
        }
        train_probabilities = base_oof_matrix(
            best_params, X[tr], y[tr], accepted[tr], groups[tr], base_splits, config
        )
        val_probabilities = np.column_stack([
            fit_base(
                learner, best_params[learner], X[tr], y[tr], accepted[tr], groups[tr], config
            ).predict_proba(X[va])
            for learner in config.base_learners
        ])
        folds.append(MetaFold(tr, va, train_probabilities, val_probabilities))
    return folds


def crossfit_meta_predictions(
    folds: Sequence[MetaFold],
    y: np.ndarray,
    accepted: np.ndarray,
    groups: np.ndarray,
    config: Config,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Cross-fitted meta probabilities for accepted rows; NaN elsewhere.

    Each meta fit uses inputs prepared without its validation stars. The caller
    supplies C selected by tuning over these folds; this does not add another
    level of cross-validation for hyperparameter selection itself.
    """
    predictions = np.full(y.shape[0], np.nan, dtype=float)
    for fold in folds:
        tr, va = fold.train_rows, fold.val_rows
        train_rows = tr[accepted[tr]]
        val_rows = va[accepted[va]]
        if val_rows.size == 0:
            continue
        if train_rows.size == 0 or np.unique(y[train_rows]).size < 2:
            continue
        meta = fit_meta(
            fold.train_probabilities[accepted[tr]],
            y[train_rows], groups[train_rows], config, params,
        )
        predictions[val_rows] = np.asarray(
            meta.predict_proba(fold.val_probabilities[accepted[va]])
        )[:, 1]
    return predictions


# ---------------------------------------------------------------------------
# Whole-stack fit
# ---------------------------------------------------------------------------
@dataclass
class StackFitResult:
    """Everything produced by fitting the stack on one training set."""

    stack: FittedStack
    best_params: dict[str, dict[str, Any]]
    tuning: dict[str, TuningResult]
    base_oof: np.ndarray
    meta_crossfit: np.ndarray
    thresholds: dict[str, ThresholdChoice]
    n_inner_folds: int
    train_stars: tuple[str, ...]

    @property
    def threshold(self) -> float:
        """Threshold of whichever model this fit selected."""
        return self.thresholds[self.stack.selected_model].threshold

    def training_probabilities(self) -> dict[str, np.ndarray]:
        """Cross-fitted training-side probabilities, per reported model."""
        out: dict[str, np.ndarray] = {STACK: self.meta_crossfit}
        for column, learner in enumerate(self.stack.base_learners):
            out[learner] = self.base_oof[:, column]
        return out


def model_probabilities(result: StackFitResult, X: np.ndarray) -> dict[str, np.ndarray]:
    """Probabilities from the stack, each base learner, and the mean baseline."""
    return result.stack.all_probabilities(X)


def reported_models(config: Config) -> tuple[str, ...]:
    """Every configured candidate, in the order that breaks selection ties."""
    return (STACK, *config.base_learners)


def select_best_model(
    scores: dict[str, float],
    candidates: Sequence[str] | None = None,
    recalls: dict[str, float] | None = None,
    min_recall: float | None = None,
) -> str:
    """Name the highest-scoring candidate, breaking ties deterministically.

    Ties fall to whichever model comes first in :func:`reported_models`, which
    puts the stack ahead of its own parts. Candidates that could not be scored
    (NaN) are skipped. When supplied, ``recalls`` and ``min_recall`` restrict
    selection to candidates that meet the recall floor.
    """
    order = list(candidates) if candidates is not None else list(scores)
    scored = [
        (name, scores[name])
        for name in order
        if name in scores and not np.isnan(scores[name])
    ]
    if min_recall is not None:
        if recalls is None:
            raise ValueError("recalls are required when min_recall is supplied.")
        scored = [
            (name, score)
            for name, score in scored
            if name in recalls and not np.isnan(recalls[name])
            and recalls[name] >= min_recall - 1e-12
        ]
    if not scored:
        if min_recall is not None:
            raise DataDiversityError(
                f"No candidate model met the recall floor of {min_recall:.3f}."
            )
        raise DataDiversityError(
            "No candidate model could be scored for selection. This needs accepted "
            "candidates of both classes in the held-out folds."
        )
    best = max(scored, key=lambda item: (item[1], -order.index(item[0])))
    return best[0]


def fit_stack(
    X: np.ndarray,
    y: np.ndarray,
    accepted: np.ndarray,
    groups: np.ndarray,
    config: Config,
    feature_names: Sequence[str],
    n_inner_splits: int | None = None,
    progress: Progress = None,
) -> StackFitResult:
    """Tune, cross-fit, and assemble the stack on one training set.

    Steps, in order: tune each base learner by inner grouped CV; build
    out-of-fold base probabilities; prepare independent base pipelines within
    each meta-training partition; tune and cross-fit the meta-model with those
    isolated inputs; choose the recall-floor threshold; then refit the
    meta-model on all accepted rows and the bases on all rows.
    """
    strata = composite_strata(y, accepted)
    cv = config.cv
    n_splits = n_inner_splits or resolve_n_splits(
        strata, groups, int(cv["inner_folds"]), int(cv["min_inner_folds"]), "inner"
    )
    splits = grouped_splits(strata, groups, n_splits, config.seed, bool(cv.get("shuffle", True)))

    tuning: dict[str, TuningResult] = {}
    for learner in config.base_learners:
        report_progress(progress, f"  tuning {learner} ({n_splits} inner folds)")
        tuning[learner] = tune_base_learner(
            learner, X, y, accepted, groups, splits, config, progress
        )
    best_params = {name: result.best_params for name, result in tuning.items()}

    report_progress(progress, "  building out-of-fold base probabilities")
    base_oof = base_oof_matrix(best_params, X, y, accepted, groups, splits, config)

    meta_folds = prepare_meta_folds(X, y, accepted, groups, splits, config, progress)

    report_progress(progress, "  tuning the meta-model")
    tuning["meta"] = tune_meta_learner(
        meta_folds, y, accepted, groups, config, progress
    )
    best_params["meta"] = tuning["meta"].best_params

    report_progress(progress, "  cross-fitting the meta-model")
    meta_crossfit = crossfit_meta_predictions(
        meta_folds, y, accepted, groups, config, best_params["meta"]
    )

    thresholds = _choose_thresholds(base_oof, meta_crossfit, y, accepted, groups, config)

    report_progress(progress, "  refitting base learners and meta-model on the full training set")
    accepted_rows = np.flatnonzero(accepted)
    meta = fit_meta(
        base_oof[accepted_rows],
        y[accepted_rows],
        groups[accepted_rows],
        config,
        best_params["meta"],
    )
    bases = {
        learner: fit_base(
            learner, best_params[learner], X, y, accepted, groups, config
        )
        for learner in config.base_learners
    }

    stack = FittedStack(
        bases=bases,
        meta=meta,
        thresholds={name: choice.threshold for name, choice in thresholds.items()},
        # Provisional: the caller selects the shipping model once it has scores
        # to compare. Evaluation folds never consult this field.
        selected_model=STACK,
        feature_names=tuple(feature_names),
    )
    return StackFitResult(
        stack=stack,
        best_params=best_params,
        tuning=tuning,
        base_oof=base_oof,
        meta_crossfit=meta_crossfit,
        thresholds=thresholds,
        n_inner_folds=n_splits,
        train_stars=tuple(sorted(set(np.asarray(groups, dtype=object).tolist()))),
    )


def _choose_thresholds(
    base_oof: np.ndarray,
    meta_crossfit: np.ndarray,
    y: np.ndarray,
    accepted: np.ndarray,
    groups: np.ndarray,
    config: Config,
) -> dict[str, ThresholdChoice]:
    """One recall-constrained threshold per model, from training rows only."""
    usable = accepted & ~np.isnan(meta_crossfit)
    rows = np.flatnonzero(usable)
    if rows.size == 0 or np.unique(y[rows]).size < 2:
        raise DataDiversityError(
            "No cross-fitted accepted candidates of both classes were available for "
            "threshold selection. Add stars with accepted CONFIRMED and accepted "
            "FALSE-POSITIVE candidates."
        )
    weights = star_balanced_weights(groups[rows]) if config.weighted_metrics else None

    sources: dict[str, np.ndarray] = {STACK: meta_crossfit}
    for column, learner in enumerate(config.base_learners):
        sources[learner] = base_oof[:, column]
    return {
        name: select_threshold(
            y[rows], probabilities[rows], config.min_recall, sample_weight=weights
        )
        for name, probabilities in sources.items()
    }
