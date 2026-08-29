"""Nested, star-grouped evaluation of the stack and its comparators."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, load_config
from .data import (
    Dataset,
    composite_strata,
    load_dataset,
    star_balanced_weights,
)
from .metrics import (
    BootstrapResult,
    accepted_average_precision,
    calibration_curve_points,
    compute_metrics,
    per_star_summary,
    star_bootstrap_intervals,
)
from .schema import FeatureSchema
from .stacking import (
    STACK,
    Progress,
    fit_stack,
    grouped_splits,
    model_probabilities,
    report_progress,
    reported_models,
    resolve_n_splits,
    select_best_model,
)


@dataclass
class EvaluationResult:
    """Outcome of the nested evaluation, ready to be written to disk."""

    n_outer_folds: int
    inner_folds_per_outer: list[int]
    fold_metrics: dict[str, list[dict[str, Any]]]
    fold_thresholds: dict[str, list[float]]
    pooled_metrics: dict[str, dict[str, Any]]
    bootstrap: dict[str, BootstrapResult]
    calibration: dict[str, list[dict[str, float]]]
    per_star: list[dict[str, Any]]
    oof_predictions: pd.DataFrame
    permutation_importance: pd.DataFrame
    fold_best_params: list[dict[str, Any]]
    dataset_summary: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def stack_meets_recall_floor(self) -> bool:
        return self.meets_recall_floor(STACK)

    def meets_recall_floor(self, model: str) -> bool:
        """Whether ``model`` held the recall floor on held-out stars."""
        floor = float(self.config.get("objective", {}).get("min_recall", 0.95))
        recall = self.pooled_metrics[model]["recall"]
        return bool(not np.isnan(recall) and recall >= floor - 1e-12)

    def selection_scores(self, metric: str = "average_precision") -> dict[str, float]:
        """Pooled held-out score per model, the basis for choosing what ships."""
        return {
            name: float(pooled.get(metric, float("nan")))
            for name, pooled in self.pooled_metrics.items()
        }

    def would_ship(self) -> str:
        """The model `train` would select from these results."""
        selection = self.config.get("selection", {})
        strategy = selection.get("strategy", "best")
        candidates = tuple(selection.get("candidates", reported_models()))
        if strategy != "best":
            return strategy
        return select_best_model(
            self.selection_scores(selection.get("metric", "average_precision")), candidates
        )

    def comparison_table(self) -> pd.DataFrame:
        """One row per reported model, ordered with the stack first."""
        selected = self.would_ship()
        rows = []
        for name in reported_models():
            pooled = self.pooled_metrics[name]
            ci = self.bootstrap[name].intervals
            rows.append(
                {
                    "model": name,
                    "is_production": name == selected,
                    "precision": pooled["precision"],
                    "precision_lo": ci.get("precision", {}).get("lower", float("nan")),
                    "precision_hi": ci.get("precision", {}).get("upper", float("nan")),
                    "recall": pooled["recall"],
                    "recall_lo": ci.get("recall", {}).get("lower", float("nan")),
                    "recall_hi": ci.get("recall", {}).get("upper", float("nan")),
                    "f2": pooled["f2"],
                    "average_precision": pooled["average_precision"],
                    "ap_lo": ci.get("average_precision", {}).get("lower", float("nan")),
                    "ap_hi": ci.get("average_precision", {}).get("upper", float("nan")),
                    "roc_auc": pooled["roc_auc"],
                    "brier_score": pooled["brier_score"],
                    "log_loss": pooled["log_loss"],
                    "n_flagged": pooled["n_flagged"],
                    "tp": pooled["confusion_matrix_counts"]["tp"],
                    "fp": pooled["confusion_matrix_counts"]["fp"],
                    "fn": pooled["confusion_matrix_counts"]["fn"],
                    "tn": pooled["confusion_matrix_counts"]["tn"],
                }
            )
        return pd.DataFrame(rows)

    def importance_summary(self, model: str | None = None) -> pd.DataFrame:
        """Permutation importance for one model, aggregated over folds and repeats.

        Defaults to the model that would ship, since that is the one whose
        feature reliance actually matters.
        """
        if self.permutation_importance.empty:
            return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
        frame = self.permutation_importance
        frame = frame[frame["model"] == (model or self.would_ship())]
        if frame.empty:
            return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
        grouped = frame.groupby("feature")["importance"]
        fold_means = (
            frame.groupby(["feature", "outer_fold"])["importance"].mean().groupby("feature")
        )
        summary = pd.DataFrame(
            {
                "importance_mean": grouped.mean(),
                "importance_std": grouped.std(ddof=1),
                "importance_fold_mean_std": fold_means.std(ddof=1),
                "importance_min": grouped.min(),
                "importance_max": grouped.max(),
                "n_samples": grouped.size(),
            }
        ).reset_index()
        return summary.sort_values("importance_mean", ascending=False).reset_index(drop=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_outer_folds": self.n_outer_folds,
            "inner_folds_per_outer": self.inner_folds_per_outer,
            "dataset": self.dataset_summary,
            "config": self.config,
            "would_ship": self.would_ship(),
            "selected_meets_recall_floor": self.meets_recall_floor(self.would_ship()),
            "stack_meets_recall_floor": self.stack_meets_recall_floor,
            "pooled_metrics": self.pooled_metrics,
            "fold_metrics": self.fold_metrics,
            "fold_thresholds": self.fold_thresholds,
            "bootstrap": {name: result.to_dict() for name, result in self.bootstrap.items()},
            "calibration": self.calibration,
            "per_star": self.per_star,
            "fold_best_params": self.fold_best_params,
        }


def _permutation_importance(
    result,
    X: np.ndarray,
    y: np.ndarray,
    accepted: np.ndarray,
    groups: np.ndarray,
    feature_names: tuple[str, ...],
    config: Config,
    fold: int,
) -> list[dict[str, Any]]:
    """Feature importance measured only on this fold's held-out stars."""
    settings = config.permutation_importance
    n_repeats = int(settings.get("n_repeats", 10))
    rows = np.flatnonzero(accepted)
    if rows.size == 0 or np.unique(y[rows]).size < 2:
        return []

    weights = star_balanced_weights(groups[rows]) if config.weighted_metrics else None

    def score(matrix: np.ndarray) -> dict[str, float]:
        """Accepted-candidate AP for every candidate model.

        All six share one pass of base-learner predictions, so scoring them
        together costs little more than scoring one.
        """
        probabilities = result.stack.all_probabilities(matrix)
        return {
            model: accepted_average_precision(
                y[rows], values[rows], sample_weight=weights
            )
            for model, values in probabilities.items()
        }

    baseline = score(X)
    if all(np.isnan(value) for value in baseline.values()):
        return []

    rng = np.random.default_rng(config.seed + fold)
    records: list[dict[str, Any]] = []
    for column, name in enumerate(feature_names):
        for repeat in range(n_repeats):
            shuffled = X.copy()
            shuffled[:, column] = X[rng.permutation(X.shape[0]), column]
            permuted = score(shuffled)
            for model, base_score in baseline.items():
                records.append(
                    {
                        "outer_fold": fold,
                        "model": model,
                        "feature": name,
                        "repeat": repeat,
                        "baseline_score": float(base_score),
                        "permuted_score": float(permuted[model]),
                        "importance": float(base_score - permuted[model]),
                    }
                )
    return records


def evaluate_dataset(
    data_dir: str | Path | None = None,
    config: Config | None = None,
    schema: FeatureSchema | None = None,
    dataset: Dataset | None = None,
    progress: Progress = None,
) -> EvaluationResult:
    """Run nested grouped cross-validation and assemble the full report."""
    config = config or load_config()
    if dataset is None:
        if data_dir is None:
            raise ValueError("Provide either data_dir or dataset.")
        dataset = load_dataset(data_dir, mode="train", schema=schema)

    y, accepted = dataset.require_supervision()
    X = dataset.X
    groups = dataset.star_id
    strata = composite_strata(y, accepted)
    feature_names = dataset.feature_names

    cv = config.cv
    n_outer = resolve_n_splits(
        strata, groups, int(cv["outer_folds"]), int(cv["min_outer_folds"]), "outer"
    )
    outer_splits = grouped_splits(
        strata, groups, n_outer, config.seed, bool(cv.get("shuffle", True))
    )
    report_progress(
        progress,
        f"nested evaluation: {n_outer} outer folds over {len(dataset.stars)} stars",
    )

    models = reported_models()
    fold_metrics: dict[str, list[dict[str, Any]]] = {name: [] for name in models}
    fold_thresholds: dict[str, list[float]] = {name: [] for name in models}
    inner_folds_per_outer: list[int] = []
    fold_best_params: list[dict[str, Any]] = []
    importance_records: list[dict[str, Any]] = []
    prediction_blocks: list[pd.DataFrame] = []

    for fold, (train_idx, val_idx) in enumerate(outer_splits):
        report_progress(
            progress,
            f"outer fold {fold + 1}/{n_outer}: "
            f"{np.unique(groups[train_idx]).size} train stars, "
            f"{np.unique(groups[val_idx]).size} held-out stars",
        )
        result = fit_stack(
            X[train_idx],
            y[train_idx],
            accepted[train_idx],
            groups[train_idx],
            config,
            feature_names,
            progress=progress,
        )
        inner_folds_per_outer.append(result.n_inner_folds)
        fold_best_params.append({"outer_fold": fold, "best_params": result.best_params})

        probabilities = model_probabilities(result, X[val_idx])
        accepted_val = np.flatnonzero(accepted[val_idx])
        val_rows = val_idx[accepted_val]
        weights = (
            star_balanced_weights(groups[val_rows]) if config.weighted_metrics else None
        )

        block = pd.DataFrame(
            {
                "outer_fold": fold,
                "row": val_rows,
                "star_id": groups[val_rows],
                "source_file": dataset.source_file[val_rows],
                "row_index": dataset.row_index[val_rows],
                "y_true": y[val_rows],
            }
        )
        for name in models:
            threshold = result.thresholds[name].threshold
            fold_thresholds[name].append(float(threshold))
            probability = probabilities[name][accepted_val]
            block[f"prob_{name}"] = probability
            block[f"threshold_{name}"] = threshold
            if val_rows.size and np.unique(y[val_rows]).size >= 1:
                fold_metrics[name].append(
                    {
                        "outer_fold": fold,
                        "n_held_out_stars": int(np.unique(groups[val_rows]).size),
                        **compute_metrics(y[val_rows], probability, threshold, weights),
                    }
                )
        prediction_blocks.append(block)

        importance_records.extend(
            _permutation_importance(
                result,
                X[val_idx],
                y[val_idx],
                accepted[val_idx],
                groups[val_idx],
                feature_names,
                config,
                fold,
            )
        )

    oof = pd.concat(prediction_blocks, ignore_index=True).sort_values(
        ["source_file", "row_index"], kind="stable"
    ).reset_index(drop=True)

    report_progress(progress, "pooling outer-fold predictions and bootstrapping over stars")
    pooled_weights = (
        star_balanced_weights(oof["star_id"].to_numpy(dtype=object))
        if config.weighted_metrics
        else None
    )
    y_pooled = oof["y_true"].to_numpy(dtype=int)
    stars_pooled = oof["star_id"].to_numpy(dtype=object)

    pooled_metrics: dict[str, dict[str, Any]] = {}
    bootstrap: dict[str, BootstrapResult] = {}
    calibration: dict[str, list[dict[str, float]]] = {}
    boot_cfg = config.bootstrap
    for name in models:
        probability = oof[f"prob_{name}"].to_numpy(dtype=float)
        thresholds = oof[f"threshold_{name}"].to_numpy(dtype=float)
        pooled_metrics[name] = compute_metrics(y_pooled, probability, thresholds, pooled_weights)
        calibration[name] = calibration_curve_points(
            y_pooled, probability, config.calibration_bins, pooled_weights
        )
        bootstrap[name] = star_bootstrap_intervals(
            stars_pooled,
            y_pooled,
            probability,
            thresholds,
            n_resamples=int(boot_cfg.get("n_resamples", 1000)),
            confidence_level=float(boot_cfg.get("confidence_level", 0.95)),
            seed=config.seed,
            weighted=config.weighted_metrics,
        )

    stack_decision = (
        oof[f"prob_{STACK}"].to_numpy(dtype=float)
        >= oof[f"threshold_{STACK}"].to_numpy(dtype=float)
    ).astype(int)
    oof["prediction"] = np.where(stack_decision == 1, "POTENTIAL", "UNLIKELY")

    # Per-star breakdown for every reported model, so a weak fold can be traced
    # to the stars behind it.
    per_star: list[dict[str, Any]] = []
    for name in models:
        decision = (
            oof[f"prob_{name}"].to_numpy(dtype=float)
            >= oof[f"threshold_{name}"].to_numpy(dtype=float)
        ).astype(int)
        for row in per_star_summary(
            stars_pooled, y_pooled, oof[f"prob_{name}"].to_numpy(dtype=float), decision
        ):
            per_star.append({"model": name, **row})

    return EvaluationResult(
        n_outer_folds=n_outer,
        inner_folds_per_outer=inner_folds_per_outer,
        fold_metrics=fold_metrics,
        fold_thresholds=fold_thresholds,
        pooled_metrics=pooled_metrics,
        bootstrap=bootstrap,
        calibration=calibration,
        per_star=per_star,
        oof_predictions=oof,
        permutation_importance=pd.DataFrame(
            importance_records,
            columns=[
                "outer_fold",
                "model",
                "feature",
                "repeat",
                "baseline_score",
                "permuted_score",
                "importance",
            ],
        ),
        fold_best_params=fold_best_params,
        dataset_summary=dataset.summary(),
        config=config.to_dict(),
    )
