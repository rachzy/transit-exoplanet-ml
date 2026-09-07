"""Final model fitting, the serialized bundle, and training artifacts."""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import plots
from .artifacts import (
    new_run_id,
    prepare_artifact_dir,
    provenance,
    write_csv,
    write_json,
    write_yaml,
)
from .config import SELECTION_STRATEGY_BEST, Config, load_config
from .data import (
    Dataset,
    composite_strata,
    load_dataset,
    star_balanced_weights,
)
from .errors import SchemaVersionError
from .evaluate import EvaluationResult, evaluate_dataset
from .metrics import (
    ThresholdChoice,
    compute_metrics,
    per_star_summary,
)
from .schema import FeatureSchema, load_schema
from .stacking import (
    STACK,
    FittedStack,
    Progress,
    StackFitResult,
    fit_stack,
    report_progress,
    reported_models,
    resolve_n_splits,
    select_best_model,
)

BUNDLE_FORMAT_VERSION = 3
BUNDLE_FILENAME = "model.joblib"

POTENTIAL = "POTENTIAL"
UNLIKELY = "UNLIKELY"


@dataclass
class SelectionRecord:
    """Why a particular model was chosen to ship."""

    selected_model: str
    strategy: str
    metric: str
    score_source: str
    scores: dict[str, float]
    candidates: tuple[str, ...]
    recalls: dict[str, float] | None = None
    min_recall: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_model": self.selected_model,
            "strategy": self.strategy,
            "metric": self.metric,
            "score_source": self.score_source,
            "candidates": list(self.candidates),
            "scores": self.scores,
            "recalls": getattr(self, "recalls", None),
            "min_recall": getattr(self, "min_recall", None),
            "runner_up": self.runner_up,
            "margin": self.margin,
        }

    @property
    def ranked(self) -> list[tuple[str, float]]:
        recalls = getattr(self, "recalls", None)
        min_recall = getattr(self, "min_recall", None)
        usable = [
            (n, s)
            for n, s in self.scores.items()
            if not np.isnan(s)
            and (
                min_recall is None
                or (
                    recalls is not None
                    and n in recalls
                    and not np.isnan(recalls[n])
                    and recalls[n] >= min_recall - 1e-12
                )
            )
        ]
        return sorted(usable, key=lambda item: -item[1])

    @property
    def runner_up(self) -> str | None:
        others = [n for n, _ in self.ranked if n != self.selected_model]
        return others[0] if others else None

    @property
    def margin(self) -> float | None:
        """How far the winner beat the next-best candidate."""
        runner_up = self.runner_up
        if runner_up is None:
            return None
        return float(self.scores[self.selected_model] - self.scores[runner_up])


@dataclass
class ModelBundle:
    """A trained model plus everything needed to reproduce and audit it.

    Every candidate is fitted and retained; ``selection`` records which one is
    served by :meth:`predict_proba` and why.
    """

    run_id: str
    created_at: str
    package_version: str
    schema: FeatureSchema
    config: dict[str, Any]
    stack: FittedStack
    threshold_choice: ThresholdChoice
    best_params: dict[str, dict[str, Any]]
    train_stars: tuple[str, ...]
    file_checksums: dict[str, str]
    provenance: dict[str, Any]
    n_oof_folds: int
    selection: SelectionRecord
    bundle_format_version: int = BUNDLE_FORMAT_VERSION

    @property
    def threshold(self) -> float:
        return float(self.stack.threshold)

    @property
    def selected_model(self) -> str:
        """The model that produces ``potential_probability``."""
        return self.stack.selected_model

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.stack.feature_names

    # -- inference --------------------------------------------------------
    def base_probabilities(self, X: np.ndarray) -> dict[str, np.ndarray]:
        matrix = self.stack.base_matrix(np.asarray(X, dtype=float))
        return {name: matrix[:, i] for i, name in enumerate(self.stack.base_learners)}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.stack.predict_proba(np.asarray(X, dtype=float))

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return np.where(probabilities >= self.threshold, POTENTIAL, UNLIKELY)

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        target = Path(path)
        if target.is_dir():
            target = target / BUNDLE_FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "bundle_format_version": self.bundle_format_version,
                "schema_version": self.schema.schema_version,
                "package_version": self.package_version,
                "run_id": self.run_id,
                "created_at": self.created_at,
                "schema": self.schema,
                "config": self.config,
                "stack": self.stack,
                "threshold_choice": self.threshold_choice,
                "best_params": self.best_params,
                "train_stars": self.train_stars,
                "file_checksums": self.file_checksums,
                "provenance": self.provenance,
                "n_oof_folds": self.n_oof_folds,
                "selection": self.selection,
            },
            target,
            compress=3,
        )
        return target

    @classmethod
    def load(cls, path: str | Path, schema: FeatureSchema | None = None) -> ModelBundle:
        """Load a bundle, refusing one built against an incompatible schema.

        Bundles are Joblib pickles: load only bundles you or your pipeline
        produced, from a location you trust.
        """
        source = Path(path)
        if source.is_dir():
            source = source / BUNDLE_FILENAME
        if not source.is_file():
            raise FileNotFoundError(
                f"No model bundle at {source}. Point --model-dir at a directory "
                f"containing {BUNDLE_FILENAME}."
            )
        payload = joblib.load(source)

        found = int(payload.get("bundle_format_version", -1))
        if found != BUNDLE_FORMAT_VERSION:
            raise SchemaVersionError(
                f"{source} uses bundle format version {found}; this build reads "
                f"version {BUNDLE_FORMAT_VERSION}. Retrain with the current package."
            )

        current = schema or load_schema()
        stored: FeatureSchema = payload["schema"]
        if stored.schema_version != current.schema_version:
            raise SchemaVersionError(
                f"{source} was trained against feature schema version "
                f"{stored.schema_version}; the installed schema is version "
                f"{current.schema_version}. Retrain the model or pin the matching "
                "package release."
            )
        if tuple(stored.feature_columns) != tuple(current.feature_columns):
            raise SchemaVersionError(
                f"{source} was trained on a different feature set than the installed "
                f"schema version {current.schema_version}, despite the matching version "
                "number. Bump schema_version whenever feature_columns changes."
            )

        return cls(
            run_id=payload["run_id"],
            created_at=payload["created_at"],
            package_version=payload["package_version"],
            schema=stored,
            config=payload["config"],
            stack=payload["stack"],
            threshold_choice=payload["threshold_choice"],
            best_params=payload["best_params"],
            train_stars=tuple(payload["train_stars"]),
            file_checksums=dict(payload["file_checksums"]),
            provenance=payload["provenance"],
            n_oof_folds=int(payload["n_oof_folds"]),
            selection=payload["selection"],
            bundle_format_version=found,
        )


def load_model(path: str | Path, schema: FeatureSchema | None = None) -> ModelBundle:
    """Load a serialized model bundle from a file or artifact directory."""
    return ModelBundle.load(path, schema=schema)


@dataclass
class TrainingRun:
    """The fitted bundle, its self-report, and where it was written."""

    bundle: ModelBundle
    fit: StackFitResult
    report: dict[str, Any]
    oof_frame: pd.DataFrame
    evaluation: EvaluationResult | None = None
    artifact_dir: Path | None = None
    written: list[Path] = field(default_factory=list)


def _training_oof_frame(fit: StackFitResult, dataset: Dataset) -> pd.DataFrame:
    """Cross-fitted training predictions, one row per candidate."""
    y, accepted = dataset.require_supervision()
    probabilities = fit.training_probabilities()
    frame = pd.DataFrame(
        {
            "source_file": dataset.source_file,
            "star_id": dataset.star_id,
            "row_index": dataset.row_index,
            "detection_status": dataset.status,
            "accepted": accepted,
            "y_true": y,
        }
    )
    for name in fit.stack.base_learners:
        frame[f"prob_{name}"] = probabilities[name]
    frame["prob_stack"] = probabilities[STACK]

    selected = fit.stack.selected_model
    frame["model_name"] = selected
    frame["potential_probability"] = probabilities[selected]
    frame["decision_threshold"] = fit.threshold
    flagged = accepted & (probabilities[selected] >= fit.threshold)
    frame["prediction"] = np.where(
        ~accepted, "", np.where(flagged, POTENTIAL, UNLIKELY)
    )
    return frame.sort_values(["source_file", "row_index"], kind="stable").reset_index(drop=True)


def _training_report(
    fit: StackFitResult,
    dataset: Dataset,
    config: Config,
) -> dict[str, Any]:
    """Cross-fitted training diagnostics for the selected operating point."""
    y, accepted = dataset.require_supervision()
    groups = dataset.star_id
    probabilities = fit.training_probabilities()

    usable = accepted & ~np.isnan(probabilities[STACK])
    rows = np.flatnonzero(usable)
    weights = star_balanced_weights(groups[rows]) if config.weighted_metrics else None

    per_model: dict[str, Any] = {}
    for name in reported_models(config):
        threshold = fit.thresholds[name].threshold
        per_model[name] = {
            "threshold_choice": fit.thresholds[name].to_dict(),
            "crossfit_metrics": compute_metrics(
                y[rows], probabilities[name][rows], threshold, weights
            ),
        }

    selected = fit.stack.selected_model
    selected_probability = probabilities[selected][rows]
    decision = (selected_probability >= fit.threshold).astype(int)
    return {
        "dataset": dataset.summary(),
        "n_oof_folds": fit.n_inner_folds,
        "selected_model": selected,
        "saved_threshold": fit.threshold,
        "models": per_model,
        "best_params": fit.best_params,
        "per_star": per_star_summary(groups[rows], y[rows], selected_probability, decision),
        "n_crossfit_rows": int(rows.size),
    }


def _crossfit_scores(
    fit: StackFitResult, dataset: Dataset, config: Config, metric: str
) -> dict[str, float]:
    """Selection scores from the training set's own cross-fitted predictions."""
    y, accepted = dataset.require_supervision()
    groups = dataset.star_id
    probabilities = fit.training_probabilities()
    rows = np.flatnonzero(accepted & ~np.isnan(probabilities[STACK]))
    weights = star_balanced_weights(groups[rows]) if config.weighted_metrics else None
    return {
        name: float(
            compute_metrics(
                y[rows], probabilities[name][rows], fit.thresholds[name].threshold, weights
            ).get(metric, float("nan"))
        )
        for name in reported_models(config)
    }


def _crossfit_recalls(
    fit: StackFitResult, dataset: Dataset, config: Config
) -> dict[str, float]:
    """Recall of each candidate at its cross-fitted recall-floor threshold."""
    y, accepted = dataset.require_supervision()
    groups = dataset.star_id
    probabilities = fit.training_probabilities()
    rows = np.flatnonzero(accepted & ~np.isnan(probabilities[STACK]))
    weights = star_balanced_weights(groups[rows]) if config.weighted_metrics else None
    return {
        name: float(
            compute_metrics(
                y[rows], probabilities[name][rows], fit.thresholds[name].threshold, weights
            ).get("recall", float("nan"))
        )
        for name in reported_models(config)
    }


def choose_model(
    config: Config,
    fit: StackFitResult,
    dataset: Dataset,
    evaluation: EvaluationResult | None,
) -> SelectionRecord:
    """Decide which fitted model ships.

    With ``selection.strategy: best`` the highest scorer on the configured
    metric wins. Scores come from the nested evaluation's pooled held-out
    numbers when it was run -- the honest estimate -- and fall back to the
    training set's own cross-fitted scores when it was skipped.
    """
    settings = config.selection
    strategy = str(settings["strategy"])
    metric = str(settings["metric"])
    candidates = config.selection_candidates
    preferred = str(settings["score_source"])

    if preferred == "nested_evaluation" and evaluation is not None:
        scores = evaluation.selection_scores(metric)
        recalls = evaluation.selection_recalls()
        source = "nested_evaluation"
    else:
        scores = _crossfit_scores(fit, dataset, config, metric)
        recalls = _crossfit_recalls(fit, dataset, config)
        source = "crossfit"

    selected = (
        strategy
        if strategy != SELECTION_STRATEGY_BEST
        else select_best_model(
            scores,
            candidates,
            recalls=recalls,
            min_recall=config.min_recall,
        )
    )
    return SelectionRecord(
        selected_model=selected,
        strategy=strategy,
        metric=metric,
        score_source=source,
        scores=scores,
        candidates=candidates,
        recalls=recalls,
        min_recall=config.min_recall if strategy == SELECTION_STRATEGY_BEST else None,
    )


def train_model(
    data_dir: str | Path | None = None,
    config: Config | None = None,
    schema: FeatureSchema | None = None,
    dataset: Dataset | None = None,
    run_evaluation: bool = True,
    evaluation: EvaluationResult | None = None,
    progress: Progress = None,
) -> TrainingRun:
    """Tune, cross-fit, and fit the production stack on the whole dataset."""
    config = config or load_config()
    if dataset is None:
        if data_dir is None:
            raise ValueError("Provide either data_dir or dataset.")
        dataset = load_dataset(data_dir, mode="train", schema=schema)

    if run_evaluation and evaluation is None:
        report_progress(progress, "running nested evaluation before final training")
        evaluation = evaluate_dataset(dataset=dataset, config=config, progress=progress)

    y, accepted = dataset.require_supervision()
    groups = dataset.star_id
    strata = composite_strata(y, accepted)
    n_folds = resolve_n_splits(
        strata,
        groups,
        int(config.cv.get("final_oof_folds", config.cv["inner_folds"])),
        int(config.cv["min_inner_folds"]),
        "final out-of-fold",
    )

    report_progress(progress, f"final training on {len(dataset.stars)} stars ({n_folds} OOF folds)")
    fit = fit_stack(
        dataset.X,
        y,
        accepted,
        groups,
        config,
        dataset.feature_names,
        n_inner_splits=n_folds,
        progress=progress,
    )

    selection = choose_model(config, fit, dataset, evaluation)
    fit.stack = fit.stack.with_selection(selection.selected_model)
    report_progress(
        progress,
        f"selected {selection.selected_model} by {selection.metric} "
        f"({selection.score_source})",
    )

    run_id = new_run_id()
    bundle = ModelBundle(
        run_id=run_id,
        created_at=dt.datetime.now(dt.UTC).isoformat(),
        package_version=_package_version(),
        schema=dataset.schema,
        config=config.to_dict(),
        stack=fit.stack,
        threshold_choice=fit.thresholds[selection.selected_model],
        best_params=fit.best_params,
        train_stars=tuple(dataset.stars),
        file_checksums=dict(dataset.file_checksums),
        provenance=provenance(config.seed, run_id=run_id),
        n_oof_folds=n_folds,
        selection=selection,
    )
    return TrainingRun(
        bundle=bundle,
        fit=fit,
        report=_training_report(fit, dataset, config),
        oof_frame=_training_oof_frame(fit, dataset),
        evaluation=evaluation,
    )


def _package_version() -> str:
    from . import __version__

    return __version__


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_evaluation_artifacts(
    result: EvaluationResult,
    directory: Path,
    seed: int,
    run_id: str | None = None,
) -> list[Path]:
    """Write metrics, per-fold detail, OOF predictions, importance, and plots."""
    directory.mkdir(parents=True, exist_ok=True)
    written = [
        write_json(directory / "metrics.json", result.to_dict()),
        write_csv(directory / "model_comparison.csv", result.comparison_table()),
        write_csv(directory / "oof_predictions.csv", result.oof_predictions),
        write_csv(directory / "per_star.csv", pd.DataFrame(result.per_star)),
        write_csv(directory / "permutation_importance.csv", result.permutation_importance),
        write_csv(
            directory / "permutation_importance_summary.csv", result.importance_summary()
        ),
        write_json(
            directory / "provenance.json", provenance(seed, run_id=run_id)
        ),
    ]
    written.extend(plots.write_all(result, directory / "plots"))
    return written


def write_training_artifacts(
    run: TrainingRun,
    artifact_base: str | Path,
    overwrite: bool = False,
) -> TrainingRun:
    """Materialise one immutable artifact directory for a training run."""
    directory = prepare_artifact_dir(artifact_base, run.bundle.run_id, overwrite=overwrite)
    bundle = run.bundle
    written: list[Path] = [bundle.save(directory / "model.joblib")]

    written.append(write_yaml(directory / "config.resolved.yaml", bundle.config))
    written.append(write_yaml(directory / "feature_schema.yaml", bundle.schema.to_dict()))
    written.append(write_json(directory / "selection.json", bundle.selection.to_dict()))
    written.append(
        write_json(
            directory / "threshold.json",
            {
                "selected_model": bundle.selected_model,
                "threshold": bundle.threshold,
                "objective": bundle.config["objective"],
                "selection": bundle.threshold_choice.to_dict(),
                "comparator_thresholds": {
                    name: choice.to_dict() for name, choice in run.fit.thresholds.items()
                },
            },
        )
    )
    written.append(write_json(directory / "training_report.json", run.report))
    written.append(
        write_json(
            directory / "tuning.json",
            {name: result.to_dict() for name, result in run.fit.tuning.items()},
        )
    )
    written.append(
        write_json(
            directory / "training_stars.json",
            {
                "n_stars": len(bundle.train_stars),
                "stars": list(bundle.train_stars),
                "input_file_checksums": bundle.file_checksums,
            },
        )
    )
    written.append(write_json(directory / "provenance.json", bundle.provenance))

    written.append(
        write_csv(directory / "training_oof_predictions.csv", run.oof_frame)
    )
    written.append(write_csv(directory / "per_star.csv", pd.DataFrame(run.report["per_star"])))

    if run.evaluation is not None:
        written.extend(
            write_evaluation_artifacts(
                run.evaluation,
                directory / "evaluation",
                seed=int(bundle.config["seed"]),
                run_id=bundle.run_id,
            )
        )

    written.append(write_json(directory / "summary.json", _summary(run)))
    manifest = {
        "run_id": bundle.run_id,
        "files": {
            str(path.relative_to(directory)): {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(written)
            if path.is_file()
        },
    }
    write_json(directory / "MANIFEST.json", manifest)

    run.artifact_dir = directory
    run.written = written
    return run


def _summary(run: TrainingRun) -> dict[str, Any]:
    selection = run.bundle.selection
    selected = run.report["models"][selection.selected_model]["crossfit_metrics"]
    payload: dict[str, Any] = {
        "run_id": run.bundle.run_id,
        "created_at": run.bundle.created_at,
        "package_version": run.bundle.package_version,
        "schema_version": run.bundle.schema.schema_version,
        "n_train_stars": len(run.bundle.train_stars),
        "n_features": len(run.bundle.feature_names),
        "selected_model": selection.selected_model,
        "selection": selection.to_dict(),
        "threshold": run.bundle.threshold,
        "crossfit_precision": selected["precision"],
        "crossfit_recall": selected["recall"],
        "crossfit_average_precision": selected["average_precision"],
        "best_params": run.bundle.best_params,
    }
    if run.evaluation is not None:
        pooled = run.evaluation.pooled_metrics
        payload["nested_evaluation"] = {
            "n_outer_folds": run.evaluation.n_outer_folds,
            "models": {
                name: {
                    "precision": pooled[name]["precision"],
                    "recall": pooled[name]["recall"],
                    "f2": pooled[name]["f2"],
                    "average_precision": pooled[name]["average_precision"],
                    "roc_auc": pooled[name]["roc_auc"],
                }
                for name in reported_models(config)
            },
            "production_model": selection.selected_model,
        }
    return payload
