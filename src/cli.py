"""The ``exoplanet-ml`` command line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from . import __version__
from .config import load_config
from .data import load_dataset, validate_dataset
from .errors import TransitExoplanetMLError
from .evaluate import evaluate_dataset
from .predict import predict_dataset, prediction_counts
from .schema import load_schema
from .stacking import reported_models
from .training import (
    train_model,
    write_evaluation_artifacts,
    write_training_artifacts,
)

app = typer.Typer(
    name="exoplanet-ml",
    help=(
        "Screen transit exoplanet candidates with a star-grouped stacked classifier. "
        "potential_probability is a screening score, not scientific confirmation."
    ),
    no_args_is_help=True,
    add_completion=False,
)

DataDir = Annotated[
    Path, typer.Option("--data-dir", help="Directory of <star-name>_<YYYYMMDD>.csv files.")
]
ConfigOpt = Annotated[
    Path | None, typer.Option("--config", help="Override the packaged configuration YAML.")
]
SchemaOpt = Annotated[
    Path | None, typer.Option("--schema", help="Override the packaged feature schema YAML.")
]
QuietOpt = Annotated[bool, typer.Option("--quiet", help="Suppress progress output.")]


def _echo(message: str) -> None:
    typer.echo(message)


def _progress(quiet: bool):
    if quiet:
        return None

    def emit(message: str) -> None:
        typer.echo(message, err=True)

    return emit


def _fail(exc: Exception) -> None:
    typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    try:
        return "n/a" if value != value else f"{value:.4f}"  # NaN check
    except TypeError:  # pragma: no cover - defensive
        return str(value)


@app.callback(invoke_without_command=True)
def main(
    version: Annotated[
        bool, typer.Option("--version", help="Print the package version and exit.")
    ] = False,
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def validate(
    data_dir: DataDir,
    mode: Annotated[
        str, typer.Option("--mode", help="train requires labels and statuses; predict does not.")
    ] = "train",
    schema: SchemaOpt = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit the summary as JSON.")] = False,
) -> None:
    """Check a data directory against the versioned data contract."""
    try:
        summary = validate_dataset(data_dir, mode=mode, schema=load_schema(schema))
    except TransitExoplanetMLError as exc:
        _fail(exc)
        return

    if as_json:
        _echo(json.dumps(summary, indent=2))
        return

    typer.secho(f"OK  {data_dir} passes the {mode} contract.", fg=typer.colors.GREEN)
    _echo(f"  schema version : {summary['schema_version']}")
    _echo(f"  files / stars  : {summary['n_files']} / {summary['n_stars']}")
    _echo(f"  candidate rows : {summary['n_rows']}")
    _echo(f"  model features : {summary['n_features']}")
    if "n_accepted" in summary:
        _echo(
            f"  labels         : {summary['n_positive']} CONFIRMED / "
            f"{summary['n_negative']} FALSE-POSITIVE"
        )
        _echo(
            f"  accepted       : {summary['n_accepted']} "
            f"({summary['n_accepted_positive']} CONFIRMED / "
            f"{summary['n_accepted_negative']} FALSE-POSITIVE)"
        )
        _echo(f"  status counts  : {summary['status_counts']}")


@app.command()
def evaluate(
    data_dir: DataDir,
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Where the evaluation report is written.")
    ],
    config: ConfigOpt = None,
    schema: SchemaOpt = None,
    quiet: QuietOpt = False,
) -> None:
    """Run nested, star-grouped evaluation and write the full report."""
    try:
        resolved = load_config(config)
        dataset = load_dataset(data_dir, mode="train", schema=load_schema(schema))
        result = evaluate_dataset(
            dataset=dataset, config=resolved, progress=_progress(quiet)
        )
        written = write_evaluation_artifacts(
            result, Path(output_dir), seed=resolved.seed
        )
    except TransitExoplanetMLError as exc:
        _fail(exc)
        return

    _echo("")
    _echo(
        f"Nested evaluation: {result.n_outer_folds} outer folds, "
        f"inner folds {result.inner_folds_per_outer}, "
        f"{result.dataset_summary['n_stars']} stars, "
        f"{result.dataset_summary['n_rows']} candidates."
    )
    metric = resolved.selection["metric"]
    scores = result.selection_scores(metric)
    would_ship = result.would_ship()

    _echo("")
    _echo(f"{'model':22s} {'precision':>10s} {'recall':>8s} {'F2':>8s} {'AP':>8s} {'ROC-AUC':>8s}")
    for name in reported_models(resolved):
        metrics = result.pooled_metrics[name]
        marker = " *" if name == would_ship else "  "
        _echo(
            f"{name:20s}{marker} {_fmt(metrics['precision']):>10s} "
            f"{_fmt(metrics['recall']):>8s} {_fmt(metrics['f2']):>8s} "
            f"{_fmt(metrics['average_precision']):>8s} {_fmt(metrics['roc_auc']):>8s}"
        )
    _echo("")
    _echo(f"* highest {metric}; `train` would ship this model")

    ranked = sorted(
        ((n, s) for n, s in scores.items() if s == s), key=lambda row: -row[1]
    )
    margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else None
    _report_selection_bias(margin, len(resolved.selection_candidates))

    _echo(f"\nWrote {len(written)} files to {output_dir}")


def _report_selection_bias(margin: float | None, n_candidates: int) -> None:
    """Warn that a metric used to choose the winner also over-rates it."""
    typer.secho(
        f"NOTE: the winner was chosen on the same held-out score that is reported for "
        f"it. Taking the best of {n_candidates} correlated estimates biases that score "
        "upward, so treat it as an optimistic estimate of future performance.",
        fg=typer.colors.YELLOW,
    )
    if margin is not None and margin < 0.02:
        typer.secho(
            f"NOTE: the winning margin over the runner-up is only {margin:.4f}. That is "
            "well inside the bootstrap intervals, so the ranking is not stable -- a "
            "different sample of stars would likely pick a different model.",
            fg=typer.colors.YELLOW,
        )


@app.command()
def train(
    data_dir: DataDir,
    artifact_dir: Annotated[
        Path,
        typer.Option(
            "--artifact-dir",
            help="Base directory; a run-id subdirectory is created inside it.",
        ),
    ],
    config: ConfigOpt = None,
    schema: SchemaOpt = None,
    skip_evaluation: Annotated[
        bool,
        typer.Option(
            "--skip-evaluation",
            help="Fit without the nested evaluation. The artifact then lacks honest metrics.",
        ),
    ] = False,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Allow writing into a non-empty run directory.")
    ] = False,
    quiet: QuietOpt = False,
) -> None:
    """Train the production stack and save an immutable artifact directory."""
    try:
        resolved = load_config(config)
        dataset = load_dataset(data_dir, mode="train", schema=load_schema(schema))
        run = train_model(
            dataset=dataset,
            config=resolved,
            run_evaluation=not skip_evaluation,
            progress=_progress(quiet),
        )
        run = write_training_artifacts(run, artifact_dir, overwrite=overwrite)
    except TransitExoplanetMLError as exc:
        _fail(exc)
        return

    report = run.report
    selection = run.bundle.selection
    selected = report["models"][selection.selected_model]["crossfit_metrics"]
    _echo("")
    typer.secho(f"Trained run {run.bundle.run_id}", fg=typer.colors.GREEN)
    _echo(f"  stars / candidates : {len(run.bundle.train_stars)} / {report['dataset']['n_rows']}")
    typer.secho(
        f"  shipping model     : {selection.selected_model}", fg=typer.colors.GREEN, bold=True
    )
    _echo(
        f"  chosen by          : {selection.metric} on {selection.score_source} "
        f"(strategy: {selection.strategy})"
    )
    _echo("  ranking            : " + ", ".join(
        f"{name} {_fmt(score)}" for name, score in selection.ranked
    ))
    _echo(f"  saved threshold    : {run.bundle.threshold:.6f}")
    _echo(
        f"  cross-fitted       : precision {_fmt(selected['precision'])}, "
        f"recall {_fmt(selected['recall'])}, AP {_fmt(selected['average_precision'])}"
    )
    _echo(f"  artifacts          : {run.artifact_dir} ({len(run.written)} files)")
    _echo("")
    if selection.strategy == "best":
        _report_selection_bias(selection.margin, len(selection.candidates))


@app.command()
def predict(
    model_dir: Annotated[
        Path, typer.Option("--model-dir", help="Artifact directory or path to model.joblib.")
    ],
    data_dir: DataDir,
    output: Annotated[
        Path, typer.Option("--output", help="Destination CSV for the consolidated predictions.")
    ] = Path("predictions.csv"),
    schema: SchemaOpt = None,
) -> None:
    """Score unseen candidate files into one consolidated prediction CSV."""
    try:
        frame = predict_dataset(
            model_dir,
            data_dir=data_dir,
            output=output,
            schema=load_schema(schema) if schema else None,
        )
    except (TransitExoplanetMLError, FileNotFoundError) as exc:
        _fail(exc)
        return

    counts = prediction_counts(frame)
    typer.secho(f"Wrote {counts['n_rows']} predictions to {output}", fg=typer.colors.GREEN)
    _echo(f"  stars     : {counts['n_stars']}")
    _echo(f"  POTENTIAL : {counts['POTENTIAL']}")
    _echo(f"  UNLIKELY  : {counts['UNLIKELY']}")
    _echo(f"  threshold : {frame['decision_threshold'].iloc[0]:.6f}")
    _echo("")
    _echo(
        "POTENTIAL is a screening flag from a model trained on literature-derived "
        "labels. It is not a confirmation."
    )


if __name__ == "__main__":  # pragma: no cover
    app()
