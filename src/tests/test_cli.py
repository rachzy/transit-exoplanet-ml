"""The exoplanet-ml command line surface."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from src import __version__
from src.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def config_file(fast_config, tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("config") / "fast.yaml"
    path.write_text(yaml.safe_dump(fast_config.to_dict(), sort_keys=False))
    return path


def run(*args: str):
    return runner.invoke(app, list(args))


def test_version():
    result = run("--version")
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_no_arguments_lists_every_command():
    result = run()
    for command in ("validate", "evaluate", "train", "predict"):
        assert command in result.stdout


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------
def test_validate_train_mode(train_dir):
    result = run("validate", "--data-dir", str(train_dir), "--mode", "train")
    assert result.exit_code == 0
    assert "passes the train contract" in result.stdout
    assert "accepted" in result.stdout


def test_validate_predict_mode(predict_dir):
    result = run("validate", "--data-dir", str(predict_dir), "--mode", "predict")
    assert result.exit_code == 0
    assert "passes the predict contract" in result.stdout


def test_validate_json_output(train_dir):
    result = run("validate", "--data-dir", str(train_dir), "--mode", "train", "--json")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["n_stars"] == 12


def test_validate_reports_a_clean_error_on_empty_directory(tmp_path):
    result = run("validate", "--data-dir", str(tmp_path))
    assert result.exit_code == 1
    assert "No candidate CSV files found" in result.stderr
    assert "<star-name>_<YYYYMMDD>.csv" in result.stderr
    assert "Traceback" not in result.stderr


def test_validate_rejects_unlabelled_data_in_train_mode(predict_dir):
    result = run("validate", "--data-dir", str(predict_dir), "--mode", "train")
    assert result.exit_code == 1
    assert "candidate_label" in result.stderr
    assert "detection_status" in result.stderr


# ---------------------------------------------------------------------------
# evaluate / train / predict
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def evaluate_output(train_dir, config_file, tmp_path_factory) -> Path:
    output = tmp_path_factory.mktemp("cli-eval") / "report"
    result = run(
        "evaluate",
        "--data-dir", str(train_dir),
        "--output-dir", str(output),
        "--config", str(config_file),
        "--quiet",
    )
    assert result.exit_code == 0, result.stdout
    return output


def test_evaluate_writes_metrics_and_plots(evaluate_output):
    for name in (
        "metrics.json",
        "model_comparison.csv",
        "oof_predictions.csv",
        "per_star.csv",
        "permutation_importance.csv",
        "permutation_importance_summary.csv",
        "provenance.json",
    ):
        assert (evaluate_output / name).is_file(), name
    plots = sorted(p.name for p in (evaluate_output / "plots").glob("*.png"))
    assert plots == [
        "calibration.png",
        "model_comparison.png",
        "permutation_importance.png",
        "precision_recall.png",
        "roc.png",
        "score_separation.png",
    ]


def test_evaluate_metrics_are_readable(evaluate_output):
    payload = json.loads((evaluate_output / "metrics.json").read_text())
    assert payload["pooled_metrics"]["stack"]["n"] > 0
    assert payload["n_outer_folds"] >= 3


@pytest.fixture(scope="module")
def trained_dir(train_dir, config_file, tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("cli-artifacts")
    result = run(
        "train",
        "--data-dir", str(train_dir),
        "--artifact-dir", str(base),
        "--config", str(config_file),
        "--skip-evaluation",
        "--quiet",
    )
    assert result.exit_code == 0, result.stdout
    runs = [p for p in base.iterdir() if p.is_dir()]
    assert len(runs) == 1, "one run directory per training run"
    return runs[0]


def test_train_creates_a_run_directory_with_a_model(trained_dir):
    assert (trained_dir / "model.joblib").is_file()
    summary = json.loads((trained_dir / "summary.json").read_text())
    assert summary["run_id"] == trained_dir.name
    assert summary["recall_floor_met"] is True


def test_predict_writes_a_consolidated_csv(trained_dir, predict_dir, tmp_path):
    output = tmp_path / "predictions.csv"
    result = run(
        "predict",
        "--model-dir", str(trained_dir),
        "--data-dir", str(predict_dir),
        "--output", str(output),
    )
    assert result.exit_code == 0, result.stdout
    assert "POTENTIAL" in result.stdout
    assert "not a confirmation" in result.stdout

    frame = pd.read_csv(output)
    assert len(frame) > 0
    assert set(frame["prediction"]) <= {"POTENTIAL", "UNLIKELY"}
    assert frame["model_run_id"].nunique() == 1


def test_predict_reports_a_clean_error_for_a_missing_model(predict_dir, tmp_path):
    result = run(
        "predict",
        "--model-dir", str(tmp_path / "nowhere"),
        "--data-dir", str(predict_dir),
        "--output", str(tmp_path / "out.csv"),
    )
    assert result.exit_code == 1
    assert "model.joblib" in result.stderr
    assert "Traceback" not in result.stderr


def test_repeated_training_never_overwrites_a_previous_run(
    train_dir, config_file, trained_dir
):
    """Runs are immutable: a second run lands in its own run-id directory."""
    base = trained_dir.parent
    result = run(
        "train",
        "--data-dir", str(train_dir),
        "--artifact-dir", str(base),
        "--config", str(config_file),
        "--skip-evaluation",
        "--quiet",
    )
    assert result.exit_code == 0, result.stdout

    runs = sorted(p for p in base.iterdir() if p.is_dir())
    assert len(runs) == 2
    assert trained_dir in runs
    assert (trained_dir / "model.joblib").is_file()
