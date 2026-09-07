"""End-to-end flow on synthetic data: evaluate, train, serialize, predict."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data import load_dataset
from src.errors import SchemaVersionError
from src.evaluate import evaluate_dataset
from src.predict import ADDED_COLUMNS, predict_dataset
from src.schema import load_schema
from src.training import (
    POTENTIAL,
    UNLIKELY,
    ModelBundle,
    load_model,
    train_model,
    write_training_artifacts,
)


@pytest.fixture(scope="module")
def evaluation(train_dir, schema, fast_config):
    dataset = load_dataset(train_dir, mode="train", schema=schema)
    return evaluate_dataset(dataset=dataset, config=fast_config)


def _model_names(evaluation):
    config = evaluation.config if hasattr(evaluation, "config") else evaluation.bundle.config
    return ("stack", *config["models"])


@pytest.fixture(scope="module")
def trained(train_dir, schema, fast_config, tmp_path_factory):
    dataset = load_dataset(train_dir, mode="train", schema=schema)
    run = train_model(dataset=dataset, config=fast_config, run_evaluation=False)
    directory = tmp_path_factory.mktemp("artifacts")
    return write_training_artifacts(run, directory)


# ---------------------------------------------------------------------------
# Nested evaluation
# ---------------------------------------------------------------------------
def test_evaluation_reports_every_model(evaluation):
    assert set(evaluation.pooled_metrics) == set(_model_names(evaluation))
    assert evaluation.n_outer_folds >= 3
    assert len(evaluation.inner_folds_per_outer) == evaluation.n_outer_folds
    assert all(n >= 2 for n in evaluation.inner_folds_per_outer)


def test_evaluation_scores_only_accepted_candidates(evaluation, train_dir, schema):
    dataset = load_dataset(train_dir, mode="train", schema=schema)
    assert len(evaluation.oof_predictions) == int(dataset.accepted.sum())


def test_every_accepted_row_is_predicted_exactly_once(evaluation, train_dir, schema):
    dataset = load_dataset(train_dir, mode="train", schema=schema)
    keys = set(
        zip(evaluation.oof_predictions["source_file"], evaluation.oof_predictions["row_index"])
    )
    expected = set(zip(dataset.source_file[dataset.accepted], dataset.row_index[dataset.accepted]))
    assert keys == expected


def test_evaluation_includes_bootstrap_calibration_and_importance(evaluation):
    for name in _model_names(evaluation):
        intervals = evaluation.bootstrap[name].intervals
        assert intervals["recall"]["lower"] <= intervals["recall"]["upper"]
        assert evaluation.calibration[name]

    importance = evaluation.importance_summary()
    assert not importance.empty
    assert set(importance["feature"]) <= set(load_schema().feature_columns)


def test_permutation_importance_comes_from_held_out_folds_only(evaluation):
    frame = evaluation.permutation_importance
    assert set(frame["outer_fold"]) == set(range(evaluation.n_outer_folds))
    assert set(frame["feature"]) == set(load_schema().feature_columns)


def test_comparison_table_marks_the_model_that_would_ship(evaluation):
    table = evaluation.comparison_table()
    shipped = table.loc[table["is_production"], "model"].tolist()
    assert shipped == [evaluation.would_ship()]
    assert len(table) == len(_model_names(evaluation))


def test_would_ship_is_the_highest_average_precision(evaluation):
    scores = evaluation.selection_scores("average_precision")
    recalls = evaluation.selection_recalls()
    feasible = [
        score for name, score in scores.items()
        if recalls[name] >= evaluation.config["objective"]["min_recall"] - 1e-12
    ]
    best = max(feasible)
    assert scores[evaluation.would_ship()] == pytest.approx(best)


def test_evaluation_result_is_json_serialisable(evaluation, tmp_path):
    from src.artifacts import write_json

    path = write_json(tmp_path / "metrics.json", evaluation.to_dict())
    payload = json.loads(path.read_text())
    assert payload["n_outer_folds"] == evaluation.n_outer_folds


# ---------------------------------------------------------------------------
# Training artifacts
# ---------------------------------------------------------------------------
REQUIRED_ARTIFACTS = (
    "model.joblib",
    "config.resolved.yaml",
    "feature_schema.yaml",
    "threshold.json",
    "training_report.json",
    "tuning.json",
    "training_stars.json",
    "provenance.json",
    "training_oof_predictions.csv",
    "per_star.csv",
    "summary.json",
    "MANIFEST.json",
)


@pytest.mark.parametrize("filename", REQUIRED_ARTIFACTS)
def test_artifact_directory_contains(trained, filename):
    assert (trained.artifact_dir / filename).is_file()


def test_artifact_records_full_provenance(trained):
    provenance = json.loads((trained.artifact_dir / "provenance.json").read_text())
    assert provenance["run_id"] == trained.bundle.run_id
    assert provenance["seed"] == trained.bundle.config["seed"]
    assert "git" in provenance
    assert provenance["dependencies"]["scikit-learn"]
    assert provenance["dependencies"]["lightgbm"]

    stars = json.loads((trained.artifact_dir / "training_stars.json").read_text())
    assert stars["n_stars"] == len(trained.bundle.train_stars)
    assert set(stars["input_file_checksums"]) == {
        f"{star}_202601{1 + i % 28:02d}.csv"
        for i, star in enumerate(trained.bundle.train_stars)
    }


def test_artifact_schema_and_threshold_are_persisted(trained):
    schema = yaml.safe_load((trained.artifact_dir / "feature_schema.yaml").read_text())
    assert schema["schema_version"] == trained.bundle.schema.schema_version
    assert schema["feature_columns"] == list(trained.bundle.feature_names)

    threshold = json.loads((trained.artifact_dir / "threshold.json").read_text())
    assert threshold["threshold"] == pytest.approx(trained.bundle.threshold)
    assert set(threshold["comparator_thresholds"]) == set(_model_names(trained))


def test_manifest_covers_every_written_file(trained):
    manifest = json.loads((trained.artifact_dir / "MANIFEST.json").read_text())
    for name in REQUIRED_ARTIFACTS:
        if name == "MANIFEST.json":
            continue
        assert name in manifest["files"]
        assert len(manifest["files"][name]["sha256"]) == 64


def test_rerunning_into_a_populated_directory_is_refused(trained, fast_config):
    from src.errors import TransitExoplanetMLError

    with pytest.raises(TransitExoplanetMLError, match="immutable"):
        write_training_artifacts(trained, trained.artifact_dir.parent)


def test_training_uses_the_precision_objective(trained):
    report = trained.report
    selected = trained.bundle.selected_model
    choice = report["models"][selected]["threshold_choice"]
    assert choice["precision"] == pytest.approx(1.0)


def test_training_oof_frame_labels_accepted_rows_only(trained):
    frame = pd.read_csv(trained.artifact_dir / "training_oof_predictions.csv")
    assert set(frame.loc[frame["accepted"], "prediction"]) <= {POTENTIAL, UNLIKELY}
    assert frame.loc[~frame["accepted"], "prediction"].isna().all()


# ---------------------------------------------------------------------------
# Serialization round trip
# ---------------------------------------------------------------------------
def test_reloaded_model_reproduces_probabilities_and_decisions(trained, train_dir, schema):
    dataset = load_dataset(train_dir, mode="train", schema=schema)
    reloaded = load_model(trained.artifact_dir)

    before = trained.bundle.predict_proba(dataset.X)
    after = reloaded.predict_proba(dataset.X)
    assert np.array_equal(before, after), "probabilities must be bit-identical after reload"

    assert np.array_equal(
        trained.bundle.predict_labels(dataset.X), reloaded.predict_labels(dataset.X)
    )
    assert reloaded.threshold == trained.bundle.threshold
    assert reloaded.run_id == trained.bundle.run_id
    assert reloaded.feature_names == trained.bundle.feature_names

    base_before = trained.bundle.base_probabilities(dataset.X)
    base_after = reloaded.base_probabilities(dataset.X)
    for name in base_before:
        assert np.array_equal(base_before[name], base_after[name]), name


def test_loading_rejects_an_incompatible_schema_version(trained, tmp_path, schema):
    payload = schema.to_dict()
    payload["schema_version"] = schema.schema_version + 1
    path = tmp_path / "future_schema.yaml"
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises(SchemaVersionError, match="schema version"):
        load_model(trained.artifact_dir, schema=load_schema(path))


def test_loading_rejects_a_changed_feature_set_at_the_same_version(
    trained, tmp_path, schema
):
    payload = schema.to_dict()
    payload["feature_columns"] = payload["feature_columns"][:-1]
    path = tmp_path / "shrunk_schema.yaml"
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises(SchemaVersionError, match="Bump schema_version"):
        load_model(trained.artifact_dir, schema=load_schema(path))


def test_loading_a_missing_bundle_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"model\.joblib"):
        ModelBundle.load(tmp_path)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def test_prediction_emits_one_consolidated_csv(trained, predict_dir, tmp_path):
    output = tmp_path / "predictions.csv"
    frame = predict_dataset(trained.artifact_dir, data_dir=predict_dir, output=output)

    assert output.is_file()
    written = pd.read_csv(output)
    assert len(written) == len(frame)
    assert written["star_id"].nunique() == 3


def test_prediction_carries_every_required_column(trained, predict_dir, schema):
    frame = predict_dataset(trained.artifact_dir, data_dir=predict_dir)
    for column in ADDED_COLUMNS:
        assert column in frame.columns, column
    # Original candidate columns are preserved.
    for column in schema.feature_columns:
        assert column in frame.columns
    for column in schema.excluded_columns["pipeline_control"]:
        assert column in frame.columns
    assert frame["model_run_id"].unique().tolist() == [trained.bundle.run_id]


def test_prediction_label_semantics(trained, predict_dir):
    frame = predict_dataset(trained.artifact_dir, data_dir=predict_dir)
    threshold = trained.bundle.threshold
    flagged = frame["potential_probability"] >= threshold
    assert (frame.loc[flagged, "prediction"] == POTENTIAL).all()
    assert (frame.loc[~flagged, "prediction"] == UNLIKELY).all()
    assert set(frame["prediction"]) <= {POTENTIAL, UNLIKELY}
    assert (frame["decision_threshold"] == threshold).all()


def test_prediction_probabilities_are_in_range(trained, predict_dir):
    frame = predict_dataset(trained.artifact_dir, data_dir=predict_dir)
    for column in ("potential_probability", *[c for c in frame.columns if c.startswith("prob_")]):
        values = frame[column].to_numpy(dtype=float)
        assert ((values >= 0.0) & (values <= 1.0)).all(), column


def test_prediction_ordering_is_deterministic(trained, predict_dir):
    first = predict_dataset(trained.artifact_dir, data_dir=predict_dir)
    second = predict_dataset(trained.artifact_dir, data_dir=predict_dir)
    pd.testing.assert_frame_equal(first, second)

    ordering = list(zip(first["source_file"], first["row_index"]))
    assert ordering == sorted(ordering)


def test_prediction_works_without_labels(trained, predict_dir, schema):
    dataset = load_dataset(predict_dir, mode="predict", schema=schema)
    assert dataset.y is None
    frame = predict_dataset(trained.artifact_dir, dataset=dataset)
    assert len(frame) == len(dataset)


def test_prediction_rejects_a_feature_set_mismatch(trained, predict_dir, tmp_path, schema):
    payload = schema.to_dict()
    payload["feature_columns"] = payload["feature_columns"][:-1]
    payload["excluded_columns"]["alias"].append(schema.feature_columns[-1])
    path = tmp_path / "narrow.yaml"
    path.write_text(yaml.safe_dump(payload))
    narrowed = load_schema(path)

    dataset = load_dataset(predict_dir, mode="predict", schema=narrowed)
    with pytest.raises(SchemaVersionError, match="do not match"):
        predict_dataset(trained.bundle, dataset=dataset)


def test_per_star_summary_covers_every_reported_model(evaluation, train_dir, schema):
    dataset = load_dataset(train_dir, mode="train", schema=schema)
    frame = pd.DataFrame(evaluation.per_star)
    assert set(frame["model"]) == set(_model_names(evaluation))

    stars_with_accepted = {
        star
        for star in dataset.stars
        if dataset.accepted[dataset.star_id == star].any()
    }
    for name in _model_names(evaluation):
        rows = frame[frame["model"] == name]
        assert set(rows["star_id"]) == stars_with_accepted
        assert (rows["n_flagged"] <= rows["n_accepted"]).all()
