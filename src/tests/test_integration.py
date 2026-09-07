"""End-to-end flow on synthetic data: evaluate, train, serialize, predict."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data import load_dataset
from src.errors import SchemaVersionError, TransitExoplanetMLError
from src.evaluate import evaluate_dataset
from src.predict import ADDED_COLUMNS, predict_dataset
from src.schema import load_schema
from src.training import (
    POTENTIAL,
    UNLIKELY,
    ModelBundle,
    load_model,
    train_model,
    write_evaluation_artifacts,
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


@pytest.mark.parametrize(
    ("scores", "expected_mean", "expected_std", "valid_count"),
    [
        ([0.5, 0.9, float("nan")], 0.7, np.sqrt(0.08), 2),
        ([0.5, float("nan")], 0.5, None, 1),
        ([float("nan")], None, None, 0),
        ([], None, None, 0),
    ],
)
def test_fold_ap_reports_equal_fold_means_and_unscorable_folds(
    evaluation, tmp_path, scores, expected_mean, expected_std, valid_count
):
    from src.artifacts import write_json

    # Unequal sample counts must not turn the fold mean into a row-weighted
    # mean. Missing and single-class folds remain visible in the report.
    folds = [
        {"outer_fold": index, "average_precision": score, "n": 10 ** (index + 1)}
        for index, score in enumerate(scores)
    ]
    changed = replace(evaluation, fold_metrics={name: folds for name in _model_names(evaluation)})
    path = write_json(tmp_path / "metrics.json", changed.to_dict())
    summary = json.loads(path.read_text())["fold_ap_summary"]["stack"]
    if expected_mean is None:
        assert summary["mean"] is None
    else:
        assert summary["mean"] == pytest.approx(expected_mean)
    if expected_std is None:
        assert summary["std"] is None
    else:
        assert summary["std"] == pytest.approx(expected_std)
    assert summary["n_valid_folds"] == valid_count
    assert summary["n_outer_folds"] == changed.n_outer_folds
    table = changed.fold_ap_table()
    assert len(table) == len(_model_names(changed)) * changed.n_outer_folds
    for _, rows in table.groupby("model"):
        assert rows.outer_fold.tolist() == list(range(changed.n_outer_folds))
        assert rows.average_precision.notna().sum() == valid_count
    assert changed.selection_scores() == evaluation.selection_scores()


def test_evaluation_csvs_agree_with_json_and_explicit_run_id_cannot_overwrite(
    evaluation, tmp_path
):
    written = write_evaluation_artifacts(
        evaluation, tmp_path, seed=42, run_id="20260907T120000Z-12345678"
    )
    directory = written[0].parent
    before = (directory / "metrics.json").read_bytes()
    payload = json.loads(before)
    comparison = pd.read_csv(directory / "model_comparison.csv").set_index("model")
    folds = pd.read_csv(directory / "fold_average_precision.csv")
    for name in _model_names(evaluation):
        np.testing.assert_allclose(
            comparison.loc[name, "mean_fold_average_precision"],
            payload["fold_ap_summary"][name]["mean"],
        )
        for row in folds.loc[folds.model == name].itertuples():
            original = next(f for f in evaluation.fold_metrics[name]
                            if f["outer_fold"] == row.outer_fold)
            assert row.average_precision == pytest.approx(original["average_precision"])
    with pytest.raises(TransitExoplanetMLError, match="Reports are immutable"):
        write_evaluation_artifacts(evaluation, tmp_path, seed=42, run_id=directory.name)
    assert (directory / "metrics.json").read_bytes() == before


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
    assert provenance["dependencies"]["catboost"]

    stars = json.loads((trained.artifact_dir / "training_stars.json").read_text())
    assert stars["n_stars"] == len(trained.bundle.train_stars)
    assert set(stars["input_file_checksums"]) == {
        f"{star}_202601{1 + i % 28:02d}.csv"
        for i, star in enumerate(trained.bundle.train_stars)
    }


def test_summary_includes_nested_evaluation_models(trained, evaluation, tmp_path):
    # ``train_model(run_evaluation=False)`` is used by the shared fixture for
    # speed. Attach the already-computed evaluation to exercise the artifact
    # path used by the CLI, where nested evaluation is present.
    run = copy.copy(trained)
    run.evaluation = evaluation
    written = write_training_artifacts(run, tmp_path)
    summary = json.loads((written.artifact_dir / "summary.json").read_text())
    assert set(summary["nested_evaluation"]["models"]) == set(_model_names(evaluation))
    assert "fold_ap_summary" in summary["nested_evaluation"]
    report_dir = written.artifact_dir / "evaluation"
    assert (report_dir / "fold_average_precision.csv").is_file()
    provenance = json.loads((report_dir / "provenance.json").read_text())
    assert provenance["run_id"] == written.bundle.run_id


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


def test_training_meets_the_configured_recall_floor(trained):
    report = trained.report
    selected = trained.bundle.selected_model
    choice = report["models"][selected]["threshold_choice"]
    assert choice["recall"] >= trained.bundle.config["objective"]["min_recall"]


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
