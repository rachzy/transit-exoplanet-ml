"""The strict allowlist: what a model may see, and what it must never see."""

from __future__ import annotations

import pytest
import yaml

from src.data import load_dataset
from src.errors import ConfigError
from src.schema import load_schema

WITHHELD = {
    "supervision": ["candidate_label", "detection_status", "matched_target",
                    "matched_period_ratio"],
    "pipeline_control": ["mes_threshold_used", "is_provisional_detection"],
    "non_generalizable": ["t0"],
    "alias": ["duration_days", "scale_skewness", "scale_kurtosis",
              "scale_outlier_resistance", "snr_per_transit_mean",
              "snr_per_transit_std", "planet_radius_rjup"],
}


def test_schema_is_versioned(schema):
    assert isinstance(schema.schema_version, int)
    assert schema.schema_version >= 1


@pytest.mark.parametrize(
    ("group", "columns"), [(group, columns) for group, columns in WITHHELD.items()]
)
def test_withheld_groups_are_declared_and_excluded(schema, group, columns):
    assert list(schema.excluded_columns[group]) == columns
    for column in columns:
        assert column not in schema.feature_columns


def test_no_withheld_column_reaches_the_feature_matrix(train_dir, schema):
    """The one check that matters: the model's input columns."""
    dataset = load_dataset(train_dir, mode="train", schema=schema)
    for column in schema.withheld_columns:
        assert column not in dataset.features.columns
    assert tuple(dataset.features.columns) == tuple(schema.feature_columns)


def test_feature_matrix_excludes_provenance_columns(train_dir, schema):
    dataset = load_dataset(train_dir, mode="train", schema=schema)
    for column in ("source_file", "star_id", "row_index"):
        assert column not in dataset.features.columns


def test_target_and_status_are_mapped_not_modelled(schema):
    assert schema.target_column == "candidate_label"
    assert schema.status_column == "detection_status"
    assert schema.label_mapping == {"CONFIRMED": 1, "FALSE-POSITIVE": 0}
    assert schema.accepted_status == "accepted"
    assert schema.accepted_status in schema.valid_statuses


def test_known_columns_cover_features_and_exclusions(schema):
    known = set(schema.known_columns)
    assert known == set(schema.feature_columns) | set(schema.withheld_columns)
    assert len(known) == len(schema.feature_columns) + len(schema.withheld_columns)


def _write(tmp_path, payload):
    path = tmp_path / "schema.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_overlapping_feature_and_exclusion_is_rejected(tmp_path, schema):
    payload = schema.to_dict()
    payload["excluded_columns"]["alias"].append(payload["feature_columns"][0])
    with pytest.raises(ConfigError, match="both feature_columns and excluded_columns"):
        load_schema(_write(tmp_path, payload))


def test_duplicate_feature_is_rejected(tmp_path, schema):
    payload = schema.to_dict()
    payload["feature_columns"].append(payload["feature_columns"][0])
    with pytest.raises(ConfigError, match="Duplicate entries"):
        load_schema(_write(tmp_path, payload))


def test_accepted_status_must_be_valid(tmp_path, schema):
    payload = schema.to_dict()
    payload["accepted_status"] = "not-a-status"
    with pytest.raises(ConfigError, match="not in valid_statuses"):
        load_schema(_write(tmp_path, payload))


def test_label_mapping_must_be_binary(tmp_path, schema):
    payload = schema.to_dict()
    payload["label_mapping"] = {"CONFIRMED": 1, "FALSE-POSITIVE": 2}
    with pytest.raises(ConfigError, match=r"exactly the values"):
        load_schema(_write(tmp_path, payload))


def test_missing_schema_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_schema(tmp_path / "nope.yaml")
