"""Configuration loading, overrides, and validation."""

from __future__ import annotations

import pytest
import yaml

from src.config import BASE_LEARNERS, load_config
from src.errors import ConfigError


def _write(tmp_path, payload):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


@pytest.fixture
def payload():
    return load_config().to_dict()


def test_packaged_config_declares_every_base_learner():
    config = load_config()
    for name in BASE_LEARNERS:
        spec = config.model_spec(name)
        assert spec["grid"]
        assert spec["n_candidates"] >= 1


def test_packaged_config_matches_the_plan():
    config = load_config()
    assert config.seed == 42
    assert config.data["objective"]["metric"] == "precision"
    assert config.accepted_candidate_multiplier == pytest.approx(5.0)
    assert config.cv["outer_folds"] == 5
    assert config.cv["inner_folds"] == 3
    assert config.cv["min_outer_folds"] == 3
    assert config.cv["min_inner_folds"] == 2
    assert config.meta["C"] == pytest.approx(0.1)
    assert config.model_spec("logistic_regression")["n_candidates"] == 6
    for name in ("svm_rbf", "extra_trees", "lightgbm"):
        assert config.model_spec(name)["n_candidates"] == 24
    assert config.model_spec("extra_trees")["fixed"]["n_estimators"] == 500


def test_overrides_are_deep_merged():
    config = load_config().with_overrides({"cv": {"outer_folds": 4}})
    assert config.cv["outer_folds"] == 4
    assert config.cv["inner_folds"] == 3, "sibling keys must survive the merge"
    assert config.seed == 42


def test_to_dict_returns_a_copy():
    config = load_config()
    snapshot = config.to_dict()
    snapshot["seed"] = 999
    assert config.seed == 42


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


def test_non_mapping_file_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ConfigError, match="YAML mapping"):
        load_config(path)


@pytest.mark.parametrize("section", ["seed", "objective", "cv", "models", "meta"])
def test_missing_section_is_rejected(tmp_path, payload, section):
    del payload[section]
    with pytest.raises(ConfigError, match=f"missing required section '{section}'"):
        load_config(_write(tmp_path, payload))


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf")])
def test_accepted_candidate_multiplier_must_be_positive_and_finite(
    tmp_path, payload, value
):
    payload["weighting"]["accepted_candidate_multiplier"] = value
    with pytest.raises(ConfigError, match="accepted_candidate_multiplier"):
        load_config(_write(tmp_path, payload))


def test_objective_metric_must_be_precision(tmp_path, payload):
    payload["objective"]["metric"] = "average_precision"
    with pytest.raises(ConfigError, match=r"objective\.metric"):
        load_config(_write(tmp_path, payload))


def test_outer_folds_cannot_drop_below_three(tmp_path, payload):
    payload["cv"]["min_outer_folds"] = 2
    with pytest.raises(ConfigError, match="at least 3 for honest reporting"):
        load_config(_write(tmp_path, payload))


def test_outer_folds_cannot_be_below_their_own_minimum(tmp_path, payload):
    payload["cv"]["outer_folds"] = 3
    payload["cv"]["min_outer_folds"] = 4
    with pytest.raises(ConfigError, match="must not be below"):
        load_config(_write(tmp_path, payload))


def test_inner_folds_cannot_be_below_their_own_minimum(tmp_path, payload):
    payload["cv"]["inner_folds"] = 2
    payload["cv"]["min_inner_folds"] = 3
    with pytest.raises(ConfigError, match="must not be below"):
        load_config(_write(tmp_path, payload))


def test_fold_counts_must_be_at_least_two(tmp_path, payload):
    payload["cv"]["min_inner_folds"] = 1
    with pytest.raises(ConfigError, match="at least 2"):
        load_config(_write(tmp_path, payload))


def test_a_missing_base_learner_is_rejected(tmp_path, payload):
    del payload["models"]["svm_rbf"]
    with pytest.raises(ConfigError, match="missing base learners"):
        load_config(_write(tmp_path, payload))


def test_a_learner_without_a_grid_is_rejected(tmp_path, payload):
    payload["models"]["lightgbm"]["grid"] = "not-a-mapping"
    with pytest.raises(ConfigError, match="grid must be a mapping"):
        load_config(_write(tmp_path, payload))


def test_zero_candidates_is_rejected(tmp_path, payload):
    payload["models"]["extra_trees"]["n_candidates"] = 0
    with pytest.raises(ConfigError, match="n_candidates must be at least 1"):
        load_config(_write(tmp_path, payload))


def test_unknown_learner_lookup_is_reported():
    with pytest.raises(ConfigError, match="No configuration for base learner"):
        load_config().model_spec("random_forest")


def test_empty_grid_values_are_rejected():
    from src.models import candidate_params

    config = load_config().with_overrides({"models": {"lightgbm": {"grid": {"max_depth": []}}}})
    with pytest.raises(ConfigError, match="has no values"):
        candidate_params("lightgbm", config)
