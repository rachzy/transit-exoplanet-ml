"""Per-learner preprocessing: dropping degenerates, imputation, scaling."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing import DropDegenerateFeatures, build_preprocessor


def test_drops_constant_and_all_nan_columns():
    X = np.array(
        [
            [1.0, 5.0, np.nan, 2.0],
            [2.0, 5.0, np.nan, 3.0],
            [3.0, 5.0, np.nan, np.nan],
        ]
    )
    dropper = DropDegenerateFeatures().fit(X)
    assert list(dropper.support_) == [True, False, False, True]
    assert dropper.transform(X).shape == (3, 2)


def test_support_is_learned_on_fit_data_only():
    """A column constant in the fit fold stays dropped at transform time."""
    fit = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    later = np.array([[1.0, 99.0], [2.0, -3.0]])
    dropper = DropDegenerateFeatures().fit(fit)
    assert dropper.transform(later).shape == (2, 1)
    assert np.array_equal(dropper.transform(later), [[1.0], [2.0]])


def test_feature_names_out_follow_the_support():
    X = np.array([[1.0, 0.0], [2.0, 0.0]])
    dropper = DropDegenerateFeatures().fit(X)
    assert list(dropper.get_feature_names_out(["a", "b"])) == ["a"]


def test_degenerate_fold_keeps_one_column():
    X = np.array([[1.0, 2.0], [1.0, 2.0]])
    dropper = DropDegenerateFeatures().fit(X)
    assert dropper.support_.sum() == 1


@pytest.mark.parametrize("learner", ["lightgbm", "catboost"])
def test_native_missing_value_learners_pass_missing_values_through(learner):
    X = np.array([[1.0, np.nan], [2.0, 4.0]])
    out = build_preprocessor(learner).fit_transform(X)
    assert np.isnan(out).any(), f"{learner} handles NaN natively; it must not be imputed"
    assert out.shape == X.shape


def test_extra_trees_pipeline_imputes_and_flags_missing():
    X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    out = build_preprocessor("extra_trees").fit_transform(X)
    assert not np.isnan(out).any()
    # Two features plus one missing indicator for the column that had a gap.
    assert out.shape == (3, 3)
    assert set(np.unique(out[:, -1])) <= {0.0, 1.0}


@pytest.mark.parametrize("learner", ["svm_rbf", "logistic_regression"])
def test_scaled_pipelines_impute_then_robust_scale(learner):
    rng = np.random.default_rng(0)
    X = rng.normal(loc=50.0, scale=20.0, size=(40, 3))
    X[3, 1] = np.nan
    out = build_preprocessor(learner).fit_transform(X)
    assert not np.isnan(out).any()
    # Robust scaling centres on the median.
    assert np.allclose(np.median(out[:, :3], axis=0), 0.0, atol=1e-9)


@pytest.mark.parametrize("learner", ["svm_rbf", "logistic_regression", "extra_trees"])
def test_scaled_pipelines_survive_a_constant_column(learner):
    X = np.array([[1.0, 9.0], [2.0, 9.0], [3.0, 9.0], [4.0, 9.0]])
    out = build_preprocessor(learner).fit_transform(X)
    assert np.isfinite(out).all(), "a zero-IQR column must not produce inf/NaN"


def test_unknown_learner_is_rejected():
    with pytest.raises(KeyError, match="No preprocessing pipeline"):
        build_preprocessor("random_forest")
