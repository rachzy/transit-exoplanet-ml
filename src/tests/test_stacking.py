"""Grouped-split isolation, tuning determinism, and meta-model discipline."""

from __future__ import annotations

import numpy as np
import pytest

from src import stacking
from src.data import load_dataset
from src.errors import DataDiversityError
from src.models import meta_candidate_params
from src.stacking import (
    STACK,
    composite_strata,
    fit_stack,
    grouped_splits,
    resolve_n_splits,
)


@pytest.fixture(scope="module")
def dataset(train_dir, schema):
    return load_dataset(train_dir, mode="train", schema=schema)


# ---------------------------------------------------------------------------
# Split isolation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_splits", [3, 4, 5])
def test_no_star_appears_on_both_sides_of_a_fold(dataset, n_splits):
    strata = composite_strata(*dataset.require_supervision())
    splits = grouped_splits(strata, dataset.star_id, n_splits, seed=42)

    assert len(splits) == n_splits
    for train_idx, val_idx in splits:
        train_stars = set(dataset.star_id[train_idx])
        val_stars = set(dataset.star_id[val_idx])
        assert train_stars.isdisjoint(val_stars)
        assert val_stars, "every fold must hold out at least one star"


def test_every_row_is_validated_exactly_once(dataset):
    strata = composite_strata(*dataset.require_supervision())
    splits = grouped_splits(strata, dataset.star_id, 3, seed=42)
    validated = np.concatenate([val for _, val in splits])
    assert sorted(validated) == list(range(len(dataset)))


def test_a_star_is_never_split_across_folds(dataset):
    strata = composite_strata(*dataset.require_supervision())
    splits = grouped_splits(strata, dataset.star_id, 3, seed=42)
    for star in dataset.stars:
        rows = set(np.flatnonzero(dataset.star_id == star))
        holding = [i for i, (_, val) in enumerate(splits) if rows & set(val)]
        assert len(holding) == 1, f"{star} was held out by folds {holding}"


def test_nested_inner_splits_stay_inside_the_outer_training_stars(dataset):
    y, accepted = dataset.require_supervision()
    strata = composite_strata(y, accepted)
    for outer_train, outer_val in grouped_splits(strata, dataset.star_id, 3, seed=42):
        held_out = set(dataset.star_id[outer_val])
        inner = grouped_splits(
            strata[outer_train], dataset.star_id[outer_train], 2, seed=42
        )
        for inner_train, inner_val in inner:
            rows_train = outer_train[inner_train]
            rows_val = outer_train[inner_val]
            assert set(dataset.star_id[rows_train]).isdisjoint(held_out)
            assert set(dataset.star_id[rows_val]).isdisjoint(held_out)
            assert set(dataset.star_id[rows_train]).isdisjoint(dataset.star_id[rows_val])


def test_splits_are_reproducible(dataset):
    strata = composite_strata(*dataset.require_supervision())
    first = grouped_splits(strata, dataset.star_id, 3, seed=42)
    second = grouped_splits(strata, dataset.star_id, 3, seed=42)
    for (a_tr, a_va), (b_tr, b_va) in zip(first, second):
        assert np.array_equal(a_tr, b_tr) and np.array_equal(a_va, b_va)


# ---------------------------------------------------------------------------
# Fold-count resolution
# ---------------------------------------------------------------------------
def test_folds_reduce_only_as_far_as_support_requires():
    # Four strata; the thinnest is backed by exactly four stars.
    strata = np.array([0] * 8 + [1] * 8 + [2] * 4 + [3] * 8)
    groups = np.array(
        [f"s{i}" for i in range(8)]
        + [f"s{i}" for i in range(8)]
        + [f"s{i}" for i in range(4)]
        + [f"s{i}" for i in range(8)],
        dtype=object,
    )
    assert resolve_n_splits(strata, groups, desired=5, minimum=3, level="outer") == 4


def test_full_fold_count_is_kept_when_support_allows(dataset):
    strata = composite_strata(*dataset.require_supervision())
    assert resolve_n_splits(strata, dataset.star_id, 5, 3, "outer") == 5


def test_insufficient_diversity_raises_a_diagnostic():
    strata = np.array([0, 0, 0, 0, 1, 1])
    groups = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
    with pytest.raises(DataDiversityError) as excinfo:
        resolve_n_splits(strata, groups, desired=5, minimum=3, level="outer")
    message = str(excinfo.value)
    assert "stars per (label, acceptance) stratum" in message
    assert "3-star minimum" in message


# ---------------------------------------------------------------------------
# Candidate grids
# ---------------------------------------------------------------------------
def test_meta_candidate_grid_is_deterministic_and_sized(fast_config):
    first = meta_candidate_params(fast_config)
    second = meta_candidate_params(fast_config)
    assert first == second
    assert len(first) == fast_config.meta["n_candidates"]


# ---------------------------------------------------------------------------
# Which rows train what
# ---------------------------------------------------------------------------
def test_rejected_rows_train_bases_but_never_the_meta_model(
    dataset, fast_config, monkeypatch
):
    """Base learners see every candidate; the meta-model sees accepted only."""
    y, accepted = dataset.require_supervision()
    base_fit_sizes: list[int] = []
    meta_fit_labels: list[np.ndarray] = []
    threshold_labels: list[np.ndarray] = []

    real_fit_base = stacking.fit_base
    real_fit_meta = stacking.fit_meta
    real_select = stacking.select_threshold

    def spy_fit_base(learner, params, X, y_, accepted_, groups, config):
        base_fit_sizes.append(len(y_))
        return real_fit_base(learner, params, X, y_, accepted_, groups, config)

    def spy_fit_meta(P, y_, groups, config, params=None):
        meta_fit_labels.append(np.asarray(y_).copy())
        return real_fit_meta(P, y_, groups, config, params)

    def spy_select(y_, prob, min_recall, sample_weight=None):
        threshold_labels.append(np.asarray(y_).copy())
        return real_select(y_, prob, min_recall, sample_weight=sample_weight)

    monkeypatch.setattr(stacking, "fit_base", spy_fit_base)
    monkeypatch.setattr(stacking, "fit_meta", spy_fit_meta)
    monkeypatch.setattr(stacking, "select_threshold", spy_select)

    fit_stack(
        dataset.X, y, accepted, dataset.star_id, fast_config, dataset.feature_names
    )

    n_accepted = int(accepted.sum())
    n_total = len(dataset)
    assert n_accepted < n_total, "the fixture must contain rejected rows"

    # Base learners are fitted on folds of the full set, which includes rejected rows.
    assert max(base_fit_sizes) > n_accepted
    # The final base refit sees every row.
    assert n_total in base_fit_sizes

    # Every meta fit is bounded by the accepted rows.
    assert meta_fit_labels, "the meta-model must be fitted"
    assert max(len(labels) for labels in meta_fit_labels) <= n_accepted
    # The final meta fit uses exactly the accepted rows.
    assert n_accepted in [len(labels) for labels in meta_fit_labels]

    # Thresholds are chosen over accepted rows only.
    assert threshold_labels
    for labels in threshold_labels:
        assert len(labels) <= n_accepted


def test_threshold_is_derived_from_cross_fitted_accepted_rows(dataset, fast_config):
    y, accepted = dataset.require_supervision()
    result = fit_stack(
        dataset.X, y, accepted, dataset.star_id, fast_config, dataset.feature_names
    )
    crossfit = result.meta_crossfit
    assert np.isnan(crossfit[~accepted]).all(), "rejected rows get no meta prediction"
    assert not np.isnan(crossfit[accepted]).all()

    choice = result.thresholds[STACK]
    assert choice.min_recall == pytest.approx(fast_config.min_recall)
    assert choice.recall >= fast_config.min_recall - 1e-12
    assert result.stack.threshold == pytest.approx(choice.threshold)


def test_every_reported_model_gets_its_own_threshold(dataset, fast_config):
    y, accepted = dataset.require_supervision()
    result = fit_stack(
        dataset.X, y, accepted, dataset.star_id, fast_config, dataset.feature_names
    )
    assert set(result.thresholds) == {"stack", *fast_config.base_learners}
    for name, choice in result.thresholds.items():
        assert choice.recall >= fast_config.min_recall - 1e-12, name


def test_base_out_of_fold_matrix_is_complete_and_probabilistic(dataset, fast_config):
    y, accepted = dataset.require_supervision()
    result = fit_stack(
        dataset.X, y, accepted, dataset.star_id, fast_config, dataset.feature_names
    )
    assert result.base_oof.shape == (len(dataset), len(fast_config.base_learners))
    assert not np.isnan(result.base_oof).any()
    assert ((result.base_oof >= 0) & (result.base_oof <= 1)).all()


def test_meta_model_sees_only_the_four_base_probabilities(dataset, fast_config):
    y, accepted = dataset.require_supervision()
    result = fit_stack(
        dataset.X, y, accepted, dataset.star_id, fast_config, dataset.feature_names
    )
    assert result.stack.meta.n_features_in_ == len(fast_config.base_learners)
    assert pytest.approx(result.best_params["meta"]["C"]) == result.stack.meta.C
    assert "meta" in result.tuning


def test_meta_inputs_exclude_validation_stars_from_all_upstream_fits(
    dataset, fast_config, monkeypatch
):
    """Trace tuning, preprocessing/base fitting, OOF prediction, and meta use."""
    y, accepted = dataset.require_supervision()
    groups = dataset.star_id
    tr, va = grouped_splits(composite_strata(y, accepted), groups, 2, seed=42)[0]
    held_out = set(groups[va])
    # Carry original row indices through subsets so prediction calls can be
    # checked against the stars used to fit their upstream pipeline.
    X = np.column_stack([np.arange(len(y)), dataset.X])
    fit_stars = []
    tuning_stars = []
    prediction_calls = []
    real_fit_base = stacking.fit_base
    real_tune_base = stacking.tune_base_learner

    def trace_fit(learner, params, X_, y_, accepted_, groups_, config):
        stars = set(groups_)
        assert stars.isdisjoint(held_out)
        fit_stars.append(stars)
        fitted = real_fit_base(learner, params, X_, y_, accepted_, groups_, config)
        real_predict = fitted.predict_proba

        def trace_predict(X_val):
            prediction_stars = set(groups[X_val[:, 0].astype(int)])
            assert stars.isdisjoint(prediction_stars)
            prediction_calls.append(prediction_stars)
            return real_predict(X_val)

        fitted.predict_proba = trace_predict
        return fitted

    def trace_tune(learner, X_, y_, accepted_, groups_, splits, config, progress=None):
        assert set(groups_).isdisjoint(held_out)
        tuning_stars.append(set(groups_))
        return real_tune_base(
            learner, X_, y_, accepted_, groups_, splits, config, progress
        )

    monkeypatch.setattr(stacking, "fit_base", trace_fit)
    monkeypatch.setattr(stacking, "tune_base_learner", trace_tune)
    folds = stacking.prepare_meta_folds(X, y, accepted, groups, [(tr, va)], fast_config)
    fold = folds[0]
    assert len(tuning_stars) == len(fast_config.base_learners)
    assert all(stars == set(groups[tr]) for stars in tuning_stars)
    assert fit_stars and prediction_calls
    assert set(groups[tr]) in fit_stars  # Refit before predicting meta-validation.
    assert held_out in prediction_calls
    assert fold.train_probabilities.shape == (len(tr), len(fast_config.base_learners))
    assert fold.val_probabilities.shape == (len(va), len(fast_config.base_learners))

    real_fit_meta = stacking.fit_meta
    meta_calls = []

    def trace_meta(P, y_, groups_, config, params=None):
        np.testing.assert_array_equal(P, fold.train_probabilities[accepted[tr]])
        np.testing.assert_array_equal(y_, y[tr][accepted[tr]])
        np.testing.assert_array_equal(groups_, groups[tr][accepted[tr]])
        meta_calls.append(params)
        fitted = real_fit_meta(P, y_, groups_, config, params)
        real_predict = fitted.predict_proba

        def trace_predict(P_val):
            np.testing.assert_array_equal(P_val, fold.val_probabilities[accepted[va]])
            return real_predict(P_val)

        fitted.predict_proba = trace_predict
        return fitted

    monkeypatch.setattr(stacking, "fit_meta", trace_meta)
    base_fit_count = len(fit_stars)
    tuning = stacking.tune_meta_learner(folds, y, accepted, groups, fast_config)
    predictions = stacking.crossfit_meta_predictions(
        folds, y, accepted, groups, fast_config, tuning.best_params
    )
    assert len(meta_calls) == len(meta_candidate_params(fast_config)) + 1
    assert len(fit_stars) == base_fit_count, "reuse base inputs across C candidates"
    expected_rows = np.zeros(len(y), dtype=bool)
    expected_rows[va[accepted[va]]] = True
    np.testing.assert_array_equal(np.isfinite(predictions), expected_rows)


@pytest.mark.parametrize("change", ["labels", "features"])
def test_meta_training_inputs_are_invariant_to_held_out_data(
    dataset, fast_config, change
):
    """Held-out labels/features cannot alter upstream tuning or training inputs."""
    y, accepted = dataset.require_supervision()
    groups = dataset.star_id
    split = grouped_splits(composite_strata(y, accepted), groups, 2, seed=42)[0]
    _, va = split
    original = stacking.prepare_meta_folds(
        dataset.X, y, accepted, groups, [split], fast_config
    )[0]
    changed_X, changed_y = dataset.X.copy(), y.copy()
    if change == "labels":
        changed_y[va] = 1 - changed_y[va]
    else:
        changed_X[va] = 10000.0
    # Freeze the split to test model-input isolation independently of the
    # label-aware stratification used to assign stars to folds.
    changed = stacking.prepare_meta_folds(
        changed_X, changed_y, accepted, groups, [split], fast_config
    )[0]
    np.testing.assert_array_equal(original.train_probabilities, changed.train_probabilities)
    if change == "labels":
        np.testing.assert_array_equal(original.val_probabilities, changed.val_probabilities)


def test_meta_input_isolation_reports_insufficient_nested_star_support(fast_config):
    X = np.arange(8, dtype=float).reshape(4, 2)
    y = np.array([0, 1, 0, 1])
    accepted = np.ones(4, dtype=bool)
    groups = np.array(["train", "train", "held-out", "held-out"])
    splits = [(np.array([0, 1]), np.array([2, 3]))]
    with pytest.raises(DataDiversityError, match=r"base-within-meta.*meta fold 1"):
        stacking.prepare_meta_folds(X, y, accepted, groups, splits, fast_config)
