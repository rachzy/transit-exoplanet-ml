"""Grouped-split isolation, tuning determinism, and meta-model discipline."""

from __future__ import annotations

import numpy as np
import pytest

from src import stacking
from src.config import BASE_LEARNERS
from src.data import load_dataset
from src.errors import DataDiversityError
from src.models import candidate_params
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
@pytest.mark.parametrize("learner", BASE_LEARNERS)
def test_candidate_grids_are_deterministic_and_sized(learner, fast_config):
    first = candidate_params(learner, fast_config)
    second = candidate_params(learner, fast_config)
    assert first == second
    assert len(first) == fast_config.model_spec(learner)["n_candidates"]


def test_large_grids_are_sampled_without_duplicates():
    from src.config import load_config

    config = load_config()
    combos = candidate_params("lightgbm", config)
    assert len(combos) == 24
    assert len({tuple(sorted(c.items())) for c in combos}) == 24


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

    def spy_fit_base(learner, params, X, y_, groups, config):
        base_fit_sizes.append(len(y_))
        return real_fit_base(learner, params, X, y_, groups, config)

    def spy_fit_meta(P, y_, groups, config):
        meta_fit_labels.append(np.asarray(y_).copy())
        return real_fit_meta(P, y_, groups, config)

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
    assert set(result.thresholds) == set(stacking.reported_models())
    for name, choice in result.thresholds.items():
        assert choice.recall >= fast_config.min_recall - 1e-12, name


def test_base_out_of_fold_matrix_is_complete_and_probabilistic(dataset, fast_config):
    y, accepted = dataset.require_supervision()
    result = fit_stack(
        dataset.X, y, accepted, dataset.star_id, fast_config, dataset.feature_names
    )
    assert result.base_oof.shape == (len(dataset), len(BASE_LEARNERS))
    assert not np.isnan(result.base_oof).any()
    assert ((result.base_oof >= 0) & (result.base_oof <= 1)).all()


def test_meta_model_sees_only_the_four_base_probabilities(dataset, fast_config):
    y, accepted = dataset.require_supervision()
    result = fit_stack(
        dataset.X, y, accepted, dataset.star_id, fast_config, dataset.feature_names
    )
    assert result.stack.meta.n_features_in_ == len(BASE_LEARNERS)
    assert pytest.approx(fast_config.meta["C"]) == result.stack.meta.C
