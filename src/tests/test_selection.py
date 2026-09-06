"""Which model ships, and why."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import load_config
from src.data import load_dataset
from src.errors import ConfigError, DataDiversityError
from src.stacking import STACK, fit_stack, reported_models, select_best_model
from src.training import train_model


# ---------------------------------------------------------------------------
# select_best_model
# ---------------------------------------------------------------------------
def test_picks_the_highest_scorer():
    scores = {
        STACK: 0.80,
        "lightgbm": 0.91,
        "extra_trees": 0.85,
        "svm_rbf": 0.88,
        "logistic_regression": 0.70,
    }
    assert select_best_model(scores) == "lightgbm"


def test_ties_fall_to_the_earlier_candidate():
    """The stack leads reported_models(), so it keeps a tie."""
    scores = dict.fromkeys(reported_models(), 0.9)
    assert select_best_model(scores) == STACK

    without_stack = [m for m in reported_models() if m != STACK]
    assert select_best_model(scores, without_stack) == without_stack[0]


def test_selection_is_restricted_to_the_candidate_list():
    scores = {STACK: 0.5, "lightgbm": 0.99, "extra_trees": 0.7}
    assert select_best_model(scores, [STACK, "extra_trees"]) == "extra_trees"


def test_unscorable_candidates_are_skipped():
    scores = {STACK: float("nan"), "lightgbm": 0.6}
    assert select_best_model(scores) == "lightgbm"


def test_no_scorable_candidate_is_an_error():
    with pytest.raises(DataDiversityError, match="No candidate model could be scored"):
        select_best_model({STACK: float("nan")})


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_default_strategy_is_best():
    config = load_config()
    assert config.selection["strategy"] == "best"
    assert config.selection["metric"] == "precision"
    assert set(config.selection_candidates) == set(reported_models())


@pytest.mark.parametrize(
    ("patch", "match"),
    [
        ({"strategy": "random_forest"}, "must be 'best' or one of"),
        ({"score_source": "vibes"}, "score_source must be one of"),
        ({"candidates": []}, "must list at least one model"),
        ({"candidates": ["not_a_model"]}, "unknown models"),
    ],
)
def test_invalid_selection_config_is_rejected(patch, match):
    with pytest.raises(ConfigError, match=match):
        load_config().with_overrides({"selection": patch})


# ---------------------------------------------------------------------------
# End-to-end selection
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dataset(train_dir, schema):
    return load_dataset(train_dir, mode="train", schema=schema)


def test_training_ships_the_highest_scoring_model(dataset, fast_config):
    run = train_model(dataset=dataset, config=fast_config, run_evaluation=False)
    selection = run.bundle.selection

    assert selection.strategy == "best"
    assert selection.score_source == "crossfit"
    best = max(s for s in selection.scores.values() if not np.isnan(s))
    assert selection.scores[selection.selected_model] == pytest.approx(best)
    assert run.bundle.stack.selected_model == selection.selected_model


def test_pinning_a_model_overrides_the_ranking(dataset, fast_config):
    pinned = fast_config.with_overrides({"selection": {"strategy": "logistic_regression"}})
    run = train_model(dataset=dataset, config=pinned, run_evaluation=False)

    assert run.bundle.selected_model == "logistic_regression"
    assert run.bundle.selection.strategy == "logistic_regression"
    # The threshold travels with the pinned model, not with the stack.
    assert run.bundle.threshold == pytest.approx(
        run.fit.thresholds["logistic_regression"].threshold
    )


def test_selection_changes_what_predict_proba_serves(dataset, fast_config):
    """Each selection must serve its own model's probabilities."""
    run = train_model(dataset=dataset, config=fast_config, run_evaluation=False)
    X = dataset.X
    candidates = run.bundle.stack.all_probabilities(X)

    for name in reported_models():
        switched = run.bundle.stack.with_selection(name)
        assert np.array_equal(switched.predict_proba(X), candidates[name])
        assert switched.threshold == pytest.approx(run.fit.thresholds[name].threshold)


def test_every_candidate_keeps_a_threshold(dataset, fast_config):
    y, accepted = dataset.require_supervision()
    fit = fit_stack(
        dataset.X, y, accepted, dataset.star_id, fast_config, dataset.feature_names
    )
    assert set(fit.stack.thresholds) == set(reported_models())


def test_selecting_an_unfitted_model_is_rejected(dataset, fast_config):
    run = train_model(dataset=dataset, config=fast_config, run_evaluation=False)
    with pytest.raises(ValueError, match="No threshold for selected model"):
        run.bundle.stack.with_selection("random_forest")


def test_selection_record_reports_the_runner_up_and_margin(dataset, fast_config):
    run = train_model(dataset=dataset, config=fast_config, run_evaluation=False)
    selection = run.bundle.selection

    assert selection.runner_up != selection.selected_model
    assert selection.margin >= 0.0
    ranked = selection.ranked
    assert ranked[0][0] == selection.selected_model
    assert [name for name, _ in ranked][1] == selection.runner_up
