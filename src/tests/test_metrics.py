"""Threshold selection semantics and the reported metric set."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import (
    calibration_curve_points,
    compute_metrics,
    per_star_summary,
    select_threshold,
    star_bootstrap_intervals,
)


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------
def test_picks_the_highest_precision_threshold():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.95, 0.90, 0.80, 0.60, 0.70, 0.40, 0.30, 0.10])

    choice = select_threshold(y, p)
    assert choice.threshold == pytest.approx(0.80)
    assert choice.recall == pytest.approx(0.75)
    assert choice.precision == pytest.approx(1.0)


def test_precision_can_take_priority_over_recall():
    y = np.array([1] * 10 + [0] * 10)
    p = np.concatenate([np.linspace(0.30, 0.99, 10), np.linspace(0.01, 0.35, 10)])

    choice = select_threshold(y, p)
    assert choice.precision == pytest.approx(1.0)
    assert choice.recall < 0.95


def test_ties_resolve_to_the_more_inclusive_threshold():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    # 0.8 and 0.3..0.8 all give perfect precision and recall; the lowest of the
    # tied candidates is kept because it generalises more safely.
    choice = select_threshold(y, p)
    assert choice.threshold == pytest.approx(0.8)
    assert choice.precision == pytest.approx(1.0)


def test_threshold_honours_sample_weights():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.9, 0.8, 0.7, 0.6])
    unweighted = select_threshold(y, p)
    heavy = select_threshold(y, p, sample_weight=np.array([1.0, 1.0, 100.0, 1.0]))
    assert unweighted.threshold == pytest.approx(0.6)
    assert heavy.threshold == pytest.approx(0.8)


def test_empty_or_all_negative_input_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        select_threshold(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="no positive"):
        select_threshold(np.array([0, 0]), np.array([0.2, 0.8]))


def test_flagging_uses_greater_or_equal():
    y = np.array([1, 0])
    p = np.array([0.5, 0.1])
    choice = select_threshold(y, p)
    assert (p >= choice.threshold).sum() == 1


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_metric_set_is_complete_and_consistent():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.7, 0.6, 0.2, 0.4, 0.1])
    metrics = compute_metrics(y, p, threshold=0.5)

    for key in (
        "precision", "recall", "f2", "average_precision", "roc_auc",
        "brier_score", "log_loss", "confusion_matrix", "confusion_matrix_counts",
    ):
        assert key in metrics

    counts = metrics["confusion_matrix_counts"]
    assert counts["tp"] == 2 and counts["fn"] == 1
    assert counts["fp"] == 1 and counts["tn"] == 2
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["n_flagged"] == 3


def test_f2_weights_recall_above_precision():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    high_recall = compute_metrics(y, p, threshold=0.5)
    # Perfect recall with 80% precision still scores above the F1 of the same point.
    assert high_recall["recall"] == pytest.approx(1.0)
    assert high_recall["f2"] > high_recall["precision"]


def test_metrics_accept_per_row_thresholds():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.6, 0.4, 0.55, 0.1])
    per_row = compute_metrics(y, p, threshold=np.array([0.5, 0.3, 0.6, 0.5]))
    assert per_row["confusion_matrix_counts"] == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}
    assert per_row["threshold"] is None
    assert per_row["threshold_min"] == pytest.approx(0.3)


def test_single_class_metrics_degrade_to_nan_not_crash():
    y = np.array([1, 1, 1])
    p = np.array([0.9, 0.8, 0.7])
    metrics = compute_metrics(y, p, threshold=0.5)
    assert np.isnan(metrics["roc_auc"])
    assert metrics["recall"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Calibration, per-star, bootstrap
# ---------------------------------------------------------------------------
def test_calibration_curve_bins_only_populated_ranges():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.15, 0.85, 0.95])
    points = calibration_curve_points(y, p, n_bins=10)
    assert len(points) == 4
    assert all(0.0 <= point["observed_rate"] <= 1.0 for point in points)


def test_per_star_summary_splits_by_star():
    stars = np.array(["a", "a", "b"], dtype=object)
    rows = per_star_summary(stars, np.array([1, 0, 1]), np.array([0.9, 0.2, 0.8]),
                            np.array([1, 0, 1]))
    assert [row["star_id"] for row in rows] == ["a", "b"]
    assert rows[0]["tp"] == 1 and rows[0]["tn"] == 1
    assert rows[1]["recall"] == pytest.approx(1.0)


def test_star_bootstrap_returns_ordered_intervals():
    rng = np.random.default_rng(0)
    stars = np.array([f"s{i // 4}" for i in range(40)], dtype=object)
    y = (rng.random(40) > 0.4).astype(int)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.15, 40), 0.01, 0.99)

    result = star_bootstrap_intervals(stars, y, p, threshold=0.5, n_resamples=100, seed=1)
    for name, interval in result.intervals.items():
        assert interval["lower"] <= interval["upper"], name
    assert result.n_resamples == 100


def test_star_bootstrap_is_reproducible():
    rng = np.random.default_rng(3)
    stars = np.array([f"s{i // 3}" for i in range(30)], dtype=object)
    y = (rng.random(30) > 0.5).astype(int)
    p = rng.random(30)
    kwargs = {"threshold": 0.5, "n_resamples": 50, "seed": 11}
    first = star_bootstrap_intervals(stars, y, p, **kwargs)
    second = star_bootstrap_intervals(stars, y, p, **kwargs)
    assert first.intervals == second.intervals


@pytest.mark.parametrize(
    "threshold", [0.5, np.float64(0.5), np.array(0.5), np.array([0.5, 0.5, 0.5, 0.5])]
)
def test_scalar_and_per_row_thresholds_agree(threshold):
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1])
    assert compute_metrics(y, p, threshold)["confusion_matrix_counts"] == {
        "tp": 1, "fp": 1, "fn": 1, "tn": 1,
    }


def test_wrong_number_of_thresholds_is_rejected():
    with pytest.raises(ValueError, match="Expected a scalar threshold or 4"):
        compute_metrics(np.array([1, 1, 0, 0]), np.zeros(4), np.array([0.5, 0.3]))
