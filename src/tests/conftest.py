"""Shared fixtures: a fast synthetic multi-star dataset and a reduced config.

Nothing in the test suite reads ``data/``; CI can run the whole flow against
these generated files.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Config, load_config
from src.schema import FeatureSchema, load_schema

# Features that carry real signal in the synthetic data; the rest are noise.
SIGNAL_FEATURES = ("snr_global", "MES", "max_mes", "depth_stability", "vshape_metric")
CONSTANT_FEATURE = "cadence_hours"
SPARSE_FEATURE = "ingress_egress_asymmetry"


def _candidate_plan(star_index: int, rng: np.random.Generator) -> list[tuple[str, str]]:
    """(label, status) pairs guaranteeing support in all four strata."""
    plan: list[tuple[str, str]] = [
        ("CONFIRMED", "accepted"),
        ("FALSE-POSITIVE", "accepted"),
    ]
    if star_index % 2 == 0:
        plan.append(("CONFIRMED", "rejected"))
    plan.extend([("FALSE-POSITIVE", "rejected")] * int(rng.integers(1, 4)))
    if star_index % 5 == 0:
        plan.append(("FALSE-POSITIVE", "provisional"))
    return plan


def build_synthetic_frame(
    star_index: int, schema: FeatureSchema, rng: np.random.Generator
) -> pd.DataFrame:
    """One star's candidate table, covering every column in the schema."""
    plan = _candidate_plan(star_index, rng)
    n = len(plan)
    labels = [label for label, _ in plan]
    statuses = [status for _, status in plan]
    positive = np.array([label == "CONFIRMED" for label in labels], dtype=float)

    data: dict[str, object] = {}
    for feature in schema.feature_columns:
        values = rng.normal(loc=0.0, scale=1.0, size=n)
        if feature in SIGNAL_FEATURES:
            values += 2.2 * positive + rng.normal(0.0, 0.45, size=n)
        data[feature] = values

    # A dataset-wide constant feature: allowed by the contract, dropped by the
    # per-learner preprocessing.
    data[CONSTANT_FEATURE] = np.full(n, 0.5)
    # A sparsely populated feature, to exercise imputation and its indicator.
    sparse = rng.normal(size=n)
    sparse[rng.random(n) < 0.6] = np.nan
    data[SPARSE_FEATURE] = sparse

    data["period_days"] = np.abs(rng.normal(12.0, 6.0, size=n)) + 0.5
    data["duration_hours"] = np.abs(rng.normal(4.0, 1.2, size=n)) + 0.3
    data["planet_radius_rearth"] = np.abs(rng.normal(3.0, 1.5, size=n)) + 0.2
    data["SES_mean"] = rng.normal(size=n)
    data["SES_std"] = np.abs(rng.normal(size=n))
    data["skewness_flux"] = rng.normal(size=n)
    data["kurtosis_flux"] = rng.normal(size=n)
    data["outlier_resistance"] = np.abs(rng.normal(size=n))

    frame = pd.DataFrame(data)

    # Withheld columns, filled the way the upstream pipeline fills them.
    frame["t0"] = rng.uniform(130.0, 1600.0, size=n)
    frame["duration_days"] = frame["duration_hours"] / 24.0
    frame["scale_skewness"] = frame["skewness_flux"]
    frame["scale_kurtosis"] = frame["kurtosis_flux"]
    frame["scale_outlier_resistance"] = frame["outlier_resistance"]
    frame["snr_per_transit_mean"] = frame["SES_mean"]
    frame["snr_per_transit_std"] = frame["SES_std"]
    frame["planet_radius_rjup"] = frame["planet_radius_rearth"] / 11.209
    frame["mes_threshold_used"] = 7.1
    frame["is_provisional_detection"] = [
        1.0 if status == "provisional" else 0.0 for status in statuses
    ]
    frame["detection_status"] = statuses
    frame["candidate_label"] = labels
    frame["matched_target"] = [
        f"Synth-{star_index:03d}{chr(98 + i)}" if label == "CONFIRMED" else None
        for i, label in enumerate(labels)
    ]
    frame["matched_period_ratio"] = [
        "direct" if label == "CONFIRMED" else None for label in labels
    ]
    return frame.loc[:, list(schema.known_columns)]


def write_synthetic_dataset(
    directory: Path,
    n_stars: int = 12,
    seed: int = 7,
    include_supervision: bool = True,
    schema: FeatureSchema | None = None,
) -> Path:
    """Write ``n_stars`` per-star CSVs into ``directory``."""
    schema = schema or load_schema()
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for index in range(n_stars):
        frame = build_synthetic_frame(index, schema, rng)
        if not include_supervision:
            frame = frame.drop(columns=list(schema.excluded_columns["supervision"]))
        day = 1 + (index % 28)
        frame.to_csv(directory / f"Synth-{index:03d}_202601{day:02d}.csv", index=False)
    return directory


@pytest.fixture(scope="session")
def schema() -> FeatureSchema:
    return load_schema()


@pytest.fixture(scope="session")
def fast_config() -> Config:
    """The packaged config with the search space cut down for CI."""
    base = load_config().to_dict()
    data = copy.deepcopy(base)
    for name in data["models"]:
        data["models"][name]["n_candidates"] = 2
    data["models"]["extra_trees"]["fixed"]["n_estimators"] = 25
    data["models"]["lightgbm"]["grid"]["n_estimators"] = [25]
    if "svm_rbf" in data["models"]:
        data["models"]["svm_rbf"]["calibration"]["cv"] = 3
    data["meta"]["n_candidates"] = 2
    data["meta"]["grid"]["C"] = [0.01, 1.0]
    data["cv"]["outer_folds"] = 3
    data["cv"]["inner_folds"] = 2
    data["cv"]["final_oof_folds"] = 2
    data["bootstrap"]["n_resamples"] = 25
    data["permutation_importance"]["n_repeats"] = 1
    return Config(data=data)


@pytest.fixture(scope="session")
def train_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("synthetic-train")
    return write_synthetic_dataset(directory, n_stars=12, seed=7)


@pytest.fixture(scope="session")
def predict_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("synthetic-predict")
    return write_synthetic_dataset(
        directory, n_stars=3, seed=99, include_supervision=False
    )
