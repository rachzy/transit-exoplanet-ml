"""Batch prediction into one consolidated, provenance-carrying CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import BASE_LEARNERS
from .data import Dataset, load_dataset
from .errors import SchemaVersionError
from .schema import FeatureSchema
from .stacking import STACK
from .training import POTENTIAL, UNLIKELY, ModelBundle, load_model

PROBABILITY_COLUMNS = tuple(f"prob_{name}" for name in BASE_LEARNERS)

ADDED_COLUMNS = (
    "source_file",
    "star_id",
    "row_index",
    *PROBABILITY_COLUMNS,
    "prob_stack",
    "potential_probability",
    "decision_threshold",
    "prediction",
    "model_name",
    "model_run_id",
)


def predict_dataset(
    model: str | Path | ModelBundle,
    data_dir: str | Path | None = None,
    output: str | Path | None = None,
    schema: FeatureSchema | None = None,
    dataset: Dataset | None = None,
) -> pd.DataFrame:
    """Score every candidate under ``data_dir`` with a saved model bundle.

    The returned frame keeps every original candidate column and appends the
    provenance, probability, and decision columns. Rows are ordered by source
    file then by their position within that file, so the output is stable
    across runs.
    """
    bundle = model if isinstance(model, ModelBundle) else load_model(model, schema=schema)

    if dataset is None:
        if data_dir is None:
            raise ValueError("Provide either data_dir or dataset.")
        dataset = load_dataset(data_dir, mode="predict", schema=schema or bundle.schema)

    if dataset.schema.schema_version != bundle.schema.schema_version:
        raise SchemaVersionError(
            f"Data was validated against schema version {dataset.schema.schema_version} "
            f"but the model expects version {bundle.schema.schema_version}."
        )
    if tuple(dataset.feature_names) != tuple(bundle.feature_names):
        missing = [c for c in bundle.feature_names if c not in dataset.feature_names]
        extra = [c for c in dataset.feature_names if c not in bundle.feature_names]
        raise SchemaVersionError(
            "Feature columns do not match the trained model. "
            f"Missing: {missing or 'none'}; unexpected: {extra or 'none'}."
        )

    X = dataset.X
    # Every candidate is scored so the output stays auditable; only the
    # selected model's column feeds potential_probability and the decision.
    candidates = bundle.stack.all_probabilities(X)
    probabilities = candidates[bundle.selected_model]

    frame = dataset.frame.copy()
    for name in BASE_LEARNERS:
        frame[f"prob_{name}"] = candidates[name]
    frame["prob_stack"] = candidates[STACK]
    frame["potential_probability"] = probabilities
    frame["decision_threshold"] = bundle.threshold
    frame["prediction"] = np.where(probabilities >= bundle.threshold, POTENTIAL, UNLIKELY)
    frame["model_name"] = bundle.selected_model
    frame["model_run_id"] = bundle.run_id

    frame = frame.sort_values(["source_file", "row_index"], kind="stable").reset_index(drop=True)

    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)

    return frame


def prediction_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["prediction"].value_counts().to_dict()
    return {
        "n_rows": len(frame),
        "n_stars": int(frame["star_id"].nunique()),
        POTENTIAL: int(counts.get(POTENTIAL, 0)),
        UNLIKELY: int(counts.get(UNLIKELY, 0)),
    }
