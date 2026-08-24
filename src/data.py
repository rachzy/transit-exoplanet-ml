"""Dataset discovery, validation, and the star-equal weighting scheme."""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .errors import DataValidationError, EmptyDatasetError
from .schema import FeatureSchema, load_schema

FILENAME_PATTERN = re.compile(r"^(?P<star_id>.+)_(?P<date>\d{8})$")

MODES = ("train", "predict")

PROVENANCE_COLUMNS = ("source_file", "star_id", "row_index")


# ---------------------------------------------------------------------------
# Filenames
# ---------------------------------------------------------------------------
def parse_source_filename(filename: str) -> tuple[str, dt.date]:
    """Split ``<star-name>_<YYYYMMDD>.csv`` into its star id and date.

    Everything before the final underscore is the star id, so star names may
    themselves contain underscores.
    """
    name = Path(filename).name
    if not name.lower().endswith(".csv"):
        raise DataValidationError(
            f"{name!r}: expected a .csv file. Rename it to '<star-name>_<YYYYMMDD>.csv'."
        )
    stem = name[: -len(".csv")]
    match = FILENAME_PATTERN.match(stem)
    if match is None:
        raise DataValidationError(
            f"{name!r}: malformed filename. Expected '<star-name>_<YYYYMMDD>.csv', "
            "where the date is exactly 8 digits after the final underscore."
        )
    star_id = match.group("star_id")
    raw_date = match.group("date")
    if not star_id.strip():
        raise DataValidationError(f"{name!r}: star id is empty before the final underscore.")
    try:
        date = dt.datetime.strptime(raw_date, "%Y%m%d").date()
    except ValueError as exc:
        raise DataValidationError(
            f"{name!r}: {raw_date!r} is not a valid YYYYMMDD calendar date."
        ) from exc
    return star_id, date


def discover_files(data_dir: str | Path) -> list[Path]:
    """Return the candidate CSVs in ``data_dir``, sorted for deterministic order."""
    directory = Path(data_dir)
    if not directory.exists():
        raise EmptyDatasetError(
            f"Data directory not found: {directory}. Create it and add "
            "'<star-name>_<YYYYMMDD>.csv' files before running this command."
        )
    if not directory.is_dir():
        raise EmptyDatasetError(f"Not a directory: {directory}.")
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    if not files:
        raise EmptyDatasetError(
            f"No candidate CSV files found in {directory}. Add at least one "
            "'<star-name>_<YYYYMMDD>.csv' file, then re-run this command."
        )
    return files


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
def composite_strata(y: np.ndarray, accepted: np.ndarray) -> np.ndarray:
    """Label x acceptance strata, so grouped splits balance both at once."""
    return (np.asarray(y, dtype=int) * 2 + np.asarray(accepted, dtype=int)).astype(int)


def star_balanced_weights(star_ids: np.ndarray | pd.Series) -> np.ndarray:
    """Weights giving every star the same total, with a mean weight of 1.

    Recomputed for each fitting subset so the property holds for that fit --
    including the accepted-only meta-model fit, where stars contribute
    different numbers of rows than they do to the base learners.
    """
    ids = np.asarray(star_ids)
    if ids.size == 0:
        return np.zeros(0, dtype=float)
    unique, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(float)
    return weights * (ids.size / float(len(unique)))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    """A validated collection of candidate rows drawn from per-star CSVs."""

    mode: str
    schema: FeatureSchema
    frame: pd.DataFrame
    features: pd.DataFrame
    star_id: np.ndarray
    source_file: np.ndarray
    row_index: np.ndarray
    weights: np.ndarray
    files: tuple[Path, ...]
    file_checksums: dict[str, str]
    y: np.ndarray | None = None
    status: np.ndarray | None = None
    accepted: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.features.columns)

    @property
    def stars(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.star_id.tolist())))

    @property
    def X(self) -> np.ndarray:
        return self.features.to_numpy(dtype=float, copy=True)

    def require_supervision(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(y, accepted)``, raising if the dataset is unlabelled."""
        if self.y is None or self.accepted is None:
            raise DataValidationError(
                "This dataset has no labels. Load it with mode='train' from a "
                "directory whose files carry candidate_label and detection_status."
            )
        return self.y, self.accepted

    def composite_strata(self) -> np.ndarray:
        """Label x acceptance strata used to stratify the grouped splits."""
        return composite_strata(*self.require_supervision())

    def summary(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "mode": self.mode,
            "schema_version": self.schema.schema_version,
            "n_rows": len(self.frame),
            "n_stars": len(self.stars),
            "n_files": len(self.files),
            "n_features": len(self.features.columns),
            "stars": list(self.stars),
        }
        if self.y is not None and self.accepted is not None:
            info["n_positive"] = int(self.y.sum())
            info["n_negative"] = int((self.y == 0).sum())
            info["n_accepted"] = int(self.accepted.sum())
            info["n_accepted_positive"] = int((self.accepted & (self.y == 1)).sum())
            info["n_accepted_negative"] = int((self.accepted & (self.y == 0)).sum())
            info["n_rejected"] = int((~self.accepted).sum())
            info["status_counts"] = {
                str(k): int(v)
                for k, v in pd.Series(self.status).value_counts().sort_index().items()
            }
        return info


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(errors: list[str]) -> None:
    if errors:
        bullet = "\n  - ".join(errors)
        raise DataValidationError(f"Dataset failed validation:\n  - {bullet}")


def load_dataset(
    data_dir: str | Path,
    mode: str = "train",
    schema: FeatureSchema | None = None,
) -> Dataset:
    """Read, validate, and assemble every candidate CSV under ``data_dir``."""
    if mode not in MODES:
        raise DataValidationError(f"Unknown mode {mode!r}; expected one of {list(MODES)}.")
    schema = schema or load_schema()

    files = discover_files(data_dir)
    errors: list[str] = []

    # -- filenames and one-file-per-star ------------------------------------
    stars_by_file: dict[Path, str] = {}
    seen: dict[str, Path] = {}
    for path in files:
        try:
            star_id, _ = parse_source_filename(path.name)
        except DataValidationError as exc:
            errors.append(str(exc))
            continue
        if star_id in seen:
            errors.append(
                f"Star {star_id!r} appears in multiple files ({seen[star_id].name} and "
                f"{path.name}). Keep exactly one file per star in a dataset."
            )
            continue
        seen[star_id] = path
        stars_by_file[path] = star_id
    _fail(errors)

    # -- per-file reads and column checks -----------------------------------
    known = set(schema.known_columns)
    required = set(schema.required_for(mode))
    frames: list[pd.DataFrame] = []

    for path in files:
        star_id = stars_by_file[path]
        try:
            raw = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - pandas surfaces many types
            errors.append(f"{path.name}: could not be parsed as CSV ({exc}).")
            continue

        if raw.empty:
            errors.append(f"{path.name}: contains a header but no candidate rows.")
            continue

        unknown = [c for c in raw.columns if c not in known]
        if unknown:
            errors.append(
                f"{path.name}: unknown columns {sorted(unknown)}. Update the feature schema "
                f"(currently version {schema.schema_version}) before ingesting them."
            )

        missing_features = [c for c in schema.feature_columns if c not in raw.columns]
        if missing_features:
            errors.append(f"{path.name}: missing required feature columns {missing_features}.")

        missing_required = [c for c in sorted(required) if c not in raw.columns]
        if missing_required:
            errors.append(
                f"{path.name}: missing columns required for mode {mode!r}: {missing_required}."
            )

        duplicated = raw.duplicated(keep=False)
        if bool(duplicated.any()):
            rows = (np.flatnonzero(duplicated.to_numpy()) + 2).tolist()
            errors.append(
                f"{path.name}: duplicate candidate rows at CSV lines {rows}. "
                "Remove repeated candidates before ingesting."
            )

        block = raw.copy()
        block.insert(0, "source_file", path.name)
        block.insert(1, "star_id", star_id)
        block.insert(2, "row_index", np.arange(len(raw), dtype=int))
        frames.append(block)

    _fail(errors)
    frame = pd.concat(frames, ignore_index=True)

    # -- feature dtypes, infinities, empty features -------------------------
    features = frame.loc[:, list(schema.feature_columns)].copy()
    for column in features.columns:
        coerced = pd.to_numeric(features[column], errors="coerce")
        bad = coerced.isna() & features[column].notna()
        if bool(bad.any()):
            offenders = sorted({str(v) for v in features.loc[bad, column].unique()})[:5]
            errors.append(
                f"Feature {column!r} contains non-numeric values (e.g. {offenders}). "
                "Feature columns must be numeric; NaN is permitted for missing values."
            )
        features[column] = coerced.astype(float)

    infinite = features.columns[np.isinf(features.to_numpy(dtype=float, na_value=np.nan)).any(0)]
    if len(infinite):
        errors.append(
            f"Infinite values in features {sorted(infinite)}. Replace them with NaN "
            "upstream; NaN is imputed inside the model folds, infinities are not accepted."
        )

    empty = [c for c in features.columns if bool(features[c].isna().all())]
    if empty:
        errors.append(
            f"Features {empty} are entirely empty across the dataset. Populate them or "
            "remove them from the schema before training."
        )

    # -- supervision --------------------------------------------------------
    y: np.ndarray | None = None
    status: np.ndarray | None = None
    accepted: np.ndarray | None = None

    has_label = schema.target_column in frame.columns
    has_status = schema.status_column in frame.columns
    if mode == "train" or (has_label and has_status):
        if has_label:
            labels = frame[schema.target_column]
            invalid = sorted(
                {str(v) for v in labels[~labels.isin(list(schema.label_mapping))].unique()}
            )
            if invalid:
                errors.append(
                    f"Invalid {schema.target_column} values {invalid}. Expected one of "
                    f"{sorted(schema.label_mapping)}."
                )
            else:
                y = labels.map(schema.label_mapping).to_numpy(dtype=int)
        if has_status:
            statuses = frame[schema.status_column]
            invalid = sorted(
                {str(v) for v in statuses[~statuses.isin(list(schema.valid_statuses))].unique()}
            )
            if invalid:
                errors.append(
                    f"Invalid {schema.status_column} values {invalid}. Expected one of "
                    f"{sorted(schema.valid_statuses)}."
                )
            else:
                status = statuses.to_numpy(dtype=object)
                accepted = (statuses == schema.accepted_status).to_numpy(dtype=bool)

    _fail(errors)

    if mode == "train":
        assert y is not None and accepted is not None  # guaranteed by the checks above
        if len(np.unique(y)) < 2:
            errors.append(
                f"Training data contains only one class of {schema.target_column}. "
                "Add stars covering both CONFIRMED and FALSE-POSITIVE candidates."
            )
        if not accepted.any():
            errors.append(
                f"No rows have {schema.status_column} == {schema.accepted_status!r}. "
                "The meta-model and threshold are fitted on accepted candidates only."
            )
        elif len(np.unique(y[accepted])) < 2:
            errors.append(
                "Accepted candidates cover only one class. The meta-model and the "
                "recall-constrained threshold both need accepted rows of both classes."
            )
        _fail(errors)

    star_id = frame["star_id"].to_numpy(dtype=object)
    return Dataset(
        mode=mode,
        schema=schema,
        frame=frame,
        features=features,
        star_id=star_id,
        source_file=frame["source_file"].to_numpy(dtype=object),
        row_index=frame["row_index"].to_numpy(dtype=int),
        weights=star_balanced_weights(star_id),
        files=tuple(files),
        file_checksums={p.name: _checksum(p) for p in files},
        y=y,
        status=status,
        accepted=accepted,
    )


def validate_dataset(
    data_dir: str | Path,
    mode: str = "train",
    schema: FeatureSchema | None = None,
) -> dict[str, Any]:
    """Validate ``data_dir`` and return a summary; raises on any violation."""
    return load_dataset(data_dir, mode=mode, schema=schema).summary()
