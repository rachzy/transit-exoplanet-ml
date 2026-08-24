"""Versioned strict feature schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

DEFAULT_SCHEMA_RESOURCE = "schema.yaml"


@dataclass(frozen=True)
class FeatureSchema:
    """The set of columns a model may see, and the rules governing the rest."""

    schema_version: int
    target_column: str
    status_column: str
    label_mapping: dict[str, int]
    valid_statuses: tuple[str, ...]
    accepted_status: str
    feature_columns: tuple[str, ...]
    excluded_columns: dict[str, tuple[str, ...]]
    required_columns: dict[str, tuple[str, ...]]
    source: Path | None = field(default=None, compare=False)

    @property
    def known_columns(self) -> tuple[str, ...]:
        """Every column the contract recognises, in schema order."""
        seen: list[str] = list(self.feature_columns)
        for group in self.excluded_columns.values():
            seen.extend(c for c in group if c not in seen)
        return tuple(seen)

    @property
    def withheld_columns(self) -> tuple[str, ...]:
        """Columns that exist in the CSVs but must never reach a model."""
        return tuple(c for group in self.excluded_columns.values() for c in group)

    def required_for(self, mode: str) -> tuple[str, ...]:
        if mode not in self.required_columns:
            raise ConfigError(
                f"Unknown mode {mode!r}; expected one of {sorted(self.required_columns)}."
            )
        return self.required_columns[mode]

    def positive_labels(self) -> tuple[str, ...]:
        return tuple(k for k, v in self.label_mapping.items() if v == 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_column": self.target_column,
            "status_column": self.status_column,
            "label_mapping": dict(self.label_mapping),
            "valid_statuses": list(self.valid_statuses),
            "accepted_status": self.accepted_status,
            "feature_columns": list(self.feature_columns),
            "excluded_columns": {k: list(v) for k, v in self.excluded_columns.items()},
            "required_columns": {k: list(v) for k, v in self.required_columns.items()},
        }


def _read_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        text = (
            resources.files(f"{__package__}.resources")
            .joinpath(DEFAULT_SCHEMA_RESOURCE)
            .read_text(encoding="utf-8")
        )
    else:
        if not path.is_file():
            raise ConfigError(f"Schema file not found: {path}")
        text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigError("Schema file must contain a YAML mapping.")
    return data


def load_schema(path: str | Path | None = None) -> FeatureSchema:
    """Load the feature schema, defaulting to the packaged versioned copy."""
    resolved = Path(path) if path is not None else None
    data = _read_yaml(resolved)

    try:
        schema = FeatureSchema(
            schema_version=int(data["schema_version"]),
            target_column=str(data["target_column"]),
            status_column=str(data["status_column"]),
            label_mapping={str(k): int(v) for k, v in data["label_mapping"].items()},
            valid_statuses=tuple(str(s) for s in data["valid_statuses"]),
            accepted_status=str(data["accepted_status"]),
            feature_columns=tuple(str(c) for c in data["feature_columns"]),
            excluded_columns={
                str(k): tuple(str(c) for c in v)
                for k, v in (data.get("excluded_columns") or {}).items()
            },
            required_columns={
                str(k): tuple(str(c) for c in (v or []))
                for k, v in (data.get("required_columns") or {}).items()
            },
            source=resolved,
        )
    except KeyError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"Schema file is missing required key: {exc}") from exc

    _check_schema_consistency(schema)
    return schema


def _check_schema_consistency(schema: FeatureSchema) -> None:
    features = set(schema.feature_columns)
    if len(features) != len(schema.feature_columns):
        dupes = sorted({c for c in schema.feature_columns if schema.feature_columns.count(c) > 1})
        raise ConfigError(f"Duplicate entries in feature_columns: {dupes}")

    overlap = features.intersection(schema.withheld_columns)
    if overlap:
        raise ConfigError(
            f"Columns appear in both feature_columns and excluded_columns: {sorted(overlap)}"
        )

    if schema.accepted_status not in schema.valid_statuses:
        raise ConfigError(
            f"accepted_status {schema.accepted_status!r} is not in valid_statuses "
            f"{list(schema.valid_statuses)}."
        )

    if set(schema.label_mapping.values()) != {0, 1}:
        raise ConfigError("label_mapping must map onto exactly the values {0, 1}.")

    known = set(schema.known_columns)
    for mode, required in schema.required_columns.items():
        missing = [c for c in required if c not in known]
        if missing:
            raise ConfigError(
                f"required_columns[{mode!r}] references unknown columns: {missing}"
            )
