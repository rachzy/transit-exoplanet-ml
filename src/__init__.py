"""Star-grouped stacked classifier for screening transit exoplanet candidates.

The public API mirrors the ``exoplanet-ml`` CLI:

* :func:`load_dataset` / :func:`validate_dataset` -- read and check processed CSVs
* :func:`evaluate_dataset` -- nested, star-grouped evaluation
* :func:`train_model` / :func:`load_model` -- fit and reload a model bundle
* :func:`predict_dataset` -- batch prediction into one consolidated frame
"""

from __future__ import annotations

from .config import Config, load_config
from .data import Dataset, load_dataset, validate_dataset
from .errors import (
    ConfigError,
    DataDiversityError,
    DataValidationError,
    EmptyDatasetError,
    SchemaVersionError,
    TransitExoplanetMLError,
)
from .evaluate import evaluate_dataset
from .predict import predict_dataset
from .schema import FeatureSchema, load_schema
from .training import ModelBundle, load_model, train_model

__version__ = "0.1.0"

__all__ = [
    "Config",
    "ConfigError",
    "DataDiversityError",
    "DataValidationError",
    "Dataset",
    "EmptyDatasetError",
    "FeatureSchema",
    "ModelBundle",
    "SchemaVersionError",
    "TransitExoplanetMLError",
    "__version__",
    "evaluate_dataset",
    "load_config",
    "load_dataset",
    "load_model",
    "load_schema",
    "predict_dataset",
    "train_model",
    "validate_dataset",
]
