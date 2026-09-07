"""Per-learner preprocessing pipelines.

Each base learner gets its own pipeline so that a transformation appropriate to
one family is never forced on another: LightGBM keeps raw values and uses its
native missing-value handling, the tree ensemble gets imputation with missing
indicators, and the two distance/coefficient based learners additionally get
robust scaling.

Every pipeline is fitted inside the fold that uses it and persisted with the
model bundle.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler


class DropDegenerateFeatures(BaseEstimator, TransformerMixin):
    """Drop features that are constant or entirely missing within the fit fold.

    Fold-local constants carry no signal and break scale-based transforms (a
    zero inter-quartile range), so they are removed before imputation. The
    columns kept are recorded at fit time and reapplied verbatim afterwards.
    """

    def __init__(self, tol: float = 0.0) -> None:
        self.tol = tol

    def fit(self, X, y=None, sample_weight=None):
        data = np.asarray(X, dtype=float)
        if data.ndim != 2:
            raise ValueError(f"Expected a 2D array, got shape {data.shape}.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
            lo = np.nanmin(data, axis=0)
            hi = np.nanmax(data, axis=0)
        all_missing = np.isnan(data).all(axis=0)
        spread = np.where(all_missing, 0.0, hi - lo)
        self.support_ = ~(all_missing | (spread <= self.tol))
        if not self.support_.any():
            # Degenerate fold: keep one column so downstream steps have input.
            self.support_ = np.zeros(data.shape[1], dtype=bool)
            self.support_[0] = True
        self.n_features_in_ = data.shape[1]
        return self

    def transform(self, X):
        data = np.asarray(X, dtype=float)
        if data.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {data.shape[1]}."
            )
        return data[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        names = (
            np.asarray(input_features, dtype=object)
            if input_features is not None
            else np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        )
        return names[self.support_]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


def _identity_pipeline() -> Pipeline:
    return Pipeline([("passthrough", FunctionTransformer(validate=False))])


def _imputed_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("drop_degenerate", DropDegenerateFeatures()),
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ]
    )


def _scaled_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("drop_degenerate", DropDegenerateFeatures()),
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", RobustScaler()),
        ]
    )


_BUILDERS = {
    "lightgbm": _identity_pipeline,
    "catboost": _identity_pipeline,
    "extra_trees": _imputed_pipeline,
    "svm_rbf": _scaled_pipeline,
    "logistic_regression": _scaled_pipeline,
}


def build_preprocessor(learner: str) -> Pipeline:
    """Return an unfitted preprocessing pipeline for ``learner``."""
    try:
        return _BUILDERS[learner]()
    except KeyError as exc:
        raise KeyError(
            f"No preprocessing pipeline for {learner!r}; known learners: {sorted(_BUILDERS)}."
        ) from exc
