"""Typed errors raised by the package.

Every error carries an actionable message: what was wrong, where, and what the
caller should do about it.
"""

from __future__ import annotations


class TransitExoplanetMLError(Exception):
    """Base class for all errors raised by this package."""


class DataValidationError(TransitExoplanetMLError):
    """A dataset violated the data contract."""


class EmptyDatasetError(DataValidationError):
    """A data directory contained no candidate CSV files."""


class SchemaVersionError(TransitExoplanetMLError):
    """A persisted artifact was built against an incompatible schema."""


class DataDiversityError(TransitExoplanetMLError):
    """The dataset lacks the class/star support required for grouped CV."""


class ConfigError(TransitExoplanetMLError):
    """The resolved configuration is invalid."""
