"""Statistical method base classes, registries, and data schemas.

Re-exports all components from models and properties modules to maintain
100% backward compatibility for imports from app.stats.base.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.core.registry import Registry
from app.stats.models import (
    DataProperties,
    MissingColumnSummary,
    MissingDataSummary,
    MissingnessAssociationResult,
    NormalityResult,
    OutlierSummary,
    SphericityResult,
    StatResult,
    VarianceHomogeneityResult,
    _convert_old_normality,
    _convert_old_variance_homogeneity,
    _get_all_groups_normal,
    _set_defaults_for_missing_fields,
)
from app.stats.properties import (
    compute_data_properties,
    compute_data_properties_for_columns,
    compute_expected_cell_counts,
    compute_missing_summary,
    compute_normality,
    compute_outliers,
    compute_properties,
    compute_properties_for_columns,
    compute_sphericity,
    compute_variance_homogeneity,
    guess_outcome_type,
)


class StatMethod(ABC):
    """Abstract base class for all statistical evaluation methods."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of the statistical method."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a brief description of the statistical method."""
        ...

    @abstractmethod
    def is_applicable(self, properties: DataProperties) -> bool:
        """Determine whether this method is applicable to the given data properties.

        Args:
            properties: A DataProperties object containing data properties.

        Returns:
            True if the method can be run on this data, False otherwise.
        """
        ...

    @abstractmethod
    def run(self, groups: Any) -> StatResult:
        """Run the statistical method on the grouped data.

        To avoid breaking existing flat-data plugins and maintain complete backwards compatibility,
        the `groups` parameter is typed as `Any` at the base class level. This allows subclasses
        to type `groups` either as `dict[str, list[float]]` (flat mode) or as `HierarchicalData`
        (hierarchical mode) without causing type checker (mypy) override mismatches. Hierarchical
        plugins must check if the input is an instance of `HierarchicalData` and handle it accordingly.

        Args:
            groups: A dictionary mapping group names to lists of numeric values (for flat mode)
                or a HierarchicalData instance (for hierarchical mode).

        Returns:
            A StatResult containing the test statistic, p-value, etc.
        """
        ...


stat_registry: Registry[StatMethod] = Registry("stat")

__all__ = [
    # Models
    "NormalityResult",
    "VarianceHomogeneityResult",
    "SphericityResult",
    "MissingColumnSummary",
    "MissingnessAssociationResult",
    "MissingDataSummary",
    "OutlierSummary",
    "DataProperties",
    "StatResult",
    "_convert_old_normality",
    "_convert_old_variance_homogeneity",
    "_get_all_groups_normal",
    "_set_defaults_for_missing_fields",
    # Base/Registry
    "StatMethod",
    "stat_registry",
    # Properties/Computations
    "guess_outcome_type",
    "compute_normality",
    "compute_variance_homogeneity",
    "compute_expected_cell_counts",
    "compute_sphericity",
    "compute_missing_summary",
    "compute_outliers",
    "compute_data_properties",
    "compute_data_properties_for_columns",
    "compute_properties",
    "compute_properties_for_columns",
]
