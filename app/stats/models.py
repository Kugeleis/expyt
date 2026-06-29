"""Statistical schemas and Pydantic models."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_serializer, model_validator


class NormalityResult(BaseModel):
    """Result of a normality test for a single group."""

    test_used: str = Field(..., description="Name of the test used.")
    p_value: float | None = Field(None, description="Normality test p-value.")
    n: int = Field(..., description="Sample size of the group.")
    is_normal: bool = Field(..., description="Whether normality is assumed.")
    note: str | None = Field(None, description="Optional warning/error note.")


class VarianceHomogeneityResult(BaseModel):
    """Result of a variance homogeneity test."""

    test_used: str = Field("Levene", description="Name of the test used.")
    statistic: float = Field(..., description="Calculated test statistic.")
    p_value: float = Field(..., description="Calculated p-value.")
    equal_variances: bool = Field(..., description="Whether variances are homogeneous.")


class SphericityResult(BaseModel):
    """Result of a sphericity test (e.g. Mauchly)."""

    statistic: float = Field(..., description="Mauchly's W statistic.")
    p_value: float = Field(..., description="Calculated p-value.")
    sphericity_assumed: bool = Field(..., description="Whether sphericity is assumed.")
    note: str | None = Field(None, description="Optional explanation or warning.")


class MissingColumnSummary(BaseModel):
    """Summary of missing values in a column."""

    count: int = Field(..., description="Number of missing values.")
    percentage: float = Field(..., description="Percentage of missing values.")


class MissingnessAssociationResult(BaseModel):
    """Result of the association check between outcome missingness and group."""

    test_used: str = Field("Chi-Square", description="Name of the test used.")
    statistic: float | None = Field(None, description="Calculated test statistic.")
    p_value: float | None = Field(None, description="Calculated p-value.")
    significant: bool | None = Field(None, description="Whether the association is significant.")
    note: str | None = Field(None, description="Optional explanation or error note.")


class MissingDataSummary(BaseModel):
    """Summary of missing data in the dataset."""

    outcome_missing: MissingColumnSummary = Field(..., description="Missingness in the outcome column.")
    group_missing: MissingColumnSummary = Field(..., description="Missingness in the group column.")
    association: MissingnessAssociationResult = Field(..., description="Missingness association result.")


class OutlierSummary(BaseModel):
    """Summary of outliers in a single group."""

    count: int = Field(..., description="Number of outliers.")
    indices: list[Any] = Field(..., description="Indices of the outliers in the DataFrame.")


class DataProperties(BaseModel):
    """Properties characterizing a dataset for statistical method applicability."""

    outcome_type_guess: str = Field(
        "continuous",
        description=("Guessed outcome type (continuous, categorical_nominal, categorical_ordinal_unclear)."),
    )
    is_numeric: bool = Field(True, description="Whether the outcome column is numeric.")
    n_groups: int = Field(..., description="Number of distinct groups.")
    group_sizes: dict[str, int] = Field(..., description="Size of each group after removing NaNs.")
    normality: dict[str, NormalityResult] = Field(..., description="Normality test result for each group.")
    all_groups_normal: bool = Field(True, description="Whether all groups are normal.")
    variance_homogeneity: VarianceHomogeneityResult | None = Field(
        None, description="Levene's test result, or None if outcome is categorical."
    )
    expected_cell_counts: list[list[float]] | None = Field(None, description="Expected cell counts contingency table.")
    min_expected_cell_count: float | None = Field(None, description="Minimum expected cell count.")
    sphericity: SphericityResult | None = Field(None, description="Mauchly's sphericity test result.")
    missing: MissingDataSummary = Field(
        default_factory=lambda: MissingDataSummary(
            outcome_missing={"count": 0, "percentage": 0.0},
            group_missing={"count": 0, "percentage": 0.0},
            association={
                "test_used": "Chi-Square",
                "statistic": None,
                "p_value": None,
                "significant": None,
                "note": "Default values",
            },
        ),
        description="Missing data summary.",
    )
    outliers: dict[str, OutlierSummary] = Field(default_factory=dict, description="Outliers per group.")
    sample_size_warning: str | None = Field(None, description="Warning if any group has sample size < 5.")
    sampled: bool = Field(False, description="Whether the data was sampled.")

    # Hierarchical data properties
    has_hierarchy: bool = Field(False, description="Whether the dataset is hierarchical.")
    has_spatial_coords: bool = Field(False, description="Whether spatial coordinates are present.")
    n_clusters_per_group: dict[str, int] | None = Field(None, description="Number of clusters per group.")
    min_clusters_per_group: int | None = Field(None, description="Minimum clusters per group.")
    n_units_per_cluster: dict[str, int] | None = Field(None, description="Number of units per cluster.")
    metric_kind: str | None = Field(None, description="Kind of metric ('continuous' or 'binary_proportion').")
    normality_cluster_means: bool | None = Field(None, description="SW normality of cluster means.")
    homogeneity_cluster: bool | None = Field(None, description="Levene homogeneity of cluster means.")
    outlier_clusters: list[str] | None = Field(None, description="Grubbs flagged cluster IDs.")
    has_boundary_clusters: bool | None = Field(None, description="Any cluster with proportion 0 or 1.")
    boundary_cluster_ids: list[str] | None = Field(None, description="Boundary cluster IDs.")
    icc: float | None = Field(None, description="Intra-class correlation.")
    power_at_observed_n: float | None = Field(None, description="Power at observed n clusters.")
    power_warning: bool | None = Field(None, description="Whether power is < 0.8.")

    @model_validator(mode="before")
    @classmethod
    def convert_old_format(cls, data: Any) -> Any:
        """Validator to map the old properties structure to the new nested format."""
        if not isinstance(data, dict):
            return data

        # Shallow copy to avoid side-effects
        data = dict(data)
        _convert_old_normality(data)
        _convert_old_variance_homogeneity(data)
        _set_defaults_for_missing_fields(data)

        return data


def _convert_old_normality(data: dict[str, Any]) -> None:
    norm = data.get("normality")
    if not isinstance(norm, dict):
        return
    new_norm = {}
    for k, v in norm.items():
        if isinstance(v, (int, float)):
            g_sizes = data.get("group_sizes", {})
            n_val = g_sizes.get(k, 10) if isinstance(g_sizes, dict) else 10
            new_norm[k] = {
                "test_used": "Shapiro-Wilk",
                "p_value": float(v),
                "n": n_val,
                "is_normal": float(v) > 0.05,
            }
        else:
            new_norm[k] = v
    data["normality"] = new_norm


def _convert_old_variance_homogeneity(data: dict[str, Any]) -> None:
    vh = data.get("variance_homogeneity")
    if isinstance(vh, (int, float)):
        data["variance_homogeneity"] = {
            "test_used": "Levene",
            "statistic": 0.0,
            "p_value": float(vh),
            "equal_variances": float(vh) > 0.05,
        }


def _get_all_groups_normal(norm: Any) -> bool:
    if not isinstance(norm, dict):
        return False
    for v in norm.values():
        if isinstance(v, dict):
            if not v.get("is_normal", False):
                return False
        elif hasattr(v, "is_normal"):
            if not v.is_normal:
                return False
        else:
            return False
    return True


def _set_defaults_for_missing_fields(data: dict[str, Any]) -> None:
    if "outcome_type_guess" not in data:
        data["outcome_type_guess"] = "continuous"
    if "all_groups_normal" not in data:
        data["all_groups_normal"] = _get_all_groups_normal(data.get("normality", {}))
    if "missing" not in data:
        data["missing"] = {
            "outcome_missing": {"count": 0, "percentage": 0.0},
            "group_missing": {"count": 0, "percentage": 0.0},
            "association": {
                "test_used": "Chi-Square",
                "statistic": None,
                "p_value": None,
                "significant": None,
                "note": "Default values",
            },
        }
    if "outliers" not in data:
        data["outliers"] = {}
    if "sampled" not in data:
        data["sampled"] = False


class StatResult(BaseModel):
    """The result of executing a statistical method."""

    column_name: str | None = Field(
        None,
        description=("Name of the dependent variable column analyzed by this statistical result."),
    )

    method_name: str = Field(..., description="Name of the statistical method.")
    test_statistic: float = Field(..., description="Calculated test statistic.")
    p_value: float = Field(..., description="Calculated p-value.")
    effect_size: float | None = Field(None, description="Optional calculated effect size.")
    summary: str = Field(..., description="Human-readable text summary of results.")

    # Hierarchical extensions
    icc: float | None = Field(None, description="Intra-class correlation.")
    power: float | None = Field(None, description="Estimated power.")
    n_clusters_used: dict[str, int] | None = Field(None, description="Number of clusters used per group.")
    posthoc: Any | None = Field(None, description="Post-hoc pairwise comparisons.")
    flags: list[str] = Field(default_factory=list, description="DECISION items for UI.")

    model_config = {"arbitrary_types_allowed": True}

    @field_serializer("posthoc")
    def serialize_posthoc(self, posthoc: Any) -> Any:
        if isinstance(posthoc, pd.DataFrame):
            return posthoc.to_dict(orient="records")
        return posthoc
