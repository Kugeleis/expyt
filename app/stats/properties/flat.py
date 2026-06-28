from typing import cast

import pandas as pd
import scipy.stats as stats

from app.stats.models import (
    DataProperties,
    MissingColumnSummary,
    MissingDataSummary,
    MissingnessAssociationResult,
    OutlierSummary,
)
from app.stats.properties.assumptions import (
    compute_expected_cell_counts,
    compute_normality,
    compute_sphericity,
    compute_variance_homogeneity,
    guess_outcome_type,
)


def compute_missing_summary(df: pd.DataFrame, outcome_col: str, group_col: str) -> MissingDataSummary:
    """Compute per-column missing metrics."""
    n_total = len(df)

    outcome_missing_count = int(df[outcome_col].isna().sum()) if n_total > 0 else 0
    outcome_missing_pct = float(outcome_missing_count / n_total * 100) if n_total > 0 else 0.0

    group_missing_count = int(df[group_col].isna().sum()) if n_total > 0 else 0
    group_missing_pct = float(group_missing_count / n_total * 100) if n_total > 0 else 0.0

    valid_group_df = df[df[group_col].notna()]
    if valid_group_df.empty:
        association = MissingnessAssociationResult(
            test_used="Chi-Square",
            statistic=None,
            p_value=None,
            significant=None,
            note="No valid groups to check missingness association.",
        )
    else:
        missing_mask = valid_group_df[outcome_col].isna()
        contingency = pd.crosstab(valid_group_df[group_col], missing_mask)
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            association = MissingnessAssociationResult(
                test_used="Chi-Square",
                statistic=None,
                p_value=None,
                significant=None,
                note="No missing values (or all missing values) to calculate association.",
            )
        else:
            try:
                chi2, p_val, _, _ = stats.chi2_contingency(contingency.values)
                chi2_val = cast(float, chi2)
                p_val_float = cast(float, p_val)
                association = MissingnessAssociationResult(
                    test_used="Chi-Square",
                    statistic=chi2_val,
                    p_value=p_val_float,
                    significant=p_val_float < 0.05,
                    note=None,
                )
            except Exception as e:
                association = MissingnessAssociationResult(
                    test_used="Chi-Square",
                    statistic=None,
                    p_value=None,
                    significant=None,
                    note=f"Association test failed: {e}",
                )

    return MissingDataSummary(
        outcome_missing=MissingColumnSummary(count=outcome_missing_count, percentage=outcome_missing_pct),
        group_missing=MissingColumnSummary(count=group_missing_count, percentage=group_missing_pct),
        association=association,
    )


def compute_outliers(df: pd.DataFrame, outcome_col: str, group_col: str) -> dict[str, OutlierSummary]:
    """Compute outliers per group using IQR rule."""
    results = {}
    grouped = df.groupby(group_col)
    for name, group_df in grouped:
        name_str = str(name)
        group_series = group_df[outcome_col].dropna()
        if len(group_series) < 4:
            results[name_str] = OutlierSummary(count=0, indices=[])
            continue
        q1 = group_series.quantile(0.25)
        q3 = group_series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_mask = (group_series < lower_bound) | (group_series > upper_bound)
        outliers_series = group_series[outliers_mask]
        results[name_str] = OutlierSummary(
            count=len(outliers_series),
            indices=outliers_series.index.tolist(),
        )
    return results


def compute_data_properties(
    df: pd.DataFrame,
    outcome_col: str,
    group_col: str,
    repeated_measures: bool = False,
    n_conditions: int | None = None,
) -> DataProperties:
    """Compute properties of the data to evaluate statistical test applicability."""
    if df.empty:
        raise ValueError("DataFrame is empty.")
    if group_col not in df.columns:
        raise ValueError(f"Group column {group_col!r} not found in DataFrame.")
    if outcome_col not in df.columns:
        raise ValueError(f"Value column {outcome_col!r} not found in DataFrame.")

    clean_df = df[[group_col, outcome_col]].dropna()
    grouped = clean_df.groupby(group_col)[outcome_col]
    group_sizes = {str(k): len(v) for k, v in grouped if len(v) > 0}
    n_groups = len(group_sizes)

    outcome_type = guess_outcome_type(df[outcome_col])

    sampled = False
    if len(df) > 50000:
        df_sampled = df.sample(n=50000, random_state=42)
        sampled = True
    else:
        df_sampled = df

    if outcome_type == "continuous":
        normality = compute_normality(df_sampled, outcome_col, group_col)
        all_groups_normal = all(g.is_normal for g in normality.values()) if normality else False
        variance_homogeneity = compute_variance_homogeneity(df_sampled, outcome_col, group_col)
        expected_cell_counts = None
        min_expected_cell_count = None
    else:
        normality = {}
        all_groups_normal = False
        variance_homogeneity = None
        expected_cell_counts, min_expected_cell_count = compute_expected_cell_counts(df, outcome_col, group_col)

    sphericity = compute_sphericity(df, outcome_col, group_col, repeated_measures, n_conditions)
    missing = compute_missing_summary(df, outcome_col, group_col)
    outliers = compute_outliers(df_sampled, outcome_col, group_col) if outcome_type == "continuous" else {}

    small_groups = [g for g, size in group_sizes.items() if size < 5]
    if small_groups:
        sample_size_warning = (
            f"Warning: The following groups have small sample sizes (n < 5): {', '.join(small_groups)}."
        )
    else:
        sample_size_warning = None

    return DataProperties(
        outcome_type_guess=outcome_type,
        n_groups=n_groups,
        group_sizes=group_sizes,
        normality=normality,
        all_groups_normal=all_groups_normal,
        variance_homogeneity=variance_homogeneity,
        expected_cell_counts=expected_cell_counts,
        min_expected_cell_count=min_expected_cell_count,
        sphericity=sphericity,
        missing=missing,
        outliers=outliers,
        sample_size_warning=sample_size_warning,
        sampled=sampled,
    )


def compute_data_properties_for_columns(
    df: pd.DataFrame,
    group_col: str,
    value_columns: list[str],
    repeated_measures: bool = False,
    n_conditions: int | None = None,
) -> dict[str, DataProperties]:
    """Compute data properties for multiple numeric value columns."""
    return {
        value_col: compute_data_properties(
            df,
            outcome_col=value_col,
            group_col=group_col,
            repeated_measures=repeated_measures,
            n_conditions=n_conditions,
        )
        for value_col in value_columns
    }
