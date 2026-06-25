"""Statistical data properties computation functions."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import scipy.stats as stats

from app.stats.models import (
    DataProperties,
    MissingColumnSummary,
    MissingDataSummary,
    MissingnessAssociationResult,
    NormalityResult,
    OutlierSummary,
    SphericityResult,
    VarianceHomogeneityResult,
)


def guess_outcome_type(series: pd.Series) -> str:
    """Guess the outcome column's statistical variable type.

    Numeric dtype with >10 unique values -> "continuous".
    Numeric or string dtype with <=10 unique values -> "categorical_nominal" by
    default. If integer with a small contiguous range (e.g. 1-5, 1-7), ->
    "categorical_ordinal_unclear".
    """
    clean_series = series.dropna()
    n_unique = clean_series.nunique()

    if n_unique == 0:
        return "categorical_nominal"

    is_numeric = pd.api.types.is_numeric_dtype(series)

    if is_numeric and n_unique > 10:
        return "continuous"

    if is_numeric:
        try:
            is_integer = bool(np.all(clean_series == clean_series.astype(int)))
        except (ValueError, TypeError, OverflowError):
            is_integer = False

        if is_integer:
            val_min = int(clean_series.min())
            val_max = int(clean_series.max())
            if val_max - val_min + 1 == n_unique:
                return "categorical_ordinal_unclear"

    return "categorical_nominal"


def compute_normality(df: pd.DataFrame, outcome_col: str, group_col: str) -> dict[str, NormalityResult]:
    """Compute Shapiro-Wilk or D'Agostino-Pearson normality tests per group."""
    results = {}
    grouped = df.groupby(group_col)[outcome_col]
    for name, group_series in grouped:
        name_str = str(name)
        vals = group_series.dropna().values
        n = len(vals)
        if n < 3:
            results[name_str] = NormalityResult(
                test_used="None",
                p_value=None,
                n=n,
                is_normal=False,
                note="Insufficient data (n < 3)",
            )
            continue

        if n <= 5000:
            test_used = "Shapiro-Wilk"
            try:
                _, p_val = stats.shapiro(vals)
                p_val_float = float(p_val)
                is_normal = p_val_float > 0.05
                note = None
            except Exception as e:
                p_val_float = None
                is_normal = False
                note = f"Shapiro-Wilk failed: {e}"
        else:
            test_used = "D'Agostino-Pearson"
            try:
                _, p_val = stats.normaltest(vals)
                p_val_float = float(p_val)
                is_normal = p_val_float > 0.05
                note = None
            except Exception as e:
                p_val_float = None
                is_normal = False
                note = f"D'Agostino-Pearson failed: {e}"

        results[name_str] = NormalityResult(
            test_used=test_used,
            p_value=p_val_float,
            n=n,
            is_normal=is_normal,
            note=note,
        )
    return results


def compute_variance_homogeneity(
    df: pd.DataFrame, outcome_col: str, group_col: str
) -> VarianceHomogeneityResult | None:
    """Compute Levene's homogeneity of variance test robustly across all groups."""
    grouped = df.groupby(group_col)[outcome_col]
    group_arrays = [group_series.dropna().values for _, group_series in grouped]
    group_arrays = [arr for arr in group_arrays if len(arr) > 0]

    if len(group_arrays) < 2:
        return None

    if any(len(arr) < 2 for arr in group_arrays):
        return VarianceHomogeneityResult(
            test_used="Levene",
            statistic=0.0,
            p_value=0.0,
            equal_variances=False,
        )

    try:
        stat, p_val = stats.levene(*group_arrays, center="median")
        p_val_float = float(p_val)
        if np.isnan(p_val_float):
            return VarianceHomogeneityResult(
                test_used="Levene",
                statistic=0.0,
                p_value=0.0,
                equal_variances=False,
            )
        return VarianceHomogeneityResult(
            test_used="Levene",
            statistic=float(stat),
            p_value=p_val_float,
            equal_variances=p_val_float > 0.05,
        )
    except Exception:
        return VarianceHomogeneityResult(
            test_used="Levene",
            statistic=0.0,
            p_value=0.0,
            equal_variances=False,
        )


def compute_expected_cell_counts(
    df: pd.DataFrame, outcome_col: str, group_col: str
) -> tuple[list[list[float]] | None, float | None]:
    """Compute expected cell counts contingent matrix and minimum expected count."""
    clean_df = df[[group_col, outcome_col]].dropna()
    if clean_df.empty:
        return None, None

    observed = pd.crosstab(clean_df[group_col], clean_df[outcome_col]).values
    if observed.size == 0 or observed.sum() == 0:
        return None, None

    try:
        _, _, _, expected = stats.chi2_contingency(observed)
        expected_list = expected.tolist()
        min_val = float(np.min(expected))
        return expected_list, min_val
    except Exception:
        row_totals = observed.sum(axis=1, keepdims=True)
        col_totals = observed.sum(axis=0, keepdims=True)
        n = observed.sum()
        if n == 0:
            return None, None
        expected = (row_totals @ col_totals) / n
        expected_list = expected.tolist()
        min_val = float(np.min(expected))
        return expected_list, min_val


def compute_sphericity(
    df: pd.DataFrame,
    outcome_col: str,
    group_col: str,
    repeated_measures: bool,
    n_conditions: int | None,
) -> SphericityResult | None:
    """Compute Mauchly's sphericity test for repeated measures."""
    if not repeated_measures:
        return None
    if n_conditions is None or n_conditions < 3:
        return None

    clean_df = df[[group_col, outcome_col]].dropna()
    grouped = clean_df.groupby(group_col)[outcome_col]
    unique_groups = list(grouped.groups.keys())
    p = len(unique_groups)

    if p < 3:
        return SphericityResult(
            statistic=1.0,
            p_value=1.0,
            sphericity_assumed=True,
            note="Sphericity mathematically guaranteed for < 3 conditions.",
        )

    groups_data = {str(name): group_series.values for name, group_series in grouped}
    lengths = [len(arr) for arr in groups_data.values()]
    n = min(lengths) if lengths else 0

    if n <= p:
        return SphericityResult(
            statistic=0.0,
            p_value=0.0,
            sphericity_assumed=False,
            note=(f"Insufficient subjects (n={n}) relative to conditions (p={p}) to perform Mauchly's test."),
        )

    X = np.column_stack([np.asarray(arr[:n], dtype=float) for arr in groups_data.values()])
    sigma = np.cov(X, rowvar=False)

    d = p - 1
    A = np.zeros((p, p))
    A[:, 0] = 1.0
    A[:, 1:] = np.eye(p)[:, :-1]
    try:
        Q, R = np.linalg.qr(A)
        M = Q[:, 1:]
    except Exception as e:
        return SphericityResult(
            statistic=0.0,
            p_value=0.0,
            sphericity_assumed=False,
            note=f"Error constructing contrast matrix: {e}",
        )

    try:
        S = M.T @ sigma @ M
        det_S = np.linalg.det(S)
        tr_S = np.trace(S)
    except Exception as e:
        return SphericityResult(
            statistic=0.0,
            p_value=0.0,
            sphericity_assumed=False,
            note=f"Error computing transformed covariance matrix: {e}",
        )

    if tr_S <= 0:
        return SphericityResult(
            statistic=0.0,
            p_value=0.0,
            sphericity_assumed=False,
            note="Zero trace of transformed covariance matrix.",
        )

    W = det_S / ((tr_S / d) ** d)
    W = float(np.clip(W, 1e-15, 1.0))

    df_chi2 = int(p * (p - 1) / 2 - 1)
    if df_chi2 <= 0:
        df_chi2 = 1

    correction = 1.0 - (2.0 * (d**2) + d + 2.0) / (6.0 * d * (n - 1))
    chi2_val = -(n - 1) * correction * np.log(W)
    chi2_val = float(max(0.0, chi2_val))

    p_val = float(stats.chi2.sf(chi2_val, df_chi2))

    return SphericityResult(
        statistic=W,
        p_value=p_val,
        sphericity_assumed=p_val > 0.05,
        note=None,
    )


def compute_missing_summary(df: pd.DataFrame, outcome_col: str, group_col: str) -> MissingDataSummary:
    """Compute per-column missing metrics.

    Includes a missingness-vs-group association chi-square test.
    """
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
                note=("No missing values (or all missing values) to calculate association."),
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
    """Compute properties of the data to evaluate statistical test applicability.

    Args:
        df: The dataset DataFrame.
        outcome_col: Column name representing the values (dependent variable).
        group_col: Column name representing the groups (independent variable).
        repeated_measures: Set True if repeated-measures design.
        n_conditions: Minimum number of conditions for repeated measures.

    Returns:
        A DataProperties object.

    Raises:
        ValueError: If columns are missing or DataFrame is empty.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")
    if group_col not in df.columns:
        raise ValueError(f"Group column {group_col!r} not found in DataFrame.")
    if outcome_col not in df.columns:
        raise ValueError(f"Value column {outcome_col!r} not found in DataFrame.")

    # Drop missing groups/outcomes for grouping purposes
    clean_df = df[[group_col, outcome_col]].dropna()

    # Exclude groups with zero rows after filtering
    grouped = clean_df.groupby(group_col)[outcome_col]
    group_sizes = {str(k): len(v) for k, v in grouped if len(v) > 0}
    n_groups = len(group_sizes)

    # Determine outcome type guess
    outcome_type = guess_outcome_type(df[outcome_col])

    # Sample if dataset is too large (> 50,000 rows) for performance
    sampled = False
    if len(df) > 50000:
        df_sampled = df.sample(n=50000, random_state=42)
        sampled = True
    else:
        df_sampled = df

    # Sub-computations
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

    # Sphericity check
    sphericity = compute_sphericity(df, outcome_col, group_col, repeated_measures, n_conditions)

    # Missing values summary
    missing = compute_missing_summary(df, outcome_col, group_col)

    # Outliers
    outliers = compute_outliers(df_sampled, outcome_col, group_col) if outcome_type == "continuous" else {}

    # Warning for sample size
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
    """Compute data properties for multiple numeric value columns.

    Args:
        df: The dataset DataFrame.
        group_col: Column name representing the groups.
        value_columns: List of columns to compute properties for.
        repeated_measures: Whether repeated measures are requested.
        n_conditions: Number of conditions for repeated measures.

    Returns:
        A mapping from column name to its computed DataProperties.
    """
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
