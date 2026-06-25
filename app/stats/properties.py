"""Statistical data properties computation functions."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pandas as pd
import scipy.stats as stats

from app.core.session import HierarchyConfig, WizardSession
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


def build_cluster_aggregates(
    unit_df: pd.DataFrame,
    config: HierarchyConfig,
    excluded_clusters: list[str],
    metric: str,
    metric_kind: Literal["continuous", "binary_proportion", "unsupported"],
) -> pd.DataFrame:
    """Build cluster-level aggregates from unit-level data."""
    if metric_kind == "unsupported":
        return pd.DataFrame(columns=[config.group_col, config.cluster_col])
    # Filter out excluded clusters and missing values
    filtered = unit_df[~unit_df[config.cluster_col].astype(str).isin(excluded_clusters)]
    filtered = filtered.dropna(subset=[config.group_col, config.cluster_col, metric])

    if filtered.empty:
        if metric_kind == "continuous":
            cols = [config.group_col, config.cluster_col, "mean", "median", "std", "iqr", "p5", "p95", "n_units"]
        else:
            cols = [config.group_col, config.cluster_col, "n_pass", "n_units", "proportion_raw", "proportion_corrected"]
        return pd.DataFrame(columns=cols)

    # Group by group and cluster
    grouped = filtered.groupby([config.group_col, config.cluster_col])

    if metric_kind == "continuous":
        agg_df = (
            grouped[metric]
            .agg(
                mean="mean",
                median="median",
                std="std",
                iqr=lambda x: float(x.quantile(0.75) - x.quantile(0.25)),
                p5=lambda x: float(x.quantile(0.05)),
                p95=lambda x: float(x.quantile(0.95)),
                n_units="count",
            )
            .reset_index()
        )
    else:
        # binary proportion
        agg_df = grouped[metric].agg(n_pass="sum", n_units="count").reset_index()
        agg_df["proportion_raw"] = agg_df["n_pass"] / agg_df["n_units"]
        agg_df["proportion_corrected"] = np.clip(
            agg_df["proportion_raw"], 0.5 / agg_df["n_units"], 1.0 - 0.5 / agg_df["n_units"]
        )

    return agg_df


def run_iterative_grubbs(df: pd.DataFrame, cluster_col: str, value_col: str, alpha: float = 0.05) -> list[str]:
    """Iteratively detect outliers using Grubbs test on cluster means."""
    temp_df = df[[cluster_col, value_col]].dropna().copy()
    outliers = []

    while len(temp_df) >= 3:
        n = len(temp_df)
        mean_val = temp_df[value_col].mean()
        std_val = temp_df[value_col].std()
        if std_val == 0 or np.isnan(std_val):
            break

        # Find absolute deviations
        temp_df["dev"] = (temp_df[value_col] - mean_val).abs()
        max_idx = temp_df["dev"].idxmax()
        max_row = temp_df.loc[max_idx]
        g_val = max_row["dev"] / std_val

        # t-distribution critical value
        t_dist = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_dist**2 / (n - 2 + t_dist**2))

        if g_val > g_crit:
            outliers.append(str(max_row[cluster_col]))
            temp_df = temp_df.drop(max_idx)
        else:
            break

    return outliers


def compute_quick_icc(df: pd.DataFrame, cluster_col: str, metric: str) -> float:
    """Calculate Intra-class Correlation (ICC) using a quick null LMM fit."""
    import statsmodels.formula.api as smf

    clean_df = df[[cluster_col, metric]].dropna()
    if clean_df[cluster_col].nunique() < 2 or len(clean_df) < 3:
        return float("nan")
    try:
        model = smf.mixedlm(f"{metric} ~ 1", clean_df, groups=clean_df[cluster_col])
        result = model.fit()
        var_w = result.scale
        var_b = result.cov_re.iloc[0, 0]
        if var_b + var_w == 0:
            return float("nan")
        return float(var_b / (var_b + var_w))
    except Exception:
        return float("nan")


def compute_hierarchical_properties(  # noqa: C901
    df: pd.DataFrame,
    config: HierarchyConfig,
    excluded_clusters: list[str],
    metric: str,
) -> DataProperties:
    """Compute properties for a hierarchical dataset."""
    # 1. Determine metric kind
    unique_vals = set(df[metric].dropna().unique())
    is_numeric = pd.api.types.is_numeric_dtype(df[metric]) or pd.api.types.is_bool_dtype(df[metric])
    if unique_vals.issubset({0, 1}) and len(unique_vals) > 0:
        metric_kind: Literal["continuous", "binary_proportion", "unsupported"] = "binary_proportion"
    elif is_numeric:
        metric_kind = "continuous"
    else:
        metric_kind = "unsupported"

    # 2. Build cluster aggregates
    agg_df = build_cluster_aggregates(df, config, excluded_clusters, metric, metric_kind)

    # 3. Structural properties
    clean_unit = df[~df[config.cluster_col].astype(str).isin(excluded_clusters)]
    clean_unit = clean_unit.dropna(subset=[config.group_col, config.cluster_col, metric])

    group_sizes = {}
    grouped_unit = clean_unit.groupby(config.group_col)
    for g_name, g_df in grouped_unit:
        group_sizes[str(g_name)] = len(g_df)

    n_groups = len(group_sizes)

    n_clusters_per_group = {}
    if not agg_df.empty:
        n_clusters_per_group = agg_df.groupby(config.group_col)[config.cluster_col].nunique().to_dict()
    n_clusters_per_group = {str(k): int(v) for k, v in n_clusters_per_group.items()}

    min_clusters_per_group = min(n_clusters_per_group.values()) if n_clusters_per_group else 0

    n_units_per_cluster = {}
    if not agg_df.empty:
        n_units_per_cluster = agg_df.set_index(config.cluster_col)["n_units"].to_dict()
    n_units_per_cluster = {str(k): int(v) for k, v in n_units_per_cluster.items()}

    # 4. Continuous path properties
    normality_cluster_means = False
    homogeneity_cluster = False
    outlier_clusters: list[str] = []
    normality_results = {}
    var_homogeneity_result = None

    if metric_kind == "continuous":
        # Normality on cluster means
        normality_cluster_means = True
        if not agg_df.empty:
            grouped_agg = agg_df.groupby(config.group_col)
            for g_name, g_df in grouped_agg:
                vals = g_df["mean"].dropna().values
                n = len(vals)
                if n < 3:
                    is_normal = False
                    test_used = "None"
                    p_val = None
                    note = "Insufficient clusters (n < 3)"
                    normality_cluster_means = False
                else:
                    test_used = "Shapiro-Wilk"
                    try:
                        _, p_val_stat = stats.shapiro(vals)
                        p_val = float(p_val_stat)
                        is_normal = p_val > 0.05
                        note = None
                        if not is_normal:
                            normality_cluster_means = False
                    except Exception as e:
                        p_val = None
                        is_normal = False
                        note = f"Shapiro-Wilk failed: {e}"
                        normality_cluster_means = False

                normality_results[str(g_name)] = NormalityResult(
                    test_used=test_used, p_value=p_val, n=n, is_normal=is_normal, note=note
                )
        else:
            normality_cluster_means = False

        # Homogeneity on cluster means
        if not agg_df.empty:
            grouped_agg = agg_df.groupby(config.group_col)
            group_means = [g_df["mean"].dropna().values for _, g_df in grouped_agg]
            group_means = [arr for arr in group_means if len(arr) >= 2]
            if len(group_means) >= 2:
                try:
                    stat, p_val = stats.levene(*group_means, center="median")
                    p_val_float = float(p_val)
                    if not np.isnan(p_val_float):
                        homogeneity_cluster = p_val_float > 0.05
                        var_homogeneity_result = VarianceHomogeneityResult(
                            test_used="Levene",
                            statistic=float(stat),
                            p_value=p_val_float,
                            equal_variances=homogeneity_cluster,
                        )
                except Exception:
                    pass

        # Outliers on cluster means via Grubbs
        if min_clusters_per_group >= 4 and not agg_df.empty:
            outlier_clusters = run_iterative_grubbs(agg_df, config.cluster_col, "mean")

    # 5. Binary proportion path properties
    has_boundary_clusters = False
    boundary_cluster_ids: list[str] = []
    if metric_kind == "binary_proportion" and not agg_df.empty:
        boundary_mask = agg_df["proportion_raw"].isin([0.0, 1.0])
        boundary_cluster_ids = agg_df[boundary_mask][config.cluster_col].astype(str).tolist()
        has_boundary_clusters = len(boundary_cluster_ids) > 0

    # 6. Clustering strength
    icc = float("nan") if metric_kind == "unsupported" else compute_quick_icc(clean_unit, config.cluster_col, metric)

    # 7. Power
    power_at_observed_n = 0.0
    if min_clusters_per_group >= 2:
        try:
            from statsmodels.stats.power import TTestIndPower

            analysis = TTestIndPower()
            power_at_observed_n = float(
                analysis.solve_power(
                    effect_size=0.8, nobs1=min_clusters_per_group, ratio=1.0, alpha=0.05, alternative="two-sided"
                )
            )
        except Exception:
            pass
    power_warning = power_at_observed_n < 0.8

    # 8. Outliers dictionary
    outliers_dict = {}
    if metric_kind == "continuous" and not agg_df.empty:
        outliers_dict = {"outlier_clusters": OutlierSummary(count=len(outlier_clusters), indices=outlier_clusters)}

    # 9. Missing summary
    missing = compute_missing_summary(df, metric, config.group_col)

    # 10. Sample size warning
    small_groups = [g for g, size in group_sizes.items() if size < 5]
    sample_size_warning = None
    if small_groups:
        groups_str = ", ".join(small_groups)
        sample_size_warning = f"Warning: The following groups have small sample sizes (n < 5): {groups_str}."

    return DataProperties(
        outcome_type_guess="continuous" if metric_kind == "continuous" else "categorical_nominal",
        n_groups=n_groups,
        group_sizes=group_sizes,
        normality=normality_results,
        all_groups_normal=normality_cluster_means if metric_kind == "continuous" else False,
        variance_homogeneity=var_homogeneity_result,
        expected_cell_counts=None,
        min_expected_cell_count=None,
        sphericity=None,
        missing=missing,
        outliers=outliers_dict,
        sample_size_warning=sample_size_warning,
        sampled=False,
        has_hierarchy=True,
        has_spatial_coords=config.x_col is not None and config.y_col is not None,
        n_clusters_per_group=n_clusters_per_group,
        min_clusters_per_group=min_clusters_per_group,
        n_units_per_cluster=n_units_per_cluster,
        metric_kind=metric_kind,
        normality_cluster_means=normality_cluster_means,
        homogeneity_cluster=homogeneity_cluster,
        outlier_clusters=outlier_clusters,
        has_boundary_clusters=has_boundary_clusters,
        boundary_cluster_ids=boundary_cluster_ids,
        icc=icc,
        power_at_observed_n=power_at_observed_n,
        power_warning=power_warning,
    )


def compute_properties(
    session: WizardSession,
    df: pd.DataFrame,
    value_col: str,
) -> DataProperties:
    """Top-level dispatcher to compute data properties, supporting hierarchical and flat data."""
    if session.hierarchy is not None:
        excluded_ids = [ex.cluster_id for ex in session.excluded_clusters]
        props = compute_hierarchical_properties(df, session.hierarchy, excluded_ids, value_col)
        props.has_hierarchy = True
        props.has_spatial_coords = session.hierarchy.x_col is not None and session.hierarchy.y_col is not None
    else:
        props = compute_data_properties(df, value_col, session.group_column or "")
        props.has_hierarchy = False
    return props


def compute_properties_for_columns(
    session: WizardSession,
    df: pd.DataFrame,
    value_columns: list[str],
) -> dict[str, DataProperties]:
    """Compute data properties for multiple columns under hierarchical or flat session configuration."""
    return {col: compute_properties(session, df, col) for col in value_columns}
