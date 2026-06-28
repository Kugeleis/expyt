from typing import Literal

import numpy as np
import pandas as pd
import scipy.stats as stats

from app.core.session import HierarchyConfig
from app.stats.models import DataProperties, NormalityResult, OutlierSummary, VarianceHomogeneityResult
from app.stats.properties.flat import compute_missing_summary


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

    filtered = unit_df[~unit_df[config.cluster_col].astype(str).isin(excluded_clusters)]
    filtered = filtered.dropna(subset=[config.group_col, config.cluster_col, metric])

    if filtered.empty:
        if metric_kind == "continuous":
            cols = [config.group_col, config.cluster_col, "mean", "median", "std", "iqr", "p5", "p95", "n_units"]
        else:
            cols = [config.group_col, config.cluster_col, "n_pass", "n_units", "proportion_raw", "proportion_corrected"]
        return pd.DataFrame(columns=cols)

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

        temp_df["dev"] = (temp_df[value_col] - mean_val).abs()
        max_idx = temp_df["dev"].idxmax()
        max_row = temp_df.loc[max_idx]
        g_val = max_row["dev"] / std_val

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
    unique_vals = set(df[metric].dropna().unique())
    is_numeric = pd.api.types.is_numeric_dtype(df[metric]) or pd.api.types.is_bool_dtype(df[metric])
    if unique_vals.issubset({0, 1}) and len(unique_vals) > 0:
        metric_kind: Literal["continuous", "binary_proportion", "unsupported"] = "binary_proportion"
    elif is_numeric:
        metric_kind = "continuous"
    else:
        metric_kind = "unsupported"

    agg_df = build_cluster_aggregates(df, config, excluded_clusters, metric, metric_kind)

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

    normality_cluster_means = False
    homogeneity_cluster = False
    outlier_clusters: list[str] = []
    normality_results = {}
    var_homogeneity_result = None

    if metric_kind == "continuous":
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

        if min_clusters_per_group >= 4 and not agg_df.empty:
            outlier_clusters = run_iterative_grubbs(agg_df, config.cluster_col, "mean")

    has_boundary_clusters = False
    boundary_cluster_ids: list[str] = []
    if metric_kind == "binary_proportion" and not agg_df.empty:
        boundary_mask = agg_df["proportion_raw"].isin([0.0, 1.0])
        boundary_cluster_ids = agg_df[boundary_mask][config.cluster_col].astype(str).tolist()
        has_boundary_clusters = len(boundary_cluster_ids) > 0

    icc = float("nan") if metric_kind == "unsupported" else compute_quick_icc(clean_unit, config.cluster_col, metric)

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

    outliers_dict = {}
    if metric_kind == "continuous" and not agg_df.empty:
        outliers_dict = {"outlier_clusters": OutlierSummary(count=len(outlier_clusters), indices=outlier_clusters)}

    missing = compute_missing_summary(df, metric, config.group_col)

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
