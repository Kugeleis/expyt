from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from app.core.session import ClusterExclusion, HierarchyConfig, WizardSession
from app.datasets.hierarchical import HierarchicalData
from app.filters.base import filter_registry
from app.plots.base import plot_registry
from app.stats.base import DataProperties, compute_properties, compute_properties_for_columns, stat_registry
from app.stats.properties import (
    build_cluster_aggregates,
    compute_hierarchical_properties,
)


def test_hierarchy_config_models() -> None:
    """Test the creation and validation of HierarchyConfig and ClusterExclusion."""
    config = HierarchyConfig(group_col="group", cluster_col="cluster", unit_col="unit", x_col="x", y_col="y")
    assert config.group_col == "group"
    assert config.cluster_col == "cluster"
    assert config.unit_col == "unit"
    assert config.x_col == "x"
    assert config.y_col == "y"

    exclusion = ClusterExclusion(cluster_id="Cluster_0", reason="outlier")
    assert exclusion.cluster_id == "Cluster_0"
    assert exclusion.reason == "outlier"

    # Reason must be non-empty
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ClusterExclusion(cluster_id="Cluster_0", reason="")


def test_build_cluster_aggregates(hierarchical_df: Any) -> None:
    """Test build_cluster_aggregates for continuous and binary metrics."""
    config = HierarchyConfig(group_col="group", cluster_col="cluster", unit_col="unit")

    # 1. Continuous
    df_cont = hierarchical_df(n_groups=2, n_clusters=4, n_units=10, is_binary=False)
    agg_cont = build_cluster_aggregates(df_cont, config, [], "metric", "continuous")
    assert not agg_cont.empty
    assert "mean" in agg_cont.columns
    assert "std" in agg_cont.columns
    assert len(agg_cont) == 8

    # 2. Binary
    df_bin = hierarchical_df(n_groups=2, n_clusters=4, n_units=10, is_binary=True)
    agg_bin = build_cluster_aggregates(df_bin, config, [], "metric", "binary_proportion")
    assert not agg_bin.empty
    assert "proportion_raw" in agg_bin.columns
    assert "proportion_corrected" in agg_bin.columns
    assert len(agg_bin) == 8


def test_compute_hierarchical_properties(hierarchical_df: Any) -> None:
    """Test compute_hierarchical_properties under balanced and imbalanced cases."""
    config = HierarchyConfig(group_col="group", cluster_col="cluster", unit_col="unit", x_col="x", y_col="y")

    # Balanced continuous
    df_balanced = hierarchical_df(n_groups=2, n_clusters=4, n_units=10, is_binary=False)
    props = compute_hierarchical_properties(df_balanced, config, [], "metric")
    assert props.has_hierarchy
    assert props.has_spatial_coords
    assert props.metric_kind == "continuous"
    assert props.min_clusters_per_group == 4
    assert props.icc is not None
    assert props.power_at_observed_n is not None
    assert not np.isnan(props.power_at_observed_n)

    # Imbalanced binary with boundary
    df_imbalanced = hierarchical_df(
        n_groups=2, n_clusters=4, n_units=10, is_binary=True, imbalanced=True, boundary=True
    )
    props_bin = compute_hierarchical_properties(df_imbalanced, config, [], "metric")
    assert props_bin.has_hierarchy
    assert props_bin.metric_kind == "binary_proportion"
    assert props_bin.has_boundary_clusters
    assert props_bin.boundary_cluster_ids is not None
    assert len(props_bin.boundary_cluster_ids) > 0


def test_dispatcher_routing(hierarchical_df: Any) -> None:
    """Test compute_properties and compute_properties_for_columns."""
    session = WizardSession(
        session_id="test_sess",
        group_column="group",
        selected_value_columns=["metric"],
    )
    df = hierarchical_df(n_groups=2, n_clusters=4, n_units=10)

    # Flat mode
    props_flat = compute_properties(session, df, "metric")
    assert not props_flat.has_hierarchy

    # Hierarchical mode
    session.hierarchy = HierarchyConfig(group_col="group", cluster_col="cluster", unit_col="unit")
    props_hier = compute_properties(session, df, "metric")
    assert props_hier.has_hierarchy

    # Columns dispatcher
    props_map = compute_properties_for_columns(session, df, ["metric"])
    assert "metric" in props_map
    assert props_map["metric"].has_hierarchy


def test_cluster_exclusion_filter(hierarchical_df: Any) -> None:
    """Test ClusterExclusionFilter registrations, applicability, and validation."""
    filt: Any = filter_registry.get("cluster_exclusion")
    assert filt.name == "cluster_exclusion"

    props = DataProperties(
        outcome_type_guess="continuous",
        n_groups=2,
        group_sizes={"Group_A": 10, "Group_B": 10},
        normality={},
        all_groups_normal=True,
        variance_homogeneity=None,
        missing={
            "outcome_missing": {"count": 0, "percentage": 0.0},
            "group_missing": {"count": 0, "percentage": 0.0},
            "association": {
                "test_used": "Chi-Square",
                "statistic": None,
                "p_value": None,
                "significant": None,
                "note": "",
            },
        },
        has_hierarchy=True,
    )

    # Applicability
    assert filt.is_applicable(props)

    # Validation
    with pytest.raises(ValueError):
        filt.validate_params({})
    with pytest.raises(ValueError):
        filt.validate_params({"exclusions": "invalid"})
    with pytest.raises(ValueError):
        filt.validate_params({"exclusions": [{"cluster_id": "C1"}]})  # missing reason
    with pytest.raises(ValueError):
        filt.validate_params({"exclusions": [{"cluster_id": "C1", "reason": ""}]})  # empty reason

    filt.validate_params({"exclusions": [{"cluster_id": "C1", "reason": "broken"}]})

    # Apply (should return df unchanged)
    df = hierarchical_df()
    df_filtered = filt.apply(df, {"exclusions": [{"cluster_id": "Cluster_0", "reason": "broken"}]})
    assert len(df) == len(df_filtered)


def test_hierarchical_stat_methods(hierarchical_df: Any) -> None:
    """Test all 6 hierarchical statistical method plugins."""
    config = HierarchyConfig(group_col="group", cluster_col="cluster", unit_col="unit")

    # 1. Balanced Continuous Data
    df_cont = hierarchical_df(n_groups=2, n_clusters=4, n_units=10, is_binary=False)
    agg_cont = build_cluster_aggregates(df_cont, config, [], "metric", "continuous")
    h_data_cont = HierarchicalData(
        unit_df=df_cont,
        cluster_agg=agg_cont,
        config=config,
        excluded_clusters=[],
        metric="metric",
        metric_kind="continuous",
        icc=0.1,
    )

    # 1a. ANOVA
    anova = stat_registry.get("cluster_mean_anova")
    res_anova = anova.run(h_data_cont)
    assert res_anova.method_name == "cluster_mean_anova"
    assert res_anova.p_value >= 0.0
    assert res_anova.effect_size is not None
    assert res_anova.power is not None
    assert res_anova.n_clusters_used == {"Group_A": 4, "Group_B": 4}

    # 1b. Kruskal-Wallis
    kw = stat_registry.get("cluster_mean_kruskal_wallis")
    res_kw = kw.run(h_data_cont)
    assert res_kw.method_name == "cluster_mean_kruskal_wallis"
    assert res_kw.p_value >= 0.0
    assert res_kw.effect_size is not None

    # 1c. Linear Mixed Model
    lmm = stat_registry.get("linear_mixed_model")
    res_lmm = lmm.run(h_data_cont)
    assert res_lmm.method_name == "linear_mixed_model"
    assert res_lmm.p_value >= 0.0
    assert res_lmm.icc is not None
    assert res_lmm.posthoc is not None

    # 1d. Levene Cluster Uniformity
    levene = stat_registry.get("levene_cluster_uniformity")
    res_levene = levene.run(h_data_cont)
    assert res_levene.method_name == "levene_cluster_uniformity"
    assert res_levene.p_value >= 0.0
    assert res_levene.n_clusters_used == {"Group_A": 4, "Group_B": 4}

    # 1e. Grubbs Outlier Diagnostic
    grubbs = stat_registry.get("grubbs_cluster_outlier")
    res_grubbs = grubbs.run(h_data_cont)
    assert res_grubbs.method_name == "grubbs_cluster_outlier"
    assert res_grubbs.p_value >= 0.0
    assert isinstance(res_grubbs.flags, list)

    # 2. Balanced Binary Data
    df_bin = hierarchical_df(n_groups=2, n_clusters=4, n_units=10, is_binary=True)
    agg_bin = build_cluster_aggregates(df_bin, config, [], "metric", "binary_proportion")
    h_data_bin = HierarchicalData(
        unit_df=df_bin,
        cluster_agg=agg_bin,
        config=config,
        excluded_clusters=[],
        metric="metric",
        metric_kind="binary_proportion",
        icc=0.1,
    )

    # 2a. Proportion Kruskal-Wallis
    prop_kw = stat_registry.get("proportion_kruskal_wallis")
    res_prop = prop_kw.run(h_data_bin)
    assert res_prop.method_name == "proportion_kruskal_wallis"
    assert res_prop.p_value >= 0.0
    assert res_prop.posthoc is not None

    # 3. Imbalanced & Boundary Cases
    df_imb_boundary = hierarchical_df(
        n_groups=2, n_clusters=4, n_units=10, is_binary=True, imbalanced=True, boundary=True
    )
    agg_imb = build_cluster_aggregates(df_imb_boundary, config, [], "metric", "binary_proportion")
    h_data_imb = HierarchicalData(
        unit_df=df_imb_boundary,
        cluster_agg=agg_imb,
        config=config,
        excluded_clusters=[],
        metric="metric",
        metric_kind="binary_proportion",
        icc=0.1,
    )
    res_prop_imb = prop_kw.run(h_data_imb)
    assert res_prop_imb.p_value >= 0.0


def test_hierarchical_plots(hierarchical_df: Any) -> None:
    """Test the three hierarchical plotting plugins."""
    config = HierarchyConfig(group_col="group", cluster_col="cluster", unit_col="unit", x_col="x", y_col="y")
    df = hierarchical_df(n_groups=2, n_clusters=4, n_units=10)

    # 1. ClusterMeanBarPlot
    mean_plot: Any = plot_registry.get("cluster_mean_bar_plot")
    res_mean = mean_plot.generate(df, "group", "metric", hierarchy=config)
    assert res_mean.plot_type == "cluster_mean_bar_plot"
    assert res_mean.image_base64 != ""

    # 2. ClusterSpatialHeatmap
    spatial_plot: Any = plot_registry.get("cluster_spatial_heatmap")
    res_spatial = spatial_plot.generate(df, "group", "metric", hierarchy=config)
    assert res_spatial.plot_type == "cluster_spatial_heatmap"
    assert res_spatial.image_base64 != ""

    # 3. ProportionBarPlot
    df_bin = hierarchical_df(n_groups=2, n_clusters=4, n_units=10, is_binary=True)
    prop_plot: Any = plot_registry.get("proportion_bar_plot")
    res_prop = prop_plot.generate(df_bin, "group", "metric", hierarchy=config)
    assert res_prop.plot_type == "proportion_bar_plot"
    assert res_prop.image_base64 != ""


def test_unsupported_hierarchical_columns(hierarchical_df: Any) -> None:
    """Test compute_hierarchical_properties and build_cluster_aggregates with unsupported columns."""
    config = HierarchyConfig(group_col="group", cluster_col="cluster", unit_col="unit")
    df = hierarchical_df(n_groups=2, n_clusters=4, n_units=10)
    df["unsupported_metric"] = "string_value"

    # 1. build_cluster_aggregates with unsupported metric_kind
    agg = build_cluster_aggregates(df, config, [], "unsupported_metric", "unsupported")
    assert agg.empty
    assert list(agg.columns) == ["group", "cluster"]

    # 2. compute_hierarchical_properties with unsupported column
    props = compute_hierarchical_properties(df, config, [], "unsupported_metric")
    assert props.has_hierarchy
    assert props.metric_kind == "unsupported"
    import numpy as np

    assert np.isnan(props.icc)
