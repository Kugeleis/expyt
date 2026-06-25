from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.plots.base import PlotGenerator, PlotResult, plot_registry
from app.stats.base import DataProperties
from app.stats.properties import build_cluster_aggregates

matplotlib.use("Agg")


@plot_registry.register("cluster_mean_bar_plot")
class ClusterMeanBarPlot(PlotGenerator):
    """Bar plot of group means of cluster-level means."""

    @property
    def name(self) -> str:
        return "cluster_mean_bar_plot"

    @property
    def description(self) -> str:
        return "Bar plot of group means of cluster means ± 95% CI with overlaid cluster means."

    def is_applicable(self, properties: DataProperties) -> bool:
        return bool(properties.has_hierarchy)

    def generate(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: str,
        **kwargs: Any,
    ) -> PlotResult:
        hierarchy = kwargs.get("hierarchy")
        excluded_clusters = kwargs.get("excluded_clusters", [])
        if not hierarchy:
            raise ValueError("HierarchyConfig is required for ClusterMeanBarPlot.")

        # Build cluster aggregates
        agg_df = build_cluster_aggregates(df, hierarchy, excluded_clusters, value_col, "continuous")

        groups = sorted(str(name) for name in agg_df[hierarchy.group_col].dropna().unique())

        fig, ax = plt.subplots()
        try:
            for i, group_name in enumerate(groups):
                group_means = agg_df[agg_df[hierarchy.group_col] == group_name]["mean"].dropna().values
                n = len(group_means)
                if n > 0:
                    mean_val = np.mean(group_means)
                    std_val = np.std(group_means, ddof=1) if n > 1 else 0.0
                    se = std_val / np.sqrt(n)
                    ci = 1.96 * se

                    # Plot bar
                    ax.bar(i, mean_val, yerr=ci, capsize=5, alpha=0.6, color="skyblue", edgecolor="blue")

                    # Overlay strip plot points
                    jitter = np.random.normal(i, 0.04, size=len(group_means))
                    ax.scatter(jitter, group_means, color="darkblue", alpha=0.8, edgecolor="none", zorder=3)

            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(groups)
            ax.set_title(f"Cluster Means of {value_col} by {group_col}")
            ax.set_xlabel(group_col)
            ax.set_ylabel(f"Cluster Mean of {value_col}")

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        finally:
            plt.close(fig)

        return PlotResult(
            column_name=value_col, plot_type=self.name, image_base64=image_base64, content_type="image/png"
        )


@plot_registry.register("cluster_spatial_heatmap")
class ClusterSpatialHeatmap(PlotGenerator):
    """Grid heatmap of unit values by x/y coordinates, faceted by group."""

    @property
    def name(self) -> str:
        return "cluster_spatial_heatmap"

    @property
    def description(self) -> str:
        return "Spatial grid heatmap of unit values by X/Y coordinates faceted by group."

    def is_applicable(self, properties: DataProperties) -> bool:
        return bool(properties.has_hierarchy) and bool(properties.has_spatial_coords)

    def generate(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: str,
        **kwargs: Any,
    ) -> PlotResult:
        hierarchy = kwargs.get("hierarchy")
        excluded_clusters = kwargs.get("excluded_clusters", [])
        if not hierarchy or not hierarchy.x_col or not hierarchy.y_col:
            raise ValueError("HierarchyConfig with spatial columns is required.")

        clean_df = df[~df[hierarchy.cluster_col].astype(str).isin(excluded_clusters)].dropna(
            subset=[hierarchy.group_col, hierarchy.x_col, hierarchy.y_col, value_col]
        )

        groups = sorted(str(name) for name in clean_df[hierarchy.group_col].unique())
        if not groups:
            fig, ax = plt.subplots()
            plt.close(fig)
            return PlotResult(column_name=value_col, plot_type=self.name, image_base64="", content_type="image/png")

        fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), squeeze=False)
        try:
            vmin = float(clean_df[value_col].min())
            vmax = float(clean_df[value_col].max())

            sc = None
            for i, group_name in enumerate(groups):
                ax = axes[0, i]
                group_df = clean_df[clean_df[hierarchy.group_col] == group_name]
                x = group_df[hierarchy.x_col]
                y = group_df[hierarchy.y_col]
                z = group_df[value_col]

                sc = ax.scatter(x, y, c=z, cmap="viridis", vmin=vmin, vmax=vmax, s=80, marker="s", edgecolor="none")
                ax.set_title(f"Group: {group_name}")
                ax.set_xlabel(hierarchy.x_col)
                if i == 0:
                    ax.set_ylabel(hierarchy.y_col)

            if sc is not None:
                fig.subplots_adjust(right=0.85)
                cbar_ax = fig.add_axes((0.88, 0.15, 0.03, 0.7))
                fig.colorbar(sc, cax=cbar_ax, label=value_col)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        finally:
            plt.close(fig)

        return PlotResult(
            column_name=value_col, plot_type=self.name, image_base64=image_base64, content_type="image/png"
        )


@plot_registry.register("proportion_bar_plot")
class ProportionBarPlot(PlotGenerator):
    """Bar plot of corrected cluster-level proportions."""

    @property
    def name(self) -> str:
        return "proportion_bar_plot"

    @property
    def description(self) -> str:
        return "Yield percentage bar plot per group with 95% CI bars."

    def is_applicable(self, properties: DataProperties) -> bool:
        return bool(properties.has_hierarchy) and properties.metric_kind == "binary_proportion"

    def generate(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: str,
        **kwargs: Any,
    ) -> PlotResult:
        hierarchy = kwargs.get("hierarchy")
        excluded_clusters = kwargs.get("excluded_clusters", [])
        if not hierarchy:
            raise ValueError("HierarchyConfig is required for ProportionBarPlot.")

        # Build cluster aggregates
        agg_df = build_cluster_aggregates(df, hierarchy, excluded_clusters, value_col, "binary_proportion")

        groups = sorted(str(name) for name in agg_df[hierarchy.group_col].dropna().unique())

        fig, ax = plt.subplots()
        try:
            for i, group_name in enumerate(groups):
                group_props = agg_df[agg_df[hierarchy.group_col] == group_name]["proportion_corrected"].dropna().values
                n = len(group_props)
                if n > 0:
                    mean_val = np.mean(group_props)
                    std_val = np.std(group_props, ddof=1) if n > 1 else 0.0
                    se = std_val / np.sqrt(n)
                    ci = 1.96 * se

                    mean_pct = mean_val * 100.0
                    ci_pct = ci * 100.0

                    # Plot bar
                    ax.bar(i, mean_pct, yerr=ci_pct, capsize=5, alpha=0.6, color="lightgreen", edgecolor="green")

                    # Overlay strip plot points
                    jitter = np.random.normal(i, 0.04, size=len(group_props))
                    ax.scatter(jitter, group_props * 100.0, color="darkgreen", alpha=0.8, edgecolor="none", zorder=3)

            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(groups)
            ax.set_title(f"Proportion (Yield %) of {value_col} by {group_col}")
            ax.set_xlabel(group_col)
            ax.set_ylabel("Yield % (mean of corrected cluster proportions)")
            ax.set_ylim(0, 105)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        finally:
            plt.close(fig)

        return PlotResult(
            column_name=value_col, plot_type=self.name, image_base64=image_base64, content_type="image/png"
        )
