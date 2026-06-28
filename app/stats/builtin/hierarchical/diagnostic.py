from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import scipy.stats as stats

from app.datasets.hierarchical import HierarchicalData
from app.stats.base import DataProperties, StatMethod, StatResult, stat_registry


@stat_registry.register("grubbs_cluster_outlier")
class GrubbsClusterOutlier(StatMethod):
    """Diagnostic Grubbs outlier test on cluster means."""

    @property
    def name(self) -> str:
        return "grubbs_cluster_outlier"

    @property
    def description(self) -> str:
        return "Diagnostic Grubbs outlier test on cluster-level means."

    def is_applicable(self, properties: DataProperties) -> bool:
        return (
            bool(properties.has_hierarchy)
            and properties.min_clusters_per_group is not None
            and properties.min_clusters_per_group >= 4
        )

    def run(self, groups: Any) -> StatResult:
        if not isinstance(groups, HierarchicalData):
            raise ValueError("GrubbsClusterOutlier requires HierarchicalData.")

        temp_df = groups.cluster_agg[[groups.config.cluster_col, "mean"]].dropna().copy()
        flagged_clusters = []
        records = []
        alpha = 0.05

        while len(temp_df) >= 3:
            n = len(temp_df)
            mean_val = temp_df["mean"].mean()
            std_val = temp_df["mean"].std()
            if std_val == 0 or np.isnan(std_val):
                break

            temp_df["dev"] = (temp_df["mean"] - mean_val).abs()
            max_idx = temp_df["dev"].idxmax()
            max_row = temp_df.loc[max_idx]
            g_val = max_row["dev"] / std_val

            t_dist = stats.t.ppf(1 - alpha / (2 * n), n - 2)
            g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_dist**2 / (n - 2 + t_dist**2))

            if g_val > g_crit:
                c_id = str(max_row[groups.config.cluster_col])
                flagged_clusters.append(c_id)

                records.append(
                    {
                        "cluster_id": c_id,
                        "mean": float(cast(float, max_row["mean"])),
                        "g_statistic": float(cast(float, g_val)),
                        "g_critical": float(g_crit),
                        "z_score": float(cast(float, (max_row["mean"] - mean_val) / std_val)),
                    }
                )
                temp_df = temp_df.drop(max_idx)
            else:
                break

        p_val = 0.0 if flagged_clusters else 1.0
        test_stat = float(len(flagged_clusters))

        flags = [f"OUTLIER_CLUSTER:{c_id}" for c_id in flagged_clusters]

        summary = f"Iterative Grubbs outlier detection on cluster means flagged {len(flagged_clusters)} outlier(s)."
        if flagged_clusters:
            summary += f" Flagged cluster IDs: {', '.join(flagged_clusters)}."

        posthoc_df = pd.DataFrame(records) if records else None

        return StatResult(
            column_name=groups.metric,
            method_name=self.name,
            test_statistic=test_stat,
            p_value=p_val,
            effect_size=None,
            summary=summary,
            flags=flags,
            posthoc=posthoc_df,
        )


@stat_registry.register("levene_cluster_uniformity")
class LeveneClusterUniformity(StatMethod):
    """Uniformity of within-cluster spread across groups."""

    @property
    def name(self) -> str:
        return "levene_cluster_uniformity"

    @property
    def description(self) -> str:
        return "Levene's test for homogeneity of within-cluster standard deviations across groups."

    def is_applicable(self, properties: DataProperties) -> bool:
        return (
            bool(properties.has_hierarchy)
            and properties.metric_kind == "continuous"
            and properties.min_clusters_per_group is not None
            and properties.min_clusters_per_group >= 3
        )

    def run(self, groups: Any) -> StatResult:
        if not isinstance(groups, HierarchicalData):
            raise ValueError("LeveneClusterUniformity requires HierarchicalData.")

        grouped_stds = groups.cluster_agg.groupby(groups.config.group_col)["std"]
        group_names = sorted(str(name) for name in grouped_stds.groups)
        group_data = {str(name): [float(v) for v in group.dropna().values] for name, group in grouped_stds}
        group_lists = [group_data[name] for name in group_names]

        w_stat, p_val = stats.levene(*group_lists, center="median")

        mean_spread = {name: float(np.mean(group_data[name])) for name in group_names}

        names_str = ", ".join(repr(n) for n in group_names)
        summary = (
            f"Levene test for uniformity of within-cluster spread across {len(group_names)} groups ({names_str}): "
            f"W = {w_stat:.4f}, p = {p_val:.4f}.\n"
            f"Per-group mean cluster standard deviation (spread): "
            + ", ".join(f"{k}: {v:.4f}" for k, v in mean_spread.items())
        )

        n_clusters_used = {name: len(group_data[name]) for name in group_names}

        return StatResult(
            column_name=groups.metric,
            method_name=self.name,
            test_statistic=float(w_stat),
            p_value=float(p_val),
            effect_size=None,
            summary=summary,
            n_clusters_used=n_clusters_used,
        )
