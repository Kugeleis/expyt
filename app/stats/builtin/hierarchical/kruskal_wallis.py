from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

from app.datasets.hierarchical import HierarchicalData
from app.stats.base import DataProperties, StatMethod, StatResult, stat_registry


@stat_registry.register("cluster_mean_kruskal_wallis")
class ClusterMeanKruskalWallis(StatMethod):
    """Kruskal-Wallis test on cluster means."""

    @property
    def name(self) -> str:
        return "cluster_mean_kruskal_wallis"

    @property
    def description(self) -> str:
        return "Kruskal-Wallis H test on cluster-level means (two-stage)."

    def is_applicable(self, properties: DataProperties) -> bool:
        if not properties.has_hierarchy:
            return False
        if properties.min_clusters_per_group is None or properties.min_clusters_per_group < 2:
            return False
        return (not properties.normality_cluster_means) or (properties.min_clusters_per_group >= 3)

    def run(self, groups: Any) -> StatResult:
        if not isinstance(groups, HierarchicalData):
            raise ValueError("ClusterMeanKruskalWallis requires HierarchicalData.")

        grouped_means = groups.cluster_agg.groupby(groups.config.group_col)["mean"]
        group_names = sorted(str(name) for name in grouped_means.groups)
        group_data = {str(name): [float(v) for v in group.dropna().values] for name, group in grouped_means}
        group_lists = [group_data[name] for name in group_names]

        h_stat, p_val = stats.kruskal(*group_lists)

        n = sum(len(g) for g in group_lists)
        effect_size = float(h_stat) / (n - 1) if n > 1 else 0.0

        records = []
        all_vals = []
        for g_list in group_lists:
            all_vals.extend(g_list)
        all_ranks = stats.rankdata(all_vals)
        n_total = len(all_vals)

        idx = 0
        group_ranks = {}
        for name in group_names:
            n_g = len(group_data[name])
            group_ranks[name] = all_ranks[idx : idx + n_g]
            idx += n_g

        mean_ranks = {name: float(np.mean(group_ranks[name])) for name in group_names}

        from collections import Counter

        counts = Counter(all_ranks)
        tie_sum = sum(c**3 - c for c in counts.values())

        for g1, g2 in itertools.combinations(group_names, 2):
            n1 = len(group_data[g1])
            n2 = len(group_data[g2])
            diff = mean_ranks[g1] - mean_ranks[g2]

            base_se = n_total * (n_total + 1) / 12.0
            if n_total > 1:
                tie_adj = tie_sum / (12.0 * (n_total - 1))
                se = np.sqrt((base_se - tie_adj) * (1.0 / n1 + 1.0 / n2))
            else:
                se = np.sqrt(base_se * (1.0 / n1 + 1.0 / n2))

            z_stat = diff / se if se > 0 else 0.0
            p_val_dunn = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))

            records.append({"group1": str(g1), "group2": str(g2), "stat": float(z_stat), "p_value": float(p_val_dunn)})

        if records:
            p_vals = [r["p_value"] for r in records]
            _, corrected_pvals, _, _ = multipletests(p_vals, alpha=0.05, method="holm")
            for r, cp in zip(records, corrected_pvals, strict=False):
                r["p_value_corrected"] = float(cp)

        posthoc_df = pd.DataFrame(records) if records else None

        names_str = ", ".join(repr(n) for n in group_names)
        summary = (
            f"Kruskal-Wallis H test on cluster means across {len(group_names)} groups ({names_str}): "
            f"H = {h_stat:.4f}, p = {p_val:.4f}, epsilon_squared = {effect_size:.4f}."
        )

        n_clusters_used = {name: len(group_data[name]) for name in group_names}

        return StatResult(
            column_name=groups.metric,
            method_name=self.name,
            test_statistic=float(h_stat),
            p_value=float(p_val),
            effect_size=effect_size,
            summary=summary,
            n_clusters_used=n_clusters_used,
            posthoc=posthoc_df,
        )


@stat_registry.register("proportion_kruskal_wallis")
class ProportionKruskalWallis(StatMethod):
    """Kruskal-Wallis test on cluster proportions."""

    @property
    def name(self) -> str:
        return "proportion_kruskal_wallis"

    @property
    def description(self) -> str:
        return "Kruskal-Wallis H test on cluster-level corrected proportions."

    def is_applicable(self, properties: DataProperties) -> bool:
        return bool(properties.has_hierarchy) and properties.metric_kind == "binary_proportion"

    def run(self, groups: Any) -> StatResult:
        if not isinstance(groups, HierarchicalData):
            raise ValueError("ProportionKruskalWallis requires HierarchicalData.")

        grouped_props = groups.cluster_agg.groupby(groups.config.group_col)["proportion_corrected"]
        group_names = sorted(str(name) for name in grouped_props.groups)
        group_data = {str(name): [float(v) for v in group.dropna().values] for name, group in grouped_props}
        group_lists = [group_data[name] for name in group_names]

        h_stat, p_val = stats.kruskal(*group_lists)

        n = sum(len(g) for g in group_lists)
        effect_size = float(h_stat) / (n - 1) if n > 1 else 0.0

        records = []
        for g1, g2 in itertools.combinations(group_names, 2):
            vals1 = np.array(group_data[g1])
            vals2 = np.array(group_data[g2])
            mean1, mean2 = np.mean(vals1), np.mean(vals2)
            var1 = np.var(vals1, ddof=1) if len(vals1) > 1 else 0.0
            var2 = np.var(vals2, ddof=1) if len(vals2) > 1 else 0.0
            n1, n2 = len(vals1), len(vals2)

            se = np.sqrt(var1 / n1 + var2 / n2) if (n1 > 0 and n2 > 0) else 0.0
            diff = mean1 - mean2
            ci_lower = diff - 1.96 * se
            ci_upper = diff + 1.96 * se

            try:
                stat, p_val_mw = stats.mannwhitneyu(vals1, vals2, alternative="two-sided")
            except Exception:
                stat, p_val_mw = 0.0, 1.0

            records.append(
                {
                    "group1": str(g1),
                    "group2": str(g2),
                    "stat": float(stat),
                    "p_value": float(p_val_mw),
                    "yield_delta": float(diff),
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                }
            )

        if records:
            p_vals = [r["p_value"] for r in records]
            _, corrected_pvals, _, _ = multipletests(p_vals, alpha=0.05, method="holm")
            for r, cp in zip(records, corrected_pvals, strict=False):
                r["p_value_corrected"] = float(cp)

        posthoc_df = pd.DataFrame(records) if records else None

        names_str = ", ".join(repr(n) for n in group_names)
        summary = (
            f"Kruskal-Wallis H test on corrected cluster proportions across {len(group_names)} groups ({names_str}): "
            f"H = {h_stat:.4f}, p = {p_val:.4f}, epsilon_squared = {effect_size:.4f}."
        )

        n_clusters_used = {name: len(group_data[name]) for name in group_names}

        return StatResult(
            column_name=groups.metric,
            method_name=self.name,
            test_statistic=float(h_stat),
            p_value=float(p_val),
            effect_size=effect_size,
            summary=summary,
            n_clusters_used=n_clusters_used,
            posthoc=posthoc_df,
        )
