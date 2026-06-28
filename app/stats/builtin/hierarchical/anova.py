from __future__ import annotations

import contextlib
import itertools
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

from app.datasets.hierarchical import HierarchicalData
from app.stats.base import DataProperties, StatMethod, StatResult, stat_registry


@stat_registry.register("cluster_mean_anova")
class ClusterMeanANOVA(StatMethod):
    """One-way ANOVA on cluster means."""

    @property
    def name(self) -> str:
        return "cluster_mean_anova"

    @property
    def description(self) -> str:
        return "One-way ANOVA on cluster-level means (two-stage)."

    def is_applicable(self, properties: DataProperties) -> bool:
        return (
            bool(properties.has_hierarchy)
            and bool(properties.normality_cluster_means)
            and properties.min_clusters_per_group is not None
            and properties.min_clusters_per_group >= 3
        )

    def run(self, groups: Any) -> StatResult:
        if not isinstance(groups, HierarchicalData):
            raise ValueError("ClusterMeanANOVA requires HierarchicalData.")

        grouped_means = groups.cluster_agg.groupby(groups.config.group_col)["mean"]
        group_names = sorted(str(name) for name in grouped_means.groups)
        group_data = {str(name): [float(v) for v in group.dropna().values] for name, group in grouped_means}
        group_lists = [group_data[name] for name in group_names]

        f_stat, p_val = stats.f_oneway(*group_lists)

        all_vals = []
        for g_list in group_lists:
            all_vals.extend(g_list)
        all_vals_arr = np.array(all_vals)
        grand_mean = all_vals_arr.mean()
        ss_total = np.sum((all_vals_arr - grand_mean) ** 2)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_lists)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
        cohen_f = np.sqrt(eta_squared / (1.0 - eta_squared)) if eta_squared < 1.0 else 0.0

        from statsmodels.stats.power import FTestAnovaPower

        power_analysis = FTestAnovaPower()
        k_groups = len(group_lists)
        nobs = len(all_vals) / k_groups if k_groups > 0 else 0
        power = 0.0
        if k_groups > 1 and nobs > 1:
            with contextlib.suppress(Exception):
                power = float(power_analysis.solve_power(effect_size=cohen_f, nobs=nobs, k_groups=k_groups, alpha=0.05))

        records = []
        if k_groups >= 2 and all(len(g) >= 2 for g in group_lists):
            try:
                tukey_res = stats.tukey_hsd(*group_lists)
                for i, j in itertools.combinations(range(k_groups), 2):
                    g1, g2 = group_names[i], group_names[j]
                    diff = np.mean(group_data[str(g1)]) - np.mean(group_data[str(g2)])
                    p_val_tukey = tukey_res.pvalue[i, j]
                    records.append(
                        {"group1": str(g1), "group2": str(g2), "mean_diff": float(diff), "p_value": float(p_val_tukey)}
                    )
                if records:
                    p_vals = [r["p_value"] for r in records]
                    _, corrected_pvals, _, _ = multipletests(p_vals, alpha=0.05, method="holm")
                    for r, cp in zip(records, corrected_pvals, strict=False):
                        r["p_value_corrected"] = float(cp)
            except Exception:
                pass

        posthoc_df = pd.DataFrame(records) if records else None

        names_str = ", ".join(repr(n) for n in group_names)
        summary = (
            f"One-way ANOVA on cluster means across {k_groups} groups ({names_str}): "
            f"F = {f_stat:.4f}, p = {p_val:.4f}, Cohen's f = {cohen_f:.4f}, power = {power:.4f}."
        )

        n_clusters_used = {name: len(group_data[name]) for name in group_names}

        return StatResult(
            column_name=groups.metric,
            method_name=self.name,
            test_statistic=float(f_stat),
            p_value=float(p_val),
            effect_size=float(cohen_f),
            summary=summary,
            icc=groups.icc,
            power=power,
            n_clusters_used=n_clusters_used,
            posthoc=posthoc_df,
        )
