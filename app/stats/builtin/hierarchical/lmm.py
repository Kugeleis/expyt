from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.datasets.hierarchical import HierarchicalData
from app.stats.base import DataProperties, StatMethod, StatResult, stat_registry


@stat_registry.register("linear_mixed_model")
class LinearMixedModel(StatMethod):
    """Linear Mixed Model with random intercept per cluster."""

    @property
    def name(self) -> str:
        return "linear_mixed_model"

    @property
    def description(self) -> str:
        return "Linear Mixed-Effects Model (LMM) with a random intercept per cluster."

    def is_applicable(self, properties: DataProperties) -> bool:
        return (
            bool(properties.has_hierarchy)
            and properties.metric_kind == "continuous"
            and properties.min_clusters_per_group is not None
            and properties.min_clusters_per_group >= 4
        )

    def run(self, groups: Any) -> StatResult:
        if not isinstance(groups, HierarchicalData):
            raise ValueError("LinearMixedModel requires HierarchicalData.")

        import statsmodels.formula.api as smf

        excluded_ids = groups.excluded_clusters
        cluster_c = groups.config.cluster_col
        clean_df = groups.unit_df[~groups.unit_df[cluster_c].astype(str).isin(excluded_ids)].copy()
        clean_df = clean_df.dropna(subset=[groups.config.group_col, groups.config.cluster_col, groups.metric])

        grouped = clean_df.groupby(groups.config.group_col)
        n_clusters_used = {str(name): int(g_df[groups.config.cluster_col].nunique()) for name, g_df in grouped}

        try:
            model = smf.mixedlm(
                formula=f"{groups.metric} ~ C({groups.config.group_col})",
                data=clean_df,
                groups=clean_df[groups.config.cluster_col],
            )
            result = model.fit()

            term_names = [name for name in result.params.index if name.startswith(f"C({groups.config.group_col})")]
            if term_names:
                constraint = ", ".join(f"{name} = 0" for name in term_names)
                wald_res = result.wald_test(constraint, scalar=True)
                p_value = float(np.squeeze(wald_res.pvalue))
                test_statistic = float(np.squeeze(wald_res.statistic))
            else:
                p_value = 1.0
                test_statistic = 0.0

            var_w = float(result.scale)
            var_b = float(result.cov_re.iloc[0, 0])
            icc = float(var_b / (var_b + var_w)) if (var_b + var_w) > 0 else 0.0

            fe_df = pd.DataFrame(
                {
                    "term": list(result.params.index),
                    "estimate": [float(v) for v in result.params.values],
                    "std_err": [float(v) for v in result.bse.values],
                    "z_stat": [float(v) for v in result.tvalues.values],
                    "p_value": [float(v) for v in result.pvalues.values],
                }
            )

            summary = (
                f"Linear Mixed Model: {groups.metric} ~ "
                f"C({groups.config.group_col}) + (1 | {groups.config.cluster_col}).\n"
                f"Wald Chi2 test for group effect: X2 = {test_statistic:.4f}, p = {p_value:.4f}.\n"
                f"ICC = {icc:.4f} (between-cluster variance = {var_b:.4f}, residual variance = {var_w:.4f})."
            )

            return StatResult(
                column_name=groups.metric,
                method_name=self.name,
                test_statistic=test_statistic,
                p_value=p_value,
                effect_size=None,
                summary=summary,
                icc=icc,
                n_clusters_used=n_clusters_used,
                posthoc=fe_df,
            )
        except Exception as e:
            return StatResult(
                column_name=groups.metric,
                method_name=self.name,
                test_statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                summary=f"Linear Mixed Model fit failed: {e}",
                icc=0.0,
                power=0.0,
                n_clusters_used=n_clusters_used,
                posthoc=None,
                flags=["LMM_FIT_FAILED"],
            )
