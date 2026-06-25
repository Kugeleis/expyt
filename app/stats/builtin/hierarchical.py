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

        # ANOVA test
        f_stat, p_val = stats.f_oneway(*group_lists)

        # Cohen's f
        all_vals = []
        for g_list in group_lists:
            all_vals.extend(g_list)
        all_vals_arr = np.array(all_vals)
        grand_mean = all_vals_arr.mean()
        ss_total = np.sum((all_vals_arr - grand_mean) ** 2)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_lists)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
        cohen_f = np.sqrt(eta_squared / (1.0 - eta_squared)) if eta_squared < 1.0 else 0.0

        # Power
        from statsmodels.stats.power import FTestAnovaPower

        power_analysis = FTestAnovaPower()
        k_groups = len(group_lists)
        nobs = len(all_vals) / k_groups if k_groups > 0 else 0
        power = 0.0
        if k_groups > 1 and nobs > 1:
            with contextlib.suppress(Exception):
                power = float(power_analysis.solve_power(effect_size=cohen_f, nobs=nobs, k_groups=k_groups, alpha=0.05))

        # Post-hoc Tukey HSD
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
                # Holm correction on Tukey p-values
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

        # Kruskal test
        h_stat, p_val = stats.kruskal(*group_lists)

        # Epsilon-squared
        n = sum(len(g) for g in group_lists)
        effect_size = float(h_stat) / (n - 1) if n > 1 else 0.0

        # Post-hoc Dunn's test with Holm correction
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

        # Tie correction
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

        # Filter unit dataframe to drop excluded clusters
        excluded_ids = groups.excluded_clusters
        cluster_c = groups.config.cluster_col
        clean_df = groups.unit_df[~groups.unit_df[cluster_c].astype(str).isin(excluded_ids)].copy()
        clean_df = clean_df.dropna(subset=[groups.config.group_col, groups.config.cluster_col, groups.metric])

        # Group stats
        grouped = clean_df.groupby(groups.config.group_col)
        n_clusters_used = {str(name): int(g_df[groups.config.cluster_col].nunique()) for name, g_df in grouped}

        try:
            model = smf.mixedlm(
                formula=f"{groups.metric} ~ C({groups.config.group_col})",
                data=clean_df,
                groups=clean_df[groups.config.cluster_col],
            )
            result = model.fit()

            # Wald test for fixed effect of C(group_col)
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

            # Build fixed effects info dataframe
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

        # Kruskal test
        h_stat, p_val = stats.kruskal(*group_lists)

        n = sum(len(g) for g in group_lists)
        effect_size = float(h_stat) / (n - 1) if n > 1 else 0.0

        # Post-hoc pairwise Mann-Whitney with Holm correction
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

        # Runs iterative Grubbs test on cluster means
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
                from typing import cast

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

        # Levene's test
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
