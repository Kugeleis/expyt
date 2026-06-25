"""One-way ANOVA test plugin."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from app.stats.base import DataProperties, StatMethod, StatResult, stat_registry


@stat_registry.register("anova")
class Anova(StatMethod):
    """One-way Analysis of Variance (ANOVA)."""

    @property
    def name(self) -> str:
        """Return the unique name of the statistical method."""
        return "anova"

    @property
    def description(self) -> str:
        """Return a brief description of the statistical method."""
        return "One-way ANOVA (parametric)."

    def is_applicable(self, properties: DataProperties) -> bool:
        """Determine whether One-way ANOVA is applicable.

        Requires >= 2 groups, each group with >= 2 samples, all groups
        satisfying normality (p > 0.05), and variance homogeneity (Levene p > 0.05).
        """
        if properties.n_groups < 2:
            return False

        if any(size < 2 for size in properties.group_sizes.values()):
            return False

        # All groups must be normal
        if any(not res.is_normal for res in properties.normality.values()):
            return False

        # Homogeneous variance
        if properties.variance_homogeneity is None:
            return False
        return properties.variance_homogeneity.equal_variances

    def run(self, groups: dict[str, list[float]]) -> StatResult:
        """Run the One-way ANOVA test.

        Args:
            groups: Dictionary with group lists.

        Returns:
            A StatResult containing the F statistic, p-value, and eta-squared.
        """
        if len(groups) < 2:
            msg = f"anova requires at least 2 groups, got {len(groups)}"
            raise ValueError(msg)

        group_names = sorted(groups.keys())
        group_lists = [groups[name] for name in group_names]

        if any(len(g) < 2 for g in group_lists):
            msg = "Each group must have at least 2 samples to compute ANOVA."
            raise ValueError(msg)

        f_stat, p_val = stats.f_oneway(*group_lists)

        # Calculate Eta-squared: SS_between / SS_total
        all_vals = []
        for g in group_lists:
            all_vals.extend(g)

        overall_mean = float(np.mean(all_vals))
        ss_total = float(np.sum((np.array(all_vals) - overall_mean) ** 2))

        ss_between = 0.0
        for g in group_lists:
            ss_between += len(g) * (float(np.mean(g)) - overall_mean) ** 2

        eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

        names_str = ", ".join(repr(n) for n in group_names)
        summary = (
            f"One-way ANOVA across {len(groups)} groups ({names_str}): "
            f"F = {f_stat:.4f}, p = {p_val:.4f}, eta_squared = {eta_squared:.4f}."
        )

        return StatResult(
            column_name=None,
            method_name=self.name,
            test_statistic=float(f_stat),
            p_value=float(p_val),
            effect_size=eta_squared,
            summary=summary,
        )
