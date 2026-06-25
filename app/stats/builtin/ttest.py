"""Independent two-sample t-test plugin."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from app.stats.base import DataProperties, StatMethod, StatResult, stat_registry


@stat_registry.register("ttest_ind")
class TTestInd(StatMethod):
    """Independent two-sample t-test (parametric test for two groups)."""

    @property
    def name(self) -> str:
        """Return the unique name of the statistical method."""
        return "ttest_ind"

    @property
    def description(self) -> str:
        """Return a brief description of the statistical method."""
        return "Independent two-sample t-test (parametric)."

    def is_applicable(self, properties: DataProperties) -> bool:
        """Determine whether the t-test is applicable.

        Requires exactly 2 groups, each group with >= 2 samples, and all groups
        satisfying Shapiro-Wilk normality test (p > 0.05).
        """
        if properties.n_groups != 2:
            return False

        # Each group must have at least 2 samples
        if any(size < 2 for size in properties.group_sizes.values()):
            return False

        # All groups must be approximately normal
        return all(res.is_normal for res in properties.normality.values())

    def run(self, groups: dict[str, list[float]]) -> StatResult:
        """Run the independent t-test.

        Args:
            groups: Dictionary with exactly two groups.

        Returns:
            A StatResult containing the test statistic, p-value, and Cohen's d.
        """
        if len(groups) != 2:
            msg = f"ttest_ind requires exactly 2 groups, got {len(groups)}"
            raise ValueError(msg)

        group_names = sorted(groups.keys())
        g1_name, g2_name = group_names[0], group_names[1]
        g1 = groups[g1_name]
        g2 = groups[g2_name]

        if len(g1) < 2 or len(g2) < 2:
            msg = "Each group must have at least 2 samples to compute t-test."
            raise ValueError(msg)

        # Check variance homogeneity via Levene
        try:
            _, levene_p = stats.levene(g1, g2)
            equal_var = levene_p > 0.05
        except Exception:
            equal_var = False

        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=equal_var)

        # Calculate Cohen's d effect size
        n1, n2 = len(g1), len(g2)
        v1 = float(np.var(g1, ddof=1))
        v2 = float(np.var(g2, ddof=1))
        m1 = float(np.mean(g1))
        m2 = float(np.mean(g2))

        pooled_se = float(np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)))
        effect_size = (m1 - m2) / pooled_se if pooled_se > 0 else 0.0

        var_type = "equal variance assumed" if equal_var else "Welch's t-test (unequal variance)"
        summary = (
            f"Independent two-sample t-test ({var_type}) between "
            f"{g1_name!r} (mean={m1:.4f}, N={n1}) and "
            f"{g2_name!r} (mean={m2:.4f}, N={n2}): "
            f"t = {t_stat:.4f}, p = {p_val:.4f}, Cohen's d = {effect_size:.4f}."
        )

        return StatResult(
            column_name=None,
            method_name=self.name,
            test_statistic=float(t_stat),
            p_value=float(p_val),
            effect_size=effect_size,
            summary=summary,
        )
