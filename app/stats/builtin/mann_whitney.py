"""Mann-Whitney U test plugin."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats

from app.stats.base import DataProperties, StatMethod, StatResult, stat_registry


@stat_registry.register("mann_whitney")
class MannWhitney(StatMethod):
    """Mann-Whitney U test (non-parametric alternative to independent t-test)."""

    @property
    def name(self) -> str:
        """Return the unique name of the statistical method."""
        return "mann_whitney"

    @property
    def description(self) -> str:
        """Return a brief description of the statistical method."""
        return "Mann-Whitney U test (non-parametric)."

    def is_applicable(self, properties: DataProperties) -> bool:
        """Determine whether the Mann-Whitney U test is applicable.

        Requires exactly 2 groups, each group with >= 2 samples.
        """
        if properties.n_groups != 2:
            return False

        return all(size >= 2 for size in properties.group_sizes.values())

    def run(self, groups: dict[str, list[float]]) -> StatResult:
        """Run the Mann-Whitney U test.

        Args:
            groups: Dictionary with exactly two groups.

        Returns:
            A StatResult containing the U statistic, p-value, and
            rank-biserial correlation.
        """
        if len(groups) != 2:
            msg = f"mann_whitney requires exactly 2 groups, got {len(groups)}"
            raise ValueError(msg)

        group_names = sorted(groups.keys())
        g1_name, g2_name = group_names[0], group_names[1]
        g1 = groups[g1_name]
        g2 = groups[g2_name]

        if len(g1) < 2 or len(g2) < 2:
            msg = "Each group must have at least 2 samples to compute Mann-Whitney U."
            raise ValueError(msg)

        u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")

        # Calculate Rank-Biserial Correlation effect size: r = 1 - (2 * U) / (n1 * n2)
        n1, n2 = len(g1), len(g2)
        effect_size = 1.0 - (2.0 * float(u_stat)) / (n1 * n2)

        med1 = float(np.median(g1))
        med2 = float(np.median(g2))
        summary = (
            f"Mann-Whitney U test between {g1_name!r} (median={med1:.4f}, N={n1}) "
            f"and {g2_name!r} (median={med2:.4f}, N={n2}): "
            f"U = {u_stat:.4f}, p = {p_val:.4f}, rank-biserial r = {effect_size:.4f}."
        )

        return StatResult(
            column_name=None,
            method_name=self.name,
            test_statistic=float(u_stat),
            p_value=float(p_val),
            effect_size=effect_size,
            summary=summary,
        )
