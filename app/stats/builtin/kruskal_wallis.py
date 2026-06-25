"""Kruskal-Wallis H test plugin."""

from __future__ import annotations

import scipy.stats as stats

from app.stats.base import DataProperties, StatMethod, StatResult, stat_registry


@stat_registry.register("kruskal_wallis")
class KruskalWallis(StatMethod):
    """Kruskal-Wallis H test (non-parametric alternative to one-way ANOVA)."""

    @property
    def name(self) -> str:
        """Return the unique name of the statistical method."""
        return "kruskal_wallis"

    @property
    def description(self) -> str:
        """Return a brief description of the statistical method."""
        return "Kruskal-Wallis H test (non-parametric)."

    def is_applicable(self, properties: DataProperties) -> bool:
        """Determine whether Kruskal-Wallis test is applicable.

        Requires >= 2 groups, each group with >= 2 samples.
        """
        if properties.n_groups < 2:
            return False

        return all(size >= 2 for size in properties.group_sizes.values())

    def run(self, groups: dict[str, list[float]]) -> StatResult:
        """Run the Kruskal-Wallis H test.

        Args:
            groups: Dictionary with group lists.

        Returns:
            A StatResult containing the H statistic, p-value, and epsilon-squared.
        """
        if len(groups) < 2:
            msg = f"kruskal_wallis requires at least 2 groups, got {len(groups)}"
            raise ValueError(msg)

        group_names = sorted(groups.keys())
        group_lists = [groups[name] for name in group_names]

        if any(len(g) < 2 for g in group_lists):
            msg = "Each group must have at least 2 samples to compute Kruskal-Wallis."
            raise ValueError(msg)

        h_stat, p_val = stats.kruskal(*group_lists)

        # Calculate Epsilon-squared: H / (N - 1)
        n = sum(len(g) for g in group_lists)
        effect_size = float(h_stat) / (n - 1) if n > 1 else 0.0

        names_str = ", ".join(repr(n) for n in group_names)
        summary = (
            f"Kruskal-Wallis H test across {len(groups)} groups ({names_str}): "
            f"H = {h_stat:.4f}, p = {p_val:.4f}, epsilon_squared = {effect_size:.4f}."
        )

        return StatResult(
            column_name=None,
            method_name=self.name,
            test_statistic=float(h_stat),
            p_value=float(p_val),
            effect_size=effect_size,
            summary=summary,
        )
