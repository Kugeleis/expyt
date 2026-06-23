"""Chi-Square test of independence plugin."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.stats as stats

from app.stats.base import DataProperties, StatMethod, StatResult, stat_registry


@stat_registry.register("chi_square")
class ChiSquare(StatMethod):
    """Chi-Square Test of Independence (non-parametric categorical test)."""

    @property
    def name(self) -> str:
        """Return the unique name of the statistical method."""
        return "chi_square"

    @property
    def description(self) -> str:
        """Return a brief description of the statistical method."""
        return "Chi-Square Test of Independence (categorical)."

    def is_applicable(self, **properties: Any) -> bool:
        """Determine whether Chi-Square is applicable.

        Requires outcome_type_guess to be categorical_nominal or categorical_ordinal_unclear.
        Requires n_groups >= 2.
        """
        data_properties = DataProperties(**properties)
        if data_properties.n_groups < 2:
            return False

        return data_properties.outcome_type_guess in ("categorical_nominal", "categorical_ordinal_unclear")

    def run(self, groups: dict[str, list[Any]]) -> StatResult:
        """Run the Chi-Square test of independence.

        Args:
            groups: Dictionary with group names to list of categorical outcomes.

        Returns:
            A StatResult containing the chi2 statistic, p-value, and Cramér's V.
        """
        # Reconstruct observation contingency table
        all_outcomes = set()
        for g_vals in groups.values():
            all_outcomes.update(g_vals)

        outcome_list = sorted(list(all_outcomes))
        if len(outcome_list) < 2:
            raise ValueError("Chi-Square test requires at least 2 unique categories in the outcome column.")

        # Build contingency table
        table = []
        group_names = sorted(groups.keys())
        for g_name in group_names:
            g_vals = groups[g_name]
            # Count occurrences of each outcome
            counts = [g_vals.count(out) for out in outcome_list]
            table.append(counts)

        observed = np.array(table)
        if observed.size == 0 or observed.sum() == 0:
            raise ValueError("Empty or invalid contingency table for Chi-Square test.")

        chi2_stat, p_val, dof, expected = stats.chi2_contingency(observed)

        # Compute Cramér's V as effect size
        n = observed.sum()
        r, c = observed.shape
        min_dim = min(r - 1, c - 1)
        cramers_v = np.sqrt(chi2_stat / (n * min_dim)) if n > 0 and min_dim > 0 else 0.0

        summary = (
            f"Chi-Square test of independence: "
            f"chi2 = {chi2_stat:.4f}, p = {p_val:.4f}, dof = {dof}, Cramér's V = {cramers_v:.4f}."
        )

        return StatResult(
            column_name=None,
            method_name=self.name,
            test_statistic=float(chi2_stat),
            p_value=float(p_val),
            effect_size=float(cramers_v),
            summary=summary,
        )
