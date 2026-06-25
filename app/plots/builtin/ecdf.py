"""ECDF plot generator plugin."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.plots.base import PlotGenerator, PlotResult, plot_registry

if TYPE_CHECKING:
    from app.stats.base import DataProperties

matplotlib.use("Agg")

if TYPE_CHECKING:
    pass


@plot_registry.register("ecdf")
class EcdfPlot(PlotGenerator):
    """Empirical Cumulative Distribution Function (ECDF) plot generator plugin."""

    @property
    def name(self) -> str:
        """Return the unique name of the plot generator."""
        return "ecdf"

    @property
    def description(self) -> str:
        """Return a brief description of what the plot displays."""
        return "Empirical Cumulative Distribution Function (ECDF) plot."

    def is_applicable(self, properties: DataProperties) -> bool:
        """Determine whether the ECDF plot is applicable.

        Requires at least 1 group, and all groups with size >= 1.
        """
        n_groups = properties.n_groups
        group_sizes = properties.group_sizes
        if n_groups < 1:
            return False
        return all(size >= 1 for size in group_sizes.values())

    def generate(self, df: pd.DataFrame, group_col: str, value_col: str) -> PlotResult:
        """Generate an ECDF plot.

        Args:
            df: The dataset DataFrame.
            group_col: Column name representing the groups.
            value_col: Column name representing the values.

        Returns:
            A PlotResult.
        """
        groups = sorted(df[group_col].dropna().unique())

        fig, ax = plt.subplots()
        try:
            for g in groups:
                group_data = df[df[group_col] == g][value_col].dropna().values
                if len(group_data) > 0:
                    x = np.sort(group_data)
                    y = np.arange(1, len(x) + 1) / len(x)
                    ax.step(x, y, where="post", label=str(g))

            ax.set_title(f"ECDF of {value_col} by {group_col}")
            ax.set_xlabel(value_col)
            ax.set_ylabel("F_n(x)")
            ax.legend()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        finally:
            plt.close(fig)

        return PlotResult(
            column_name=None,
            plot_type=self.name,
            image_base64=image_base64,
            content_type="image/png",
        )
