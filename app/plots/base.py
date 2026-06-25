"""Plot generator base classes and utilities.

All plot generators must inherit from ``PlotGenerator`` and register
themselves using the global ``plot_registry``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel, Field

from app.core.registry import Registry

if TYPE_CHECKING:
    from app.stats.base import DataProperties


class PlotResult(BaseModel):
    """The result of generating a plot."""

    column_name: str | None = Field(
        None,
        description="Name of the dependent variable column shown by this plot.",
    )

    plot_type: str = Field(..., description="Type of the plot (e.g. boxplot).")
    image_base64: str = Field(..., description="Base64 encoded string of the plot image.")
    content_type: str = Field("image/png", description="MIME type of the image.")


class PlotGenerator(ABC):
    """Abstract base class for all plot generators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of the plot generator."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a brief description of what the plot displays."""
        ...

    @abstractmethod
    def is_applicable(self, properties: DataProperties) -> bool:
        """Determine whether this plot is applicable given the data properties.

        Args:
            properties: A DataProperties object containing data properties.

        Returns:
            True if the plot can be generated, False otherwise.
        """
        ...

    @abstractmethod
    def generate(self, df: pd.DataFrame, group_col: str, value_col: str) -> PlotResult:
        """Generate the plot and return it as a PlotResult.

        Args:
            df: The filtered dataset DataFrame.
            group_col: Column name representing the groups.
            value_col: Column name representing the values.

        Returns:
            A PlotResult containing the base64-encoded plot image.
        """
        ...


plot_registry: Registry[PlotGenerator] = Registry("plot")
