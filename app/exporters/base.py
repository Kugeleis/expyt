"""Exporter base classes and utilities.

All exporters must inherit from ``Exporter`` and register themselves using
the global ``exporter_registry``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel, Field

from app.core.registry import Registry

if TYPE_CHECKING:
    from app.plots.base import PlotResult
    from app.stats.base import StatResult


class ExportResult(BaseModel):
    """The result of an export operation."""

    content: bytes = Field(..., description="Binary content of the exported file.")
    content_type: str = Field(..., description="MIME type of the content.")
    filename: str = Field(..., description="Suggested filename for download.")


class Exporter(ABC):
    """Abstract base class for all report exporters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of the exporter."""
        ...

    @property
    @abstractmethod
    def content_type(self) -> str:
        """Return the default content type of the exporter."""
        ...

    @abstractmethod
    def export(
        self,
        stat_results: list[StatResult],
        plots: list[PlotResult],
        df: pd.DataFrame,
    ) -> ExportResult:
        """Export the session results, plots, or dataset.

        Args:
            stat_results: A list of calculated statistical test results.
            plots: A list of generated plot results.
            df: The filtered dataset DataFrame.

        Returns:
            An ExportResult containing the file contents, MIME type, and filename.
        """
        ...


exporter_registry: Registry[Exporter] = Registry("exporter")
