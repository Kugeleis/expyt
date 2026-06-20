"""CSV exporter plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from app.exporters.base import Exporter, ExportResult, exporter_registry

if TYPE_CHECKING:
    from app.plots.base import PlotResult
    from app.stats.base import StatResult


@exporter_registry.register("csv")
class CsvExporter(Exporter):
    """Exporter that saves the filtered dataset as a CSV file."""

    @property
    def name(self) -> str:
        """Return the unique name of the exporter."""
        return "csv"

    @property
    def content_type(self) -> str:
        """Return the default content type of the exporter."""
        return "text/csv"

    def export(
        self,
        stat_results: list[StatResult],
        plots: list[PlotResult],
        df: pd.DataFrame,
    ) -> ExportResult:
        """Export dataset to CSV format.

        Args:
            stat_result: The statistical test result (ignored for CSV).
            plots: The list of plot results (ignored for CSV).
            df: The dataset DataFrame.

        Returns:
            An ExportResult containing the CSV bytes.
        """
        csv_str = df.to_csv(index=False)
        content_bytes = csv_str.encode("utf-8")

        return ExportResult(
            content=content_bytes,
            content_type=self.content_type,
            filename="dataset_export.csv",
        )
