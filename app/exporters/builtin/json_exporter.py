"""JSON exporter plugin."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pandas as pd

from app.exporters.base import Exporter, ExportResult, exporter_registry

if TYPE_CHECKING:
    from app.plots.base import PlotResult
    from app.stats.base import StatResult


@exporter_registry.register("json")
class JsonExporter(Exporter):
    """Exporter that saves evaluation session results as a JSON file."""

    @property
    def name(self) -> str:
        """Return the unique name of the exporter."""
        return "json"

    @property
    def content_type(self) -> str:
        """Return the default content type of the exporter."""
        return "application/json"

    def export(
        self,
        stat_results: list[StatResult],
        plots: list[PlotResult],
        df: pd.DataFrame,
    ) -> ExportResult:
        """Export session results and dataset to JSON format.

        Args:
            stat_results: The statistical test results.
            plots: The list of plot results.
            df: The dataset DataFrame.

        Returns:
            An ExportResult containing the JSON bytes.
        """
        data: dict[str, Any] = {
            "dataset": df.to_dict(orient="records"),
            "statistical_results": [r.model_dump() for r in stat_results],
            "plots": [
                {"plot_type": p.plot_type, "image_base64": p.image_base64}
                for p in plots
            ],
        }

        json_str = json.dumps(data, indent=2)
        content_bytes = json_str.encode("utf-8")

        return ExportResult(
            content=content_bytes,
            content_type=self.content_type,
            filename="evaluation_export.json",
        )
