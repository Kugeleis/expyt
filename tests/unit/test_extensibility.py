"""Extensibility proof tests verifying that custom plugins can be registered and run.

This verifies that the wizard system is extensible without modifying core code.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.datasets.repository import CsvDatasetRepository
from app.exporters.base import Exporter, ExportResult, exporter_registry
from app.filters.base import Filter, filter_registry
from app.main import app
from app.plots.base import PlotGenerator, PlotResult, plot_registry
from app.stats.base import DataProperties, StatMethod, StatResult, stat_registry
from app.wizard.router import get_dataset_repository


@filter_registry.register("dummy_filter")
class DummyFilter(Filter):
    """A dummy filter that drops rows where value matches a dummy value."""

    @property
    def name(self) -> str:
        """Return the name of the filter."""
        return "dummy_filter"

    @property
    def description(self) -> str:
        """Return a description of the filter."""
        return "A dummy filter for testing extensibility."

    def apply(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        """Apply the filter."""
        column = params["column"]
        value = params["value"]
        res = df[df[column] != value]
        assert isinstance(res, pd.DataFrame)
        return res

    def validate_params(self, params: dict[str, Any]) -> None:
        """Validate the parameters."""
        if "column" not in params:
            raise ValueError("Missing column")
        if "value" not in params:
            raise ValueError("Missing value")


@stat_registry.register("dummy_stat")
class DummyStatMethod(StatMethod):
    """A dummy statistical method for testing extensibility."""

    @property
    def name(self) -> str:
        """Return the name of the statistical method."""
        return "dummy_stat"

    @property
    def description(self) -> str:
        """Return a description of the statistical method."""
        return "A dummy statistical method."

    def is_applicable(self, properties: DataProperties) -> bool:
        """Check if applicable."""
        return True

    def run(self, groups: dict[str, list[float]]) -> StatResult:
        """Run the statistical method."""
        return StatResult(
            method_name="dummy_stat",
            test_statistic=100.0,
            p_value=0.001,
            effect_size=None,
            summary="Dummy stat run complete.",
        )


@plot_registry.register("dummy_plot")
class DummyPlotGenerator(PlotGenerator):
    """A dummy plot generator for testing extensibility."""

    @property
    def name(self) -> str:
        """Return the name of the plot generator."""
        return "dummy_plot"

    @property
    def description(self) -> str:
        """Return a description of the plot generator."""
        return "A dummy plot generator."

    def is_applicable(self, properties: DataProperties) -> bool:
        """Check if applicable."""
        return True

    def generate(self, df: pd.DataFrame, group_col: str, value_col: str) -> PlotResult:
        """Generate the plot."""
        return PlotResult(
            plot_type="dummy_plot",
            image_base64="ZHVtbXlfcGxvdF9iYXNlNjQ=",
            content_type="image/png",
        )


@exporter_registry.register("dummy_export")
class DummyExporter(Exporter):
    """A dummy exporter for testing extensibility."""

    @property
    def name(self) -> str:
        """Return the name of the exporter."""
        return "dummy_export"

    @property
    def content_type(self) -> str:
        """Return the content type of the exporter."""
        return "text/plain"

    def export(
        self,
        stat_results: list[StatResult],
        plots: list[PlotResult],
        df: pd.DataFrame,
    ) -> ExportResult:
        """Export the results."""
        summary = stat_results[0].summary if stat_results else "None"
        content = f"Dummy report. Stat: {summary}. Plots: {len(plots)}."
        return ExportResult(
            content=content.encode("utf-8"),
            content_type="text/plain",
            filename="dummy_report.txt",
        )


@pytest.fixture(scope="module", autouse=True)
def cleanup_dummy_plugins() -> Generator[None, None, None]:
    """Ensure dummy plugins are removed after this module's tests run."""
    yield
    filter_registry._plugins.pop("dummy_filter", None)
    stat_registry._plugins.pop("dummy_stat", None)
    plot_registry._plugins.pop("dummy_plot", None)
    exporter_registry._plugins.pop("dummy_export", None)


@pytest.fixture
def temp_dataset_dir(tmp_path: Path) -> Path:
    """Create a temporary dataset folder with a test CSV file."""
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", "C"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    df.to_csv(tmp_path / "dummy_data.csv", index=False)
    return tmp_path


@pytest.fixture
def client(temp_dataset_dir: Path) -> Generator[TestClient, None, None]:
    """Provide a TestClient with overridden dataset repository."""
    repo = CsvDatasetRepository(temp_dataset_dir)
    app.dependency_overrides[get_dataset_repository] = lambda: repo
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_custom_plugin_registration() -> None:
    """Verify that the custom plugins are registered correctly in the registries."""
    assert "dummy_filter" in filter_registry.list_all()
    assert "dummy_stat" in stat_registry.list_all()
    assert "dummy_plot" in plot_registry.list_all()
    assert "dummy_export" in exporter_registry.list_all()


def test_custom_plugins_in_wizard_flow(client: TestClient) -> None:
    """Run a full evaluation wizard flow using only custom plugins."""
    # Create Session
    resp = client.post("/wizard/sessions")
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]

    # Step 1: Select dataset
    resp = client.post(
        f"/wizard/sessions/{session_id}/dataset",
        json={
            "dataset_id": "dummy_data",
            "group_column": "group",
            "selected_value_columns": [],
        },
    )
    assert resp.status_code == 200

    # Step 2: Configure custom filter (dummy_filter)
    # This filter will drop group C (value 5.0)
    resp = client.post(
        f"/wizard/sessions/{session_id}/filters",
        json={
            "filters_config": [
                {
                    "name": "dummy_filter",
                    "params": {"column": "value", "value": 5.0},
                }
            ]
        },
    )
    assert resp.status_code == 200

    # Verify that the custom stat method is returned as applicable
    resp = client.get(f"/wizard/sessions/{session_id}/methods")
    assert resp.status_code == 200
    methods = [m["name"] for m in resp.json()]
    assert "dummy_stat" in methods

    # Step 3: Select custom stat method
    resp = client.post(
        f"/wizard/sessions/{session_id}/method",
        json={"selected_method": "dummy_stat"},
    )
    assert resp.status_code == 200

    # Step 4: Execute statistical method
    resp = client.get(f"/wizard/sessions/{session_id}/results")
    assert resp.status_code == 200
    assert resp.json()[0]["method_name"] == "dummy_stat"
    assert resp.json()[0]["test_statistic"] == 100.0

    # Verify that the custom plot generator is returned as applicable
    resp = client.get(f"/wizard/sessions/{session_id}/plots")
    assert resp.status_code == 200
    plots = [p["name"] for p in resp.json()]
    assert "dummy_plot" in plots

    # Step 5: Generate custom plot
    resp = client.post(
        f"/wizard/sessions/{session_id}/plots",
        json={"selected_plots": ["dummy_plot"]},
    )
    assert resp.status_code == 200

    # Step 6: Export using custom exporter
    resp = client.post(
        f"/wizard/sessions/{session_id}/export",
        json={"export_format": "dummy_export"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/plain; charset=utf-8"
    assert "Dummy report. Stat: Dummy stat run complete.. Plots: 1." in resp.text
