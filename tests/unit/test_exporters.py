"""Unit tests for the report exporters."""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest

from app.exporters.base import exporter_registry
from app.exporters.builtin.csv_exporter import CsvExporter
from app.exporters.builtin.json_exporter import JsonExporter
from app.exporters.builtin.pdf_exporter import PdfExporter
from app.plots.base import PlotResult
from app.stats.base import StatResult


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return a sample DataFrame for exporter testing."""
    return pd.DataFrame(
        {
            "group": ["A", "B", "A", "B"],
            "value": [1.0, 2.0, 1.5, 2.5],
        }
    )


@pytest.fixture
def sample_stat_results() -> list[StatResult]:
    """Return a sample StatResult for exporter testing."""
    return [
        StatResult(
            method_name="test_method",
            test_statistic=1.2345,
            p_value=0.0432,
            effect_size=0.85,
            summary="Test method summary showing p < 0.05",
        )
    ]


@pytest.fixture
def sample_plots() -> list[PlotResult]:
    """Return a list with a dummy PlotResult for exporter testing."""
    # A tiny 1x1 transparent PNG base64 string
    dummy_png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8"
        "AAAAASUVORK5CYII="
    )
    return [
        PlotResult(
            plot_type="boxplot",
            image_base64=dummy_png_b64,
            content_type="image/png",
        )
    ]


def test_registrations() -> None:
    """Verify built-in exporters are registered."""
    assert isinstance(exporter_registry.get("csv"), CsvExporter)
    assert isinstance(exporter_registry.get("json"), JsonExporter)
    assert isinstance(exporter_registry.get("pdf"), PdfExporter)


def test_csv_exporter(
    sample_df: pd.DataFrame,
    sample_stat_results: list[StatResult],
    sample_plots: list[PlotResult],
) -> None:
    """Test CsvExporter output."""
    csv_exp = CsvExporter()
    res = csv_exp.export(sample_stat_results, sample_plots, sample_df)

    assert res.content_type == "text/csv"
    assert res.filename == "dataset_export.csv"
    assert len(res.content) > 0

    # Parse and verify CSV contents
    csv_str = res.content.decode("utf-8")
    df_parsed = pd.read_csv(io_stream_from_str(csv_str))
    assert list(df_parsed.columns) == ["group", "value"]
    assert len(df_parsed) == 4


def test_json_exporter(
    sample_df: pd.DataFrame,
    sample_stat_results: list[StatResult],
    sample_plots: list[PlotResult],
) -> None:
    """Test JsonExporter output."""
    json_exp = JsonExporter()
    res = json_exp.export(sample_stat_results, sample_plots, sample_df)

    assert res.content_type == "application/json"
    assert res.filename == "evaluation_export.json"
    assert len(res.content) > 0

    # Parse JSON and verify keys
    json_str = res.content.decode("utf-8")
    data = json.loads(json_str)

    assert "dataset" in data
    assert len(data["dataset"]) == 4
    assert data["statistical_results"][0]["method_name"] == "test_method"
    assert len(data["plots"]) == 1
    assert data["plots"][0]["plot_type"] == "boxplot"


def test_pdf_exporter(
    sample_df: pd.DataFrame,
    sample_stat_results: list[StatResult],
    sample_plots: list[PlotResult],
) -> None:
    """Test PdfExporter output."""
    pdf_exp = PdfExporter()
    res = pdf_exp.export(sample_stat_results, sample_plots, sample_df)

    assert res.content_type == "application/pdf"
    assert res.filename == "evaluation_report.pdf"
    assert len(res.content) > 0

    # PDF header signature check
    assert res.content.startswith(b"%PDF-")


def test_pdf_exporter_no_stats(
    sample_df: pd.DataFrame,
    sample_plots: list[PlotResult],
) -> None:
    """Test PdfExporter output without statistical results."""
    pdf_exp = PdfExporter()
    res = pdf_exp.export(None, sample_plots, sample_df)

    assert res.content_type == "application/pdf"
    assert res.content.startswith(b"%PDF-")


# Helper helper
def io_stream_from_str(s: str) -> io.StringIO:
    return io.StringIO(s)
