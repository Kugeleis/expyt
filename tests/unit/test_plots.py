"""Unit tests for the plot generators."""

from __future__ import annotations

import base64

import pandas as pd
import pytest

from app.plots.base import plot_registry
from app.plots.builtin.boxplot import BoxPlot
from app.plots.builtin.ecdf import EcdfPlot
from app.plots.builtin.violin import ViolinPlot
from app.stats.base import DataProperties


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return a sample DataFrame for plot testing."""
    return pd.DataFrame(
        {
            "group": ["A"] * 5 + ["B"] * 5,
            "value": [1.0, 2.0, 1.5, 2.5, 3.0, 2.0, 3.0, 2.5, 3.5, 4.0],
        }
    )


def test_registrations() -> None:
    """Verify built-in plot generators are registered."""
    bp = plot_registry.get("boxplot")
    assert isinstance(bp, BoxPlot)
    assert bp.name == "boxplot"
    assert bp.description == "Box plot of values grouped by category."

    vp = plot_registry.get("violinplot")
    assert isinstance(vp, ViolinPlot)
    assert vp.name == "violinplot"
    assert vp.description == "Violin plot of values grouped by category."

    ep = plot_registry.get("ecdf")
    assert isinstance(ep, EcdfPlot)
    assert ep.name == "ecdf"
    assert ep.description == "Empirical Cumulative Distribution Function (ECDF) plot."


def test_boxplot_applicability() -> None:
    """Test boxplot applicability logic."""
    bp = BoxPlot()

    # Applicable
    props_valid = DataProperties(
        n_groups=2,
        group_sizes={"A": 5, "B": 5},
        normality={"A": 0.5, "B": 0.5},
        variance_homogeneity=0.8,
    )
    assert bp.is_applicable(props_valid) is True

    # Inapplicable: No groups
    props_no_groups = DataProperties(
        n_groups=0,
        group_sizes={},
        normality={},
        variance_homogeneity=0.0,
    )
    assert bp.is_applicable(props_no_groups) is False

    # Inapplicable: Empty group
    props_empty_group = DataProperties(
        n_groups=2,
        group_sizes={"A": 0, "B": 5},
        normality={"A": 0.0, "B": 0.5},
        variance_homogeneity=0.0,
    )
    assert bp.is_applicable(props_empty_group) is False


def test_violin_applicability() -> None:
    """Test violinplot applicability logic."""
    vp = ViolinPlot()

    # Applicable: size >= 3
    props_valid = DataProperties(
        n_groups=2,
        group_sizes={"A": 3, "B": 4},
        normality={"A": 0.5, "B": 0.5},
        variance_homogeneity=0.8,
    )
    assert vp.is_applicable(props_valid) is True

    # Inapplicable: size < 3
    props_small_group = DataProperties(
        n_groups=2,
        group_sizes={"A": 2, "B": 5},
        normality={"A": 0.5, "B": 0.5},
        variance_homogeneity=0.8,
    )
    assert vp.is_applicable(props_small_group) is False

    # Inapplicable: n_groups < 1
    props_no_groups = DataProperties(
        n_groups=0,
        group_sizes={},
        normality={},
        variance_homogeneity=0.0,
    )
    assert vp.is_applicable(props_no_groups) is False


def test_ecdf_applicability() -> None:
    """Test ecdf applicability logic."""
    ep = EcdfPlot()

    # Applicable
    props_valid = DataProperties(
        n_groups=2,
        group_sizes={"A": 1, "B": 1},
        normality={"A": 0.0, "B": 0.0},
        variance_homogeneity=0.0,
    )
    assert ep.is_applicable(props_valid) is True

    # Inapplicable: No groups
    props_no_groups = DataProperties(
        n_groups=0,
        group_sizes={},
        normality={},
        variance_homogeneity=0.0,
    )
    assert ep.is_applicable(props_no_groups) is False


def test_boxplot_generate(sample_df: pd.DataFrame) -> None:
    """Test generating a boxplot."""
    bp = BoxPlot()
    res = bp.generate(sample_df, "group", "value")

    assert res.plot_type == "boxplot"
    assert res.content_type == "image/png"
    assert isinstance(res.image_base64, str)
    assert len(res.image_base64) > 0

    # Verify valid PNG signature
    img_bytes = base64.b64decode(res.image_base64)
    assert img_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_violin_generate(sample_df: pd.DataFrame) -> None:
    """Test generating a violin plot."""
    vp = ViolinPlot()
    res = vp.generate(sample_df, "group", "value")

    assert res.plot_type == "violinplot"
    assert res.content_type == "image/png"
    assert isinstance(res.image_base64, str)
    assert len(res.image_base64) > 0

    img_bytes = base64.b64decode(res.image_base64)
    assert img_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_ecdf_generate(sample_df: pd.DataFrame) -> None:
    """Test generating an ECDF plot."""
    ep = EcdfPlot()
    res = ep.generate(sample_df, "group", "value")

    assert res.plot_type == "ecdf"
    assert res.content_type == "image/png"
    assert isinstance(res.image_base64, str)
    assert len(res.image_base64) > 0

    img_bytes = base64.b64decode(res.image_base64)
    assert img_bytes.startswith(b"\x89PNG\r\n\x1a\n")
