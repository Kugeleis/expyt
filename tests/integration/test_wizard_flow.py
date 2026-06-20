"""Integration tests for the full wizard workflow."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.datasets.repository import CsvDatasetRepository
from app.main import app
from app.wizard.router import get_dataset_repository


@pytest.fixture
def test_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with test CSV files."""
    # Create two groups: normally distributed, size=15 each
    np.random.seed(42)
    g1 = np.random.normal(loc=10.0, scale=0.5, size=15)
    g2 = np.random.normal(loc=12.0, scale=0.5, size=15)
    df = pd.DataFrame(
        {
            "group": ["A"] * 15 + ["B"] * 15,
            "value": np.concatenate([g1, g2]),
        }
    )
    df.to_csv(tmp_path / "normal_data.csv", index=False)
    return tmp_path


@pytest.fixture
def client(test_data_dir: Path) -> Generator[TestClient, None, None]:
    """TestClient with overridden dataset repository dependency."""
    # Override dataset repository to point to the temporary folder
    repo = CsvDatasetRepository(test_data_dir)
    app.dependency_overrides[get_dataset_repository] = lambda: repo
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_full_wizard_flow_success(client: TestClient) -> None:
    """Verify a successful walk-through of all wizard steps."""
    # Upload dataset via endpoint
    csv_content = (
        b"group,value\nA,10.0\nA,10.5\nA,11.0\nA,10.2\nA,9.8\n"
        b"B,12.0\nB,12.5\nB,13.0\nB,12.2\nB,11.8\n"
    )
    files = {"file": ("uploaded_normal_data.csv", csv_content, "text/csv")}
    resp = client.post("/wizard/upload", files=files)
    assert resp.status_code == 200
    assert resp.json()["id"] == "uploaded_normal_data"

    # Create session
    resp = client.post("/wizard/sessions")
    assert resp.status_code == 200
    session = resp.json()
    session_id = session["session_id"]
    assert session["current_step"] == "dataset_selection"

    # Step 1: Select dataset
    resp = client.post(
        f"/wizard/sessions/{session_id}/dataset",
        json={
            "dataset_id": "uploaded_normal_data",
            "group_column": "group",
            "value_column": "value",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["dataset_id"] == "uploaded_normal_data"

    # Step 2: Configure filters (filter out values < 5.0 - doesn't drop anything)
    resp = client.post(
        f"/wizard/sessions/{session_id}/filters",
        json={
            "filters_config": [
                {
                    "name": "numeric_range",
                    "params": {"column": "value", "min": 5.0},
                }
            ]
        },
    )
    assert resp.status_code == 200
    assert len(resp.json()["filters_config"]) == 1

    # Check applicable methods helper
    resp = client.get(f"/wizard/sessions/{session_id}/methods")
    assert resp.status_code == 200
    methods = resp.json()
    # "ttest_ind" and "mann_whitney" should be applicable
    method_names = {m["name"] for m in methods}
    assert "ttest_ind" in method_names
    assert "mann_whitney" in method_names

    # Step 3: Choose method
    resp = client.post(
        f"/wizard/sessions/{session_id}/method",
        json={"selected_method": "ttest_ind"},
    )
    assert resp.status_code == 200
    assert resp.json()["selected_method"] == "ttest_ind"

    # Step 4: Run statistical results
    resp = client.get(f"/wizard/sessions/{session_id}/results")
    assert resp.status_code == 200
    res = resp.json()
    assert res["method_name"] == "ttest_ind"
    assert "p_value" in res
    assert "test_statistic" in res

    # Check applicable plots helper
    resp = client.get(f"/wizard/sessions/{session_id}/plots")
    assert resp.status_code == 200
    plots = resp.json()
    plot_names = {p["name"] for p in plots}
    assert "boxplot" in plot_names
    assert "violinplot" in plot_names

    # Step 5: Choose plots
    resp = client.post(
        f"/wizard/sessions/{session_id}/plots",
        json={"selected_plots": ["boxplot", "violinplot"]},
    )
    assert resp.status_code == 200
    session_after_plots = resp.json()
    assert len(session_after_plots["plot_results"]) == 2

    # Step 6: Export results (PDF)
    resp = client.post(
        f"/wizard/sessions/{session_id}/export",
        json={"export_format": "pdf"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.headers["content-disposition"].startswith("attachment; filename=")
    assert resp.content.startswith(b"%PDF-")


def test_wizard_negative_missing_session(client: TestClient) -> None:
    """Endpoints return 404 for invalid/missing session IDs."""
    resp = client.get("/wizard/sessions/nonexistent_id")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


def test_wizard_negative_step_guards(client: TestClient) -> None:
    """Wizard step guards block moving to steps prematurely."""
    # Create session
    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    # Try to skip to method selection without dataset/filters
    resp = client.post(
        f"/wizard/sessions/{session_id}/method",
        json={"selected_method": "ttest_ind"},
    )
    assert resp.status_code == 400
    err = resp.json()
    assert "prerequisite" in err["detail"]
    assert "dataset_selection" in err["missing"]

    # Upload dataset via endpoint first
    csv_content = (
        b"group,value\nA,10.0\nA,10.5\nA,11.0\nA,10.2\nA,9.8\n"
        b"B,12.0\nB,12.5\nB,13.0\nB,12.2\nB,11.8\n"
    )
    files = {"file": ("normal_data.csv", csv_content, "text/csv")}
    client.post("/wizard/upload", files=files)

    # Select dataset
    client.post(
        f"/wizard/sessions/{session_id}/dataset",
        json={
            "dataset_id": "normal_data",
            "group_column": "group",
            "value_column": "value",
        },
    )

    # Now filters is required. Try to run method selection.
    resp = client.post(
        f"/wizard/sessions/{session_id}/method",
        json={"selected_method": "ttest_ind"},
    )
    assert resp.status_code == 400
    assert "filters" in resp.json()["missing"]


def test_wizard_negative_invalid_payloads(client: TestClient) -> None:
    """Invalid requests return 400 Bad Request."""
    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    # Upload dataset via endpoint first
    csv_content = (
        b"group,value\nA,10.0\nA,10.5\nA,11.0\nA,10.2\nA,9.8\n"
        b"B,12.0\nB,12.5\nB,13.0\nB,12.2\nB,11.8\n"
    )
    files = {"file": ("normal_data.csv", csv_content, "text/csv")}
    client.post("/wizard/upload", files=files)

    # Step 1: Missing column
    resp = client.post(
        f"/wizard/sessions/{session_id}/dataset",
        json={
            "dataset_id": "normal_data",
            "group_column": "missing_group",
            "value_column": "value",
        },
    )
    assert resp.status_code == 400
    assert "Group column" in resp.json()["detail"]

    # Set valid dataset
    client.post(
        f"/wizard/sessions/{session_id}/dataset",
        json={
            "dataset_id": "normal_data",
            "group_column": "group",
            "value_column": "value",
        },
    )

    # Step 2: Invalid filter params (min > max)
    resp = client.post(
        f"/wizard/sessions/{session_id}/filters",
        json={
            "filters_config": [
                {
                    "name": "numeric_range",
                    "params": {"column": "value", "min": 20.0, "max": 10.0},
                }
            ]
        },
    )
    assert resp.status_code == 400
    assert "cannot be greater than" in resp.json()["detail"]
