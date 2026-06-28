"""Basic integration tests for the wizard flow."""

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
    # also add a preselect_test.csv for restart/preselect tests
    df_pre = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "numeric_val": np.random.normal(loc=5.0, scale=1.0, size=20),
            "categorical_val": ["Yes", "No"] * 10,
        }
    )
    df_pre.to_csv(tmp_path / "preselect_test.csv", index=False)
    return tmp_path


@pytest.fixture
def client(test_data_dir: Path) -> Generator[TestClient, None, None]:
    """TestClient with overridden dataset repository dependency."""
    repo = CsvDatasetRepository(test_data_dir)
    app.dependency_overrides[get_dataset_repository] = lambda: repo
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_full_wizard_flow_success(client: TestClient) -> None:
    """Verify a successful walk-through of all wizard steps."""
    csv_content = b"group,value\nA,10.0\nA,10.5\nA,11.0\nA,10.2\nA,9.8\nB,12.0\nB,12.5\nB,13.0\nB,12.2\nB,11.8\n"
    files = {"file": ("uploaded_normal_data.csv", csv_content, "text/csv")}
    resp = client.post("/wizard/upload", files=files)
    assert resp.status_code == 200
    assert resp.json()["id"] == "uploaded_normal_data"

    resp = client.post("/wizard/sessions")
    assert resp.status_code == 200
    session = resp.json()
    session_id = session["session_id"]
    assert session["current_step"] == "dataset_selection"

    resp = client.post(
        f"/wizard/sessions/{session_id}/dataset",
        json={
            "dataset_id": "uploaded_normal_data",
            "group_column": "group",
            "selected_value_columns": [],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["dataset_id"] == "uploaded_normal_data"

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

    resp = client.get(f"/wizard/sessions/{session_id}/methods")
    assert resp.status_code == 200
    methods = resp.json()
    method_names = {m["name"] for m in methods}
    assert "ttest_ind" in method_names
    assert "mann_whitney" in method_names

    resp = client.post(
        f"/wizard/sessions/{session_id}/method",
        json={"selected_method": "ttest_ind"},
    )
    assert resp.status_code == 200
    assert resp.json()["selected_method"] == "ttest_ind"

    resp = client.get(f"/wizard/sessions/{session_id}/results")
    assert resp.status_code == 200
    res = resp.json()
    assert res[0]["method_name"] == "ttest_ind"
    assert "p_value" in res[0]
    assert "test_statistic" in res[0]

    resp = client.get(f"/wizard/sessions/{session_id}/plots")
    assert resp.status_code == 200
    plots = resp.json()
    plot_names = {p["name"] for p in plots}
    assert "boxplot" in plot_names
    assert "violinplot" in plot_names

    resp = client.post(
        f"/wizard/sessions/{session_id}/plots",
        json={"selected_plots": ["boxplot", "violinplot"]},
    )
    assert resp.status_code == 200
    session_after_plots = resp.json()
    assert len(session_after_plots["plot_results"]) == 2

    resp = client.post(
        f"/wizard/sessions/{session_id}/export",
        json={"export_format": "pdf"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.headers["content-disposition"].startswith("attachment; filename=")
    assert resp.content.startswith(b"%PDF-")


def test_wizard_back_navigation(client: TestClient) -> None:
    """Navigate back to a previous step, change data, and re-proceed forward."""
    csv_content = b"group,value\nA,10.0\nA,10.5\nA,11.0\nA,10.2\nA,9.8\nB,12.0\nB,12.5\nB,13.0\nB,12.2\nB,11.8\n"
    files = {"file": ("backtest.csv", csv_content, "text/csv")}
    resp = client.post("/wizard/upload", files=files)
    assert resp.status_code == 200

    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    resp = client.post(
        f"/wizard/sessions/{session_id}/dataset",
        json={
            "dataset_id": "backtest",
            "group_column": "group",
            "selected_value_columns": [],
        },
    )
    assert resp.status_code == 200

    resp = client.post(
        f"/wizard/sessions/{session_id}/filters",
        json={"filters_config": []},
    )
    assert resp.status_code == 200

    resp = client.post(
        f"/wizard/sessions/{session_id}/method",
        json={"selected_method": "ttest_ind"},
    )
    assert resp.status_code == 200

    resp = client.get(f"/wizard/sessions/{session_id}/results")
    assert resp.status_code == 200

    resp = client.post(f"/wizard/sessions/{session_id}/go-to/filters")
    assert resp.status_code == 200
    session_data = resp.json()
    assert session_data["current_step"] == "filters"
    assert session_data["selected_method"] is None
    assert not session_data["stat_results"]
    assert session_data["selected_plots"] == []
    assert session_data["plot_results"] == []
    assert session_data["dataset_id"] == "backtest"
    assert session_data["group_column"] == "group"

    resp = client.post(
        f"/wizard/sessions/{session_id}/filters",
        json={
            "filters_config": [
                {
                    "name": "numeric_range",
                    "params": {"column": "value", "min": 10.0},
                }
            ]
        },
    )
    assert resp.status_code == 200

    resp = client.post(
        f"/wizard/sessions/{session_id}/method",
        json={"selected_method": "mann_whitney"},
    )
    assert resp.status_code == 200

    resp = client.get(f"/wizard/sessions/{session_id}/results")
    assert resp.status_code == 200
    assert resp.json()[0]["method_name"] == "mann_whitney"


def test_get_column_unique_values(client: TestClient) -> None:
    """Verify unique values endpoint returns sorted non-null values of a column."""
    csv_content = b"group,value\nB,1.0\nA,2.0\nB,3.0\nC,\n"
    files = {"file": ("uniquetest.csv", csv_content, "text/csv")}
    resp = client.post("/wizard/upload", files=files)
    assert resp.status_code == 200

    resp = client.get("/wizard/datasets/uniquetest/columns/group/unique")
    assert resp.status_code == 200
    assert resp.json() == ["A", "B", "C"]


def test_restart_session_e2e(client: TestClient) -> None:
    """Test that restarting a session creates a new fresh session or redirects."""
    resp = client.post("/wizard/sessions")
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]

    resp = client.post(
        f"/wizard/sessions/{session_id}/select-dataset-id",
        data={"dataset_id": "preselect_test"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200

    resp = client.post(f"/wizard/sessions/{session_id}/restart", headers={"HX-Request": "true"})
    assert resp.status_code == 200
    assert resp.headers.get("HX-Redirect") == "/"

    resp = client.post(f"/wizard/sessions/{session_id}/restart", headers={"Accept": "application/json"})
    assert resp.status_code == 200
    new_session = resp.json()
    assert new_session["session_id"] != session_id
    assert new_session["dataset_id"] is None
    assert new_session["current_step"] == "dataset_selection"


def test_restart_session_works_on_all_steps(client: TestClient) -> None:
    """Test that restarting a session works successfully from all 6 steps."""
    from app.wizard.router import get_session_store

    store = get_session_store()

    for step_name in ["dataset_selection", "filters", "stat_method", "results", "plot_selection", "export"]:
        session = store.create()
        session_id = session.session_id
        session.dataset_id = "preselect_test"
        session.current_step = step_name

        if step_name in ("results", "plot_selection", "export"):
            session.selected_value_columns = ["numeric_val"]
            session.selected_method = "ttest"
            session.stat_results = [{"column_name": "numeric_val", "p_value": 0.01}]

        store.save(session)

        resp = client.post(f"/wizard/sessions/{session_id}/restart", headers={"HX-Request": "true"})
        assert resp.status_code == 200
        assert resp.headers.get("HX-Redirect") == "/"
