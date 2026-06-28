"""Negative and validation integration tests for the wizard flow."""

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
    # create standard normal data
    g1 = np.random.normal(loc=10.0, scale=0.5, size=15)
    g2 = np.random.normal(loc=12.0, scale=0.5, size=15)
    df = pd.DataFrame(
        {
            "group": ["A"] * 15 + ["B"] * 15,
            "value": np.concatenate([g1, g2]),
        }
    )
    df.to_csv(tmp_path / "normal_data.csv", index=False)
    # create numeric group data
    df_num = pd.DataFrame(
        {
            "group": [1.0] * 10 + [2.0] * 10,
            "value": np.random.normal(loc=10.0, scale=1.0, size=20),
        }
    )
    df_num.to_csv(tmp_path / "uploaded_numeric_group.csv", index=False)
    return tmp_path


@pytest.fixture
def client(test_data_dir: Path) -> Generator[TestClient, None, None]:
    """TestClient with overridden dataset repository dependency."""
    repo = CsvDatasetRepository(test_data_dir)
    app.dependency_overrides[get_dataset_repository] = lambda: repo
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_wizard_negative_missing_session(client: TestClient) -> None:
    """Endpoints return 404 for invalid/missing session IDs."""
    resp = client.get("/wizard/sessions/nonexistent_id")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


def test_wizard_negative_step_guards(client: TestClient) -> None:
    """Wizard step guards block moving to steps prematurely."""
    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    resp = client.post(
        f"/wizard/sessions/{session_id}/method",
        json={"selected_method": "ttest_ind"},
    )
    assert resp.status_code == 400
    err = resp.json()
    assert "prerequisite" in err["detail"]
    assert "dataset_selection" in err["missing"]

    csv_content = b"group,value\nA,10.0\nA,10.5\nA,11.0\nA,10.2\nA,9.8\nB,12.0\nB,12.5\nB,13.0\nB,12.2\nB,11.8\n"
    files = {"file": ("normal_data.csv", csv_content, "text/csv")}
    client.post("/wizard/upload", files=files)


def test_wizard_negative_invalid_payloads(client: TestClient) -> None:
    """Invalid requests return 400 Bad Request."""
    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    csv_content = b"group,value\nA,10.0\nA,10.5\nA,11.0\nA,10.2\nA,9.8\nB,12.0\nB,12.5\nB,13.0\nB,12.2\nB,11.8\n"
    files = {"file": ("normal_data.csv", csv_content, "text/csv")}
    client.post("/wizard/upload", files=files)

    resp = client.post(
        f"/wizard/sessions/{session_id}/dataset",
        json={
            "dataset_id": "normal_data",
            "group_column": "missing_group",
            "selected_value_columns": [],
        },
    )
    assert resp.status_code == 400
    assert "Group column" in resp.json()["detail"]

    client.post(
        f"/wizard/sessions/{session_id}/dataset",
        json={
            "dataset_id": "normal_data",
            "group_column": "group",
            "selected_value_columns": [],
        },
    )

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


def test_wizard_go_to_invalid_step(client: TestClient) -> None:
    """Going to an unknown step returns 400."""
    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    resp = client.post(f"/wizard/sessions/{session_id}/go-to/nonexistent")
    assert resp.status_code == 400
    assert "Unknown wizard step" in resp.json()["detail"]


def test_wizard_go_to_uncompleted_step_fails(client: TestClient) -> None:
    """Cannot go-to a step whose prerequisites are not met."""
    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    resp = client.post(f"/wizard/sessions/{session_id}/go-to/stat_method")
    assert resp.status_code == 400
    assert "prerequisite" in resp.json()["detail"]


def test_select_dataset_numeric_group_column_fails(client: TestClient) -> None:
    """Selecting a numeric column as group column returns 400."""
    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    resp = client.post(
        f"/wizard/sessions/{session_id}/dataset",
        json={
            "dataset_id": "uploaded_numeric_group",
            "group_column": "group",
            "selected_value_columns": ["value"],
        },
    )
    assert resp.status_code == 400
    assert "must be discrete/categorical, but it is numeric" in resp.json()["detail"]
