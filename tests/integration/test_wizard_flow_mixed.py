"""Integration tests for mixed variables and subgroup selections in wizard flow."""

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
    # also add a preselect_test.csv for general usage
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


def test_wizard_subgroup_selection_filtering(client: TestClient) -> None:
    """Verify that statistical analysis is executed using only selected subgroups."""
    csv_content = (
        b"group,value\n"
        b"A,10.0\nA,10.5\nA,11.0\nA,10.2\nA,9.8\n"
        b"B,12.0\nB,12.5\nB,13.0\nB,12.2\nB,11.8\n"
        b"C,100.0\nC,105.0\nC,110.0\nC,102.0\nC,98.0\n"
    )
    files = {"file": ("subgroup_test.csv", csv_content, "text/csv")}
    resp = client.post("/wizard/upload", files=files)
    assert resp.status_code == 200

    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    resp = client.post(
        f"/wizard/sessions/{session_id}/dataset",
        json={
            "dataset_id": "subgroup_test",
            "group_column": "group",
            "selected_value_columns": ["value"],
            "selected_groups": ["A", "B"],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["selected_groups"] == ["A", "B"]

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
    results = resp.json()
    assert len(results) == 1
    assert results[0]["test_statistic"] < 0
    assert abs(results[0]["test_statistic"] - (-6.741998624632423)) < 1e-3
    assert results[0]["p_value"] < 0.001


def test_wizard_discrete_columns_stay_dependent_variables(client: TestClient) -> None:
    """Verify that other discrete columns are allowed and stay as dependent variables."""
    csv_content = b"group1,group2,value\nA,X,10.0\nA,Y,10.5\nA,X,11.0\nB,Y,12.0\nB,X,12.5\nB,Y,13.0\n"
    files = {"file": ("discrete_dep_test.csv", csv_content, "text/csv")}
    resp = client.post("/wizard/upload", files=files)
    assert resp.status_code == 200

    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    resp = client.post(
        f"/wizard/sessions/{session_id}/dataset",
        json={
            "dataset_id": "discrete_dep_test",
            "group_column": "group1",
            "selected_value_columns": [],
            "selected_discrete_columns": [],
        },
    )
    assert resp.status_code == 200
    res_data = resp.json()
    assert "group2" in res_data["selected_discrete_columns"]
    assert "value" in res_data["selected_value_columns"]

    resp = client.post(
        f"/wizard/sessions/{session_id}/dataset",
        json={
            "dataset_id": "discrete_dep_test",
            "group_column": "group1",
            "selected_discrete_columns": ["group2"],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["selected_discrete_columns"] == ["group2"]


def test_wizard_mixed_columns_processing(client: TestClient) -> None:
    """Verify that mixed continuous and discrete dependent columns are processed independently."""
    csv_content = b"group,value,category\nA,10.0,Yes\nA,10.5,Yes\nA,11.0,No\nB,12.0,No\nB,12.5,No\nB,13.0,Yes\n"
    files = {"file": ("mixed_test.csv", csv_content, "text/csv")}
    resp = client.post("/wizard/upload", files=files)
    assert resp.status_code == 200

    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    resp = client.post(
        f"/wizard/sessions/{session_id}/dataset",
        json={
            "dataset_id": "mixed_test",
            "group_column": "group",
            "selected_value_columns": [],
            "selected_discrete_columns": [],
        },
    )
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["selected_value_columns"] == ["value"]
    assert res_data["selected_discrete_columns"] == ["category"]

    resp = client.post(
        f"/wizard/sessions/{session_id}/filters",
        json={"filters_config": []},
    )
    assert resp.status_code == 200

    resp = client.get(f"/wizard/sessions/{session_id}/methods")
    assert resp.status_code == 200
    methods = resp.json()

    continuous_names = {m["name"] for m in methods if m["variable_type"] == "continuous"}
    discrete_names = {m["name"] for m in methods if m["variable_type"] == "discrete"}

    assert "ttest_ind" in continuous_names
    assert "chi_square" in discrete_names

    resp = client.post(
        f"/wizard/sessions/{session_id}/method",
        json={
            "selected_method": "ttest_ind",
            "selected_discrete_method": "chi_square",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["selected_method"] == "ttest_ind"
    assert resp.json()["selected_discrete_method"] == "chi_square"

    resp = client.get(f"/wizard/sessions/{session_id}/results")
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 2

    col_names = {r["column_name"] for r in results}
    assert "value" in col_names
    assert "category" in col_names

    value_res = next(r for r in results if r["column_name"] == "value")
    category_res = next(r for r in results if r["column_name"] == "category")

    assert value_res["method_name"] == "ttest_ind"
    assert category_res["method_name"] == "chi_square"


def test_nycflights_origin_group(client: TestClient) -> None:
    """Verify that selecting origin as the group column works correctly."""
    with open("test_data/nycflights.csv", "rb") as f:
        csv_content = f.read()

    files = {"file": ("nycflights.csv", csv_content, "text/csv")}
    resp = client.post("/wizard/upload", files=files)
    assert resp.status_code == 200
    assert resp.json()["id"] == "nycflights"

    resp = client.post("/wizard/sessions")
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]

    payload = {
        "dataset_id": "nycflights",
        "group_column": "origin",
        "selected_value_columns": [
            "year",
            "month",
            "day",
            "dep_time",
            "dep_delay",
            "arr_time",
            "arr_delay",
            "flight",
            "air_time",
            "distance",
            "hour",
            "minute",
        ],
        "selected_discrete_columns": ["tailnum", "dest", "carrier"],
        "selected_groups": ["EWR", "LGA", "JFK"],
    }
    resp = client.post(f"/wizard/sessions/{session_id}/dataset", json=payload)
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["group_column"] == "origin"
    assert set(res_data["selected_value_columns"]) == set(payload["selected_value_columns"])
    assert set(res_data["selected_discrete_columns"]) == set(payload["selected_discrete_columns"])
