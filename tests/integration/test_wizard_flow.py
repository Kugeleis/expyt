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
    csv_content = b"group,value\nA,10.0\nA,10.5\nA,11.0\nA,10.2\nA,9.8\nB,12.0\nB,12.5\nB,13.0\nB,12.2\nB,11.8\n"
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
            "selected_value_columns": [],
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
    assert res[0]["method_name"] == "ttest_ind"
    assert "p_value" in res[0]
    assert "test_statistic" in res[0]

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
    csv_content = b"group,value\nA,10.0\nA,10.5\nA,11.0\nA,10.2\nA,9.8\nB,12.0\nB,12.5\nB,13.0\nB,12.2\nB,11.8\n"
    files = {"file": ("normal_data.csv", csv_content, "text/csv")}
    client.post("/wizard/upload", files=files)


def test_wizard_negative_invalid_payloads(client: TestClient) -> None:
    """Invalid requests return 400 Bad Request."""
    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    # Upload dataset via endpoint first
    csv_content = b"group,value\nA,10.0\nA,10.5\nA,11.0\nA,10.2\nA,9.8\nB,12.0\nB,12.5\nB,13.0\nB,12.2\nB,11.8\n"
    files = {"file": ("normal_data.csv", csv_content, "text/csv")}
    client.post("/wizard/upload", files=files)

    # Step 1: Missing column
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

    # Set valid dataset
    client.post(
        f"/wizard/sessions/{session_id}/dataset",
        json={
            "dataset_id": "normal_data",
            "group_column": "group",
            "selected_value_columns": [],
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


def test_wizard_back_navigation(client: TestClient) -> None:
    """Navigate back to a previous step, change data, and re-proceed forward."""
    # Upload dataset
    csv_content = b"group,value\nA,10.0\nA,10.5\nA,11.0\nA,10.2\nA,9.8\nB,12.0\nB,12.5\nB,13.0\nB,12.2\nB,11.8\n"
    files = {"file": ("backtest.csv", csv_content, "text/csv")}
    resp = client.post("/wizard/upload", files=files)
    assert resp.status_code == 200

    # Create session
    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    # Step 1: Select dataset
    resp = client.post(
        f"/wizard/sessions/{session_id}/dataset",
        json={
            "dataset_id": "backtest",
            "group_column": "group",
            "selected_value_columns": [],
        },
    )
    assert resp.status_code == 200

    # Step 2: Configure filters
    resp = client.post(
        f"/wizard/sessions/{session_id}/filters",
        json={"filters_config": []},
    )
    assert resp.status_code == 200

    # Step 3: Choose method
    resp = client.post(
        f"/wizard/sessions/{session_id}/method",
        json={"selected_method": "ttest_ind"},
    )
    assert resp.status_code == 200
    assert resp.json()["selected_method"] == "ttest_ind"

    # Step 4: Run results
    resp = client.get(f"/wizard/sessions/{session_id}/results")
    assert resp.status_code == 200

    # --- Now go back to filters ---
    resp = client.post(f"/wizard/sessions/{session_id}/go-to/filters")
    assert resp.status_code == 200
    session_data = resp.json()
    assert session_data["current_step"] == "filters"
    # Downstream state should be cleared
    assert session_data["selected_method"] is None
    assert not session_data["stat_results"]
    assert session_data["selected_plots"] == []
    assert session_data["plot_results"] == []
    # Upstream state should be preserved
    assert session_data["dataset_id"] == "backtest"
    assert session_data["group_column"] == "group"

    # Re-do step 2 with a different filter
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

    # Re-do step 3 with a different method
    resp = client.post(
        f"/wizard/sessions/{session_id}/method",
        json={"selected_method": "mann_whitney"},
    )
    assert resp.status_code == 200
    assert resp.json()["selected_method"] == "mann_whitney"

    # Re-do step 4
    resp = client.get(f"/wizard/sessions/{session_id}/results")
    assert resp.status_code == 200
    assert resp.json()[0]["method_name"] == "mann_whitney"


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

    # Try to go to stat_method without having completed dataset/filters
    resp = client.post(f"/wizard/sessions/{session_id}/go-to/stat_method")
    assert resp.status_code == 400
    assert "prerequisite" in resp.json()["detail"]


def test_select_dataset_numeric_group_column_fails(client: TestClient) -> None:
    """Selecting a numeric column as group column returns 400."""
    # Upload dataset via endpoint
    csv_content = b"group,value\n1.0,10.0\n1.0,10.5\n2.0,11.0\n2.0,10.2\n"
    files = {"file": ("uploaded_numeric_group.csv", csv_content, "text/csv")}
    resp = client.post("/wizard/upload", files=files)
    assert resp.status_code == 200

    # Create session
    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    # Try to select numeric 'group' column (which is float64) as grouping column
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


def test_get_column_unique_values(client: TestClient) -> None:
    """Verify unique values endpoint returns sorted non-null values of a column."""
    # Upload dataset
    csv_content = b"group,value\nB,1.0\nA,2.0\nB,3.0\nC,\n"
    files = {"file": ("uniquetest.csv", csv_content, "text/csv")}
    resp = client.post("/wizard/upload", files=files)
    assert resp.status_code == 200

    # Fetch unique values
    resp = client.get("/wizard/datasets/uniquetest/columns/group/unique")
    assert resp.status_code == 200
    assert resp.json() == ["A", "B", "C"]


def test_wizard_subgroup_selection_filtering(client: TestClient) -> None:
    """Verify that statistical analysis is executed.

    Uses only the selected subgroups.
    """
    # Upload dataset with groups A, B, C
    csv_content = (
        b"group,value\n"
        b"A,10.0\nA,10.5\nA,11.0\nA,10.2\nA,9.8\n"
        b"B,12.0\nB,12.5\nB,13.0\nB,12.2\nB,11.8\n"
        b"C,100.0\nC,105.0\nC,110.0\nC,102.0\nC,98.0\n"
    )
    files = {"file": ("subgroup_test.csv", csv_content, "text/csv")}
    resp = client.post("/wizard/upload", files=files)
    assert resp.status_code == 200

    # Create session
    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    # Step 1: Select dataset, but only include subgroups A and B (exclude C)
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

    # Step 2: Configure empty filters
    resp = client.post(
        f"/wizard/sessions/{session_id}/filters",
        json={"filters_config": []},
    )
    assert resp.status_code == 200

    # Step 3: Choose method (ttest_ind)
    resp = client.post(
        f"/wizard/sessions/{session_id}/method",
        json={"selected_method": "ttest_ind"},
    )
    assert resp.status_code == 200

    # Step 4: Run statistical results
    resp = client.get(f"/wizard/sessions/{session_id}/results")
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 1
    # Group A mean ~10.3, Group B mean ~12.3. Test statistic is ~-6.742
    assert results[0]["test_statistic"] < 0
    assert abs(results[0]["test_statistic"] - (-6.741998624632423)) < 1e-3
    assert results[0]["p_value"] < 0.001


def test_wizard_discrete_columns_stay_dependent_variables(client: TestClient) -> None:
    """Verify that other discrete columns are allowed and stay as dependent variables."""
    # Upload dataset with two discrete columns: group1, group2 and one numeric column: value
    csv_content = b"group1,group2,value\nA,X,10.0\nA,Y,10.5\nA,X,11.0\nB,Y,12.0\nB,X,12.5\nB,Y,13.0\n"
    files = {"file": ("discrete_dep_test.csv", csv_content, "text/csv")}
    resp = client.post("/wizard/upload", files=files)
    assert resp.status_code == 200

    # Create session
    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    # Step 1: Select dataset. Select group1 as group_column.
    # By default, all other columns (group2, value) should be resolved as dependent variables.
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

    # Now verify we can explicitly request a discrete column as the dependent variable.
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
    # Upload mixed dataset: group column, continuous value, discrete category
    csv_content = b"group,value,category\nA,10.0,Yes\nA,10.5,Yes\nA,11.0,No\nB,12.0,No\nB,12.5,No\nB,13.0,Yes\n"
    files = {"file": ("mixed_test.csv", csv_content, "text/csv")}
    resp = client.post("/wizard/upload", files=files)
    assert resp.status_code == 200

    # Create session
    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    # Step 1: Select dataset, auto-populate all columns (both lists empty)
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

    # Step 2: Configure empty filters
    resp = client.post(
        f"/wizard/sessions/{session_id}/filters",
        json={"filters_config": []},
    )
    assert resp.status_code == 200

    # Check applicable methods has both continuous and discrete methods flat-listed with type info
    resp = client.get(f"/wizard/sessions/{session_id}/methods")
    assert resp.status_code == 200
    methods = resp.json()

    continuous_names = {m["name"] for m in methods if m["variable_type"] == "continuous"}
    discrete_names = {m["name"] for m in methods if m["variable_type"] == "discrete"}

    assert "ttest_ind" in continuous_names
    assert "chi_square" in discrete_names

    # Step 3: Choose method for both continuous and discrete columns
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

    # Step 4: Run statistical results and verify mixed results are returned
    resp = client.get(f"/wizard/sessions/{session_id}/results")
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 2

    col_names = {r["column_name"] for r in results}
    assert "value" in col_names
    assert "category" in col_names

    # Assert methods matches the column type
    value_res = next(r for r in results if r["column_name"] == "value")
    category_res = next(r for r in results if r["column_name"] == "category")

    assert value_res["method_name"] == "ttest_ind"
    assert category_res["method_name"] == "chi_square"


def test_nycflights_origin_group(client: TestClient) -> None:
    """Verify that selecting origin as the group column works correctly with the expected frontend payload."""
    # Read the real nycflights.csv data
    with open("test_data/nycflights.csv", "rb") as f:
        csv_content = f.read()

    files = {"file": ("nycflights.csv", csv_content, "text/csv")}
    resp = client.post("/wizard/upload", files=files)
    assert resp.status_code == 200
    assert resp.json()["id"] == "nycflights"

    # Create session
    resp = client.post("/wizard/sessions")
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]

    # Post Step 1: Select dataset with group_column="origin"
    # value columns and discrete columns are populated matching the frontend state
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


def test_select_cluster_column_numeric_fails(client: TestClient) -> None:
    """Selecting a numeric column as cluster column (L1) returns 400."""
    csv_content = b"group,cluster,value\nA,1.0,10.0\nA,1.0,10.5\nB,2.0,11.0\nB,2.0,10.2\n"
    files = {"file": ("uploaded_numeric_cluster.csv", csv_content, "text/csv")}
    resp = client.post("/wizard/upload", files=files)
    assert resp.status_code == 200

    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    # Toggle hierarchy to enable hierarchical mode
    resp = client.post(f"/wizard/sessions/{session_id}/toggle-hierarchy?enabled=true", headers={"HX-Request": "true"})
    assert resp.status_code == 200

    # Select dataset
    resp = client.post(
        f"/wizard/sessions/{session_id}/select-dataset-id",
        data={"dataset_id": "uploaded_numeric_cluster"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200

    # Try to submit config with numeric 'cluster' (which is float64) as cluster column
    # via HTMX endpoint
    resp = client.post(
        f"/wizard/sessions/{session_id}/submit-dataset-config",
        data={
            "group_column": "group",
            "cluster_col": "cluster",
            "selected_value_columns": ["value"],
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 400
    assert "must be discrete/categorical, but it is numeric" in resp.text


def test_set_hierarchy_numeric_cols_fails(client: TestClient) -> None:
    """Hierarchy API validation fails if group or cluster column is numeric."""
    csv_content = b"group,cluster,value\n1.0,X,10.0\n1.0,Y,10.5\n2.0,X,11.0\n2.0,Y,10.2\n"
    files = {"file": ("uploaded_hier_numeric.csv", csv_content, "text/csv")}
    resp = client.post("/wizard/upload", files=files)
    assert resp.status_code == 200

    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    # Set dataset first
    resp = client.post(
        f"/wizard/sessions/{session_id}/dataset",
        json={
            "dataset_id": "uploaded_hier_numeric",
            "group_column": "cluster",
            "selected_value_columns": ["value"],
        },
    )
    assert resp.status_code == 200

    # Try to set hierarchy with numeric group_col
    resp = client.post(
        f"/wizard/sessions/{session_id}/hierarchy",
        json={
            "group_col": "group",
            "cluster_col": "cluster",
        },
    )
    assert resp.status_code == 400
    assert "Group column 'group' must be discrete/categorical, but it is numeric." in resp.json()["detail"]

    # Try to set hierarchy with numeric cluster_col
    csv_content_cluster = b"group,cluster,value\nA,1.0,10.0\nA,2.0,10.5\nB,1.0,11.0\nB,2.0,10.2\n"
    files = {"file": ("uploaded_hier_numeric_cluster.csv", csv_content_cluster, "text/csv")}
    resp = client.post("/wizard/upload", files=files)
    assert resp.status_code == 200

    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    resp = client.post(
        f"/wizard/sessions/{session_id}/dataset",
        json={
            "dataset_id": "uploaded_hier_numeric_cluster",
            "group_column": "group",
            "selected_value_columns": ["value"],
        },
    )
    assert resp.status_code == 200

    resp = client.post(
        f"/wizard/sessions/{session_id}/hierarchy",
        json={
            "group_col": "group",
            "cluster_col": "cluster",
        },
    )
    assert resp.status_code == 400
    assert "Cluster column 'cluster' must be discrete/categorical, but it is numeric." in resp.json()["detail"]


def test_update_group_and_cluster_cols_populate_defaults(client: TestClient) -> None:
    """Verifies that update-group-col and update-cluster-col populate default selections."""
    csv_content = b"group,cluster,value\nA,X,10.0\nA,Y,10.5\nB,X,11.0\nB,Y,10.2\n"
    files = {"file": ("defaults_test.csv", csv_content, "text/csv")}
    resp = client.post("/wizard/upload", files=files)
    assert resp.status_code == 200

    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    # Select dataset id
    resp = client.post(
        f"/wizard/sessions/{session_id}/select-dataset-id",
        data={"dataset_id": "defaults_test"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200

    # Update group column
    resp = client.post(
        f"/wizard/sessions/{session_id}/update-group-col",
        data={"group_column": "group"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200

    # Verify session has selected_groups populated by default
    resp_get = client.get(f"/wizard/sessions/{session_id}")
    assert resp_get.json()["selected_groups"] == ["A", "B"]

    # Toggle hierarchy to true
    resp = client.post(f"/wizard/sessions/{session_id}/toggle-hierarchy?enabled=true", headers={"HX-Request": "true"})
    assert resp.status_code == 200

    # Update cluster column
    resp = client.post(
        f"/wizard/sessions/{session_id}/update-cluster-col",
        data={"cluster_col": "cluster"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200

    # Verify session has selected_clusters populated by default
    resp_get = client.get(f"/wizard/sessions/{session_id}")
    assert resp_get.json()["hierarchy"]["selected_clusters"] == ["X", "Y"]


def test_select_cluster_column_same_as_group_fails(client: TestClient) -> None:
    """Selecting the same column for both group and cluster returns 400."""
    csv_content = b"group,cluster,value\nA,X,10.0\nA,Y,10.5\nB,X,11.0\nB,Y,10.2\n"
    files = {"file": ("uploaded_same_col.csv", csv_content, "text/csv")}
    resp = client.post("/wizard/upload", files=files)
    assert resp.status_code == 200

    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    resp = client.post(f"/wizard/sessions/{session_id}/toggle-hierarchy?enabled=true", headers={"HX-Request": "true"})
    assert resp.status_code == 200

    resp = client.post(
        f"/wizard/sessions/{session_id}/select-dataset-id",
        data={"dataset_id": "uploaded_same_col"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200

    # Try to submit config with cluster_col equal to group_column
    resp = client.post(
        f"/wizard/sessions/{session_id}/submit-dataset-config",
        data={
            "group_column": "group",
            "cluster_col": "group",
            "selected_value_columns": ["value"],
        },
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 400
    assert "Cluster column must not be the same as the group column." in resp.text


def test_set_hierarchy_same_cols_fails(client: TestClient) -> None:
    """Hierarchy API validation fails if group_col is equal to cluster_col."""
    csv_content = b"group,cluster,value\nA,X,10.0\nA,Y,10.5\nB,X,11.0\nB,Y,10.2\n"
    files = {"file": ("uploaded_hier_same.csv", csv_content, "text/csv")}
    resp = client.post("/wizard/upload", files=files)
    assert resp.status_code == 200

    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    resp = client.post(
        f"/wizard/sessions/{session_id}/dataset",
        json={
            "dataset_id": "uploaded_hier_same",
            "group_column": "group",
            "selected_value_columns": ["value"],
        },
    )
    assert resp.status_code == 200

    # Try to set hierarchy with group_col == cluster_col
    resp = client.post(
        f"/wizard/sessions/{session_id}/hierarchy",
        json={
            "group_col": "group",
            "cluster_col": "group",
        },
    )
    assert resp.status_code == 400
    assert "Cluster column must not be the same as the group column." in resp.json()["detail"]


def test_select_dataset_preselects_all_dependent_columns(client: TestClient) -> None:
    """Select dataset endpoint preselects all dependent columns by default."""
    csv_content = b"group,numeric_val,categorical_val\nA,10.0,Yes\nA,10.5,No\nB,11.0,Yes\nB,10.2,No\n"
    files = {"file": ("preselect_test.csv", csv_content, "text/csv")}
    resp = client.post("/wizard/upload", files=files)
    assert resp.status_code == 200

    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    # Select dataset
    resp = client.post(
        f"/wizard/sessions/{session_id}/select-dataset-id",
        data={"dataset_id": "preselect_test"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200

    # Update group column to "group"
    resp = client.post(
        f"/wizard/sessions/{session_id}/update-group-col",
        data={"group_column": "group"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200

    # Verify session has both columns pre-populated by default and the group column excluded
    resp_get = client.get(f"/wizard/sessions/{session_id}")
    assert resp_get.json()["selected_value_columns"] == ["numeric_val"]
    assert resp_get.json()["selected_discrete_columns"] == ["categorical_val"]


def test_restart_session_e2e(client: TestClient) -> None:
    """Test that restarting a session creates a new fresh session or redirects."""
    # 1. Create a session
    resp = client.post("/wizard/sessions")
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]

    # Select dataset to make session dirty
    resp = client.post(
        f"/wizard/sessions/{session_id}/select-dataset-id",
        data={"dataset_id": "preselect_test"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200

    # 2. Restart via HTMX request
    resp = client.post(f"/wizard/sessions/{session_id}/restart", headers={"HX-Request": "true"})
    assert resp.status_code == 200
    # HTMX request should receive the HX-Redirect header to reload root
    assert resp.headers.get("HX-Redirect") == "/"

    # 3. Restart via JSON request (standard REST client)
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
        # Create a session
        session = store.create()
        session_id = session.session_id
        session.dataset_id = "preselect_test"
        session.current_step = step_name

        if step_name in ("results", "plot_selection", "export"):
            session.selected_value_columns = ["numeric_val"]
            session.selected_method = "ttest"
            session.stat_results = [{"column_name": "numeric_val", "p_value": 0.01}]

        store.save(session)

        # Call restart endpoint
        resp = client.post(f"/wizard/sessions/{session_id}/restart", headers={"HX-Request": "true"})
        assert resp.status_code == 200
        assert resp.headers.get("HX-Redirect") == "/"
