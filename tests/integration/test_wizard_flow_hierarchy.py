"""Hierarchical configuration integration tests for the wizard flow."""

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
    # numeric cluster
    df_cluster = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "cluster": [1.0] * 5 + [2.0] * 5 + [3.0] * 5 + [4.0] * 5,
            "value": np.random.normal(loc=10.0, scale=1.0, size=20),
        }
    )
    df_cluster.to_csv(tmp_path / "uploaded_numeric_cluster.csv", index=False)

    # numeric group hierarchical
    df_group = pd.DataFrame(
        {
            "group": [1.0] * 10 + [2.0] * 10,
            "cluster": ["X"] * 5 + ["Y"] * 5 + ["X"] * 5 + ["Y"] * 5,
            "value": np.random.normal(loc=10.0, scale=1.0, size=20),
        }
    )
    df_group.to_csv(tmp_path / "uploaded_hier_numeric.csv", index=False)

    # numeric cluster hierarchical
    df_hier_cluster = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "cluster": [1.0] * 5 + [2.0] * 5 + [1.0] * 5 + [2.0] * 5,
            "value": np.random.normal(loc=10.0, scale=1.0, size=20),
        }
    )
    df_hier_cluster.to_csv(tmp_path / "uploaded_hier_numeric_cluster.csv", index=False)

    # defaults test
    df_defaults = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "cluster": ["X"] * 5 + ["Y"] * 5 + ["X"] * 5 + ["Y"] * 5,
            "value": np.random.normal(loc=10.0, scale=1.0, size=20),
        }
    )
    df_defaults.to_csv(tmp_path / "defaults_test.csv", index=False)

    # same col test
    df_same = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "cluster": ["X"] * 5 + ["Y"] * 5 + ["X"] * 5 + ["Y"] * 5,
            "value": np.random.normal(loc=10.0, scale=1.0, size=20),
        }
    )
    df_same.to_csv(tmp_path / "uploaded_same_col.csv", index=False)
    df_same.to_csv(tmp_path / "uploaded_hier_same.csv", index=False)

    # preselect test
    df_preselect = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10,
            "numeric_val": np.random.normal(loc=10.0, scale=1.0, size=20),
            "categorical_val": ["Yes", "No"] * 10,
        }
    )
    df_preselect.to_csv(tmp_path / "preselect_test.csv", index=False)

    return tmp_path


@pytest.fixture
def client(test_data_dir: Path) -> Generator[TestClient, None, None]:
    """TestClient with overridden dataset repository dependency."""
    repo = CsvDatasetRepository(test_data_dir)
    app.dependency_overrides[get_dataset_repository] = lambda: repo
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_select_cluster_column_numeric_fails(client: TestClient) -> None:
    """Selecting a numeric column as cluster column (L1) returns 400."""
    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    resp = client.post(f"/wizard/sessions/{session_id}/toggle-hierarchy?enabled=true", headers={"HX-Request": "true"})
    assert resp.status_code == 200

    resp = client.post(
        f"/wizard/sessions/{session_id}/select-dataset-id",
        data={"dataset_id": "uploaded_numeric_cluster"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200

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
    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    resp = client.post(
        f"/wizard/sessions/{session_id}/dataset",
        json={
            "dataset_id": "uploaded_hier_numeric",
            "group_column": "cluster",
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
    assert "Group column 'group' must be discrete/categorical, but it is numeric." in resp.json()["detail"]

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
    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    resp = client.post(
        f"/wizard/sessions/{session_id}/select-dataset-id",
        data={"dataset_id": "defaults_test"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200

    resp = client.post(
        f"/wizard/sessions/{session_id}/update-group-col",
        data={"group_column": "group"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200

    resp_get = client.get(f"/wizard/sessions/{session_id}")
    assert resp_get.json()["selected_groups"] == ["A", "B"]

    resp = client.post(f"/wizard/sessions/{session_id}/toggle-hierarchy?enabled=true", headers={"HX-Request": "true"})
    assert resp.status_code == 200

    resp = client.post(
        f"/wizard/sessions/{session_id}/update-cluster-col",
        data={"cluster_col": "cluster"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200

    resp_get = client.get(f"/wizard/sessions/{session_id}")
    assert resp_get.json()["hierarchy"]["selected_clusters"] == ["X", "Y"]


def test_select_cluster_column_same_as_group_fails(client: TestClient) -> None:
    """Selecting the same column for both group and cluster returns 400."""
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
    resp = client.post("/wizard/sessions")
    session_id = resp.json()["session_id"]

    resp = client.post(
        f"/wizard/sessions/{session_id}/select-dataset-id",
        data={"dataset_id": "preselect_test"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200

    resp = client.post(
        f"/wizard/sessions/{session_id}/update-group-col",
        data={"group_column": "group"},
        headers={"HX-Request": "true"},
    )
    assert resp.status_code == 200

    resp_get = client.get(f"/wizard/sessions/{session_id}")
    assert resp_get.json()["selected_value_columns"] == ["numeric_val"]
    assert resp_get.json()["selected_discrete_columns"] == ["categorical_val"]
