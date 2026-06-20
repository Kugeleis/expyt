"""End-to-end tests for the evaluation wizard API.

These tests run the actual uvicorn server in a background thread and perform
real HTTP requests to exercise the API endpoints.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Generator

import httpx
import pandas as pd
import pytest
import uvicorn

from app.main import app


def get_free_port() -> int:
    """Get a free port on localhost.

    Returns:
        An available port number.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


@pytest.fixture(scope="module")
def e2e_server(tmp_path_factory: pytest.TempPathFactory) -> Generator[str, None, None]:
    """Start uvicorn server in a background thread with an isolated dataset dir.

    Yields:
        The base URL of the running FastAPI server.
    """
    # Create temp directory for data files
    tmp_dir = tmp_path_factory.mktemp("e2e_data")

    # Save a test dataset in it
    df = pd.DataFrame(
        {
            "group": ["A"] * 5 + ["B"] * 5,
            "value": [10.0, 10.5, 11.0, 10.2, 9.8, 12.0, 12.5, 13.0, 12.2, 11.8],
        }
    )
    df.to_csv(tmp_dir / "e2e_normal_data.csv", index=False)

    # Configure app to use this temp data directory via env var
    original_env = os.environ.get("EXPYT_DATA_DIR")
    os.environ["EXPYT_DATA_DIR"] = str(tmp_dir)

    # Find an open port
    port = get_free_port()

    # Create server instance
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    # Run in background daemon thread
    thread = threading.Thread(target=server.run)
    thread.daemon = True
    thread.start()

    # Wait for the server to start accepting connections
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.1)
    else:
        raise RuntimeError("E2E test uvicorn server failed to start")

    yield base_url

    # Shutdown the uvicorn server
    server.should_exit = True
    thread.join(timeout=5)

    # Restore environment variable
    if original_env is not None:
        os.environ["EXPYT_DATA_DIR"] = original_env
    else:
        os.environ.pop("EXPYT_DATA_DIR", None)


def test_e2e_wizard_flow(e2e_server: str) -> None:
    """Execute a complete wizard flow end-to-end over HTTP.

    Args:
        e2e_server: The base URL of the running API server.
    """
    with httpx.Client(base_url=e2e_server) as client:
        # Upload a dataset first
        csv_content = (
            b"group,value\nA,10.0\nA,10.5\nA,11.0\nA,10.2\nA,9.8\n"
            b"B,12.0\nB,12.5\nB,13.0\nB,12.2\nB,11.8\n"
        )
        files = {"file": ("uploaded_data.csv", csv_content, "text/csv")}
        resp = client.post("/wizard/upload", files=files)
        assert resp.status_code == 200
        assert resp.json()["id"] == "uploaded_data"

        # Create Session
        resp = client.post("/wizard/sessions")
        assert resp.status_code == 200
        session = resp.json()
        session_id = session["session_id"]
        assert session["current_step"] == "dataset_selection"

        # Step 1: Select dataset
        resp = client.post(
            f"/wizard/sessions/{session_id}/dataset",
            json={
                "dataset_id": "uploaded_data",
                "group_column": "group",
                "selected_value_columns": [],
            },
        )
        if resp.status_code != 200:
            print("RESPONSE BODY:", resp.json())
        assert resp.status_code == 200
        assert resp.json()["dataset_id"] == "uploaded_data"

        # Step 2: Configure filters
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

        # List applicable statistical methods
        resp = client.get(f"/wizard/sessions/{session_id}/methods")
        assert resp.status_code == 200
        methods = [m["name"] for m in resp.json()]
        assert "ttest_ind" in methods

        # Step 3: Choose method
        resp = client.post(
            f"/wizard/sessions/{session_id}/method",
            json={"selected_method": "ttest_ind"},
        )
        assert resp.status_code == 200

        # Step 4: Run statistical evaluation
        resp = client.get(f"/wizard/sessions/{session_id}/results")
        assert resp.status_code == 200
        result = resp.json()
        assert result[0]["method_name"] == "ttest_ind"
        assert "p_value" in result[0]

        # List applicable plots
        resp = client.get(f"/wizard/sessions/{session_id}/plots")
        assert resp.status_code == 200
        plots = [p["name"] for p in resp.json()]
        assert "boxplot" in plots

        # Step 5: Generate plots
        resp = client.post(
            f"/wizard/sessions/{session_id}/plots",
            json={"selected_plots": ["boxplot"]},
        )
        assert resp.status_code == 200
        assert len(resp.json()["plot_results"]) == 1

        # Step 6: Export results as JSON
        resp = client.post(
            f"/wizard/sessions/{session_id}/export",
            json={"export_format": "json"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/json"
        data = resp.json()
        assert "statistical_results" in data
        assert data["statistical_results"][0]["method_name"] == "ttest_ind"
