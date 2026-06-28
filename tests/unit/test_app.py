"""Smoke test for the FastAPI application."""

from fastapi.testclient import TestClient


def test_app_starts(client: TestClient) -> None:
    """The application should respond to OpenAPI docs endpoint."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert data["info"]["title"] == "ExPyRi — Experiment Evaluation Wizard"


def test_root_serves_homepage(client: TestClient) -> None:
    """The root URL should serve the wizard homepage HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "ExPyRi" in response.text
    assert "Experiment Evaluation Wizard" in response.text


def test_app_version_fallback() -> None:
    """Test fallback version if package is not installed."""
    import importlib.metadata
    import sys
    from unittest.mock import patch

    original_main = sys.modules.get("app.main")
    try:
        with patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError):
            if "app.main" in sys.modules:
                del sys.modules["app.main"]
            import app.main

            assert app.main.__version__ == "0.1.0"
    finally:
        if original_main is not None:
            sys.modules["app.main"] = original_main
        else:
            sys.modules.pop("app.main", None)
