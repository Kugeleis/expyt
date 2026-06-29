"""HTMX routers for Step 1: Dataset selection and configuration."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from app.wizard.router.dependencies import get_session_store, get_wizard_service, render_step
from app.wizard.service import WizardService

router = APIRouter()


@router.post("/sessions/{session_id}/restart")
def restart_session(
    session_id: str,
    request: Request,
    service: WizardService = Depends(get_wizard_service),
) -> Response:
    """Restart and destroy the current session by redirecting to root or returning JSON."""
    session = service.restart_session(session_id)

    if "hx-request" in request.headers:
        return HTMLResponse(content="", headers={"HX-Redirect": "/"})

    accept = request.headers.get("accept", "")
    if "application/json" in accept or "json" in accept:
        import json

        return JSONResponse(content=json.loads(session.model_dump_json()))

    return RedirectResponse(url="/", status_code=303)


@router.post("/sessions/{session_id}/select-dataset-id")
def select_dataset_id(
    session_id: str,
    request: Request,
    dataset_id: str = Form(...),
    service: WizardService = Depends(get_wizard_service),
) -> Response:
    """Select an existing dataset from the repository list."""
    session = service.select_dataset_id(session_id, dataset_id if dataset_id else None)
    store = get_session_store(request)
    return render_step(request, session, store)


@router.post("/sessions/{session_id}/toggle-hierarchy")
def toggle_hierarchy_htmx(
    session_id: str,
    request: Request,
    enabled: str,
    service: WizardService = Depends(get_wizard_service),
) -> Response:
    """Toggle hierarchical configuration mode (HTMX view)."""
    session = service.toggle_hierarchy(session_id, enabled == "true")
    store = get_session_store(request)
    return render_step(request, session, store)


@router.post("/sessions/{session_id}/update-group-col")
def update_group_col(
    session_id: str,
    request: Request,
    group_column: str = Form(...),
    service: WizardService = Depends(get_wizard_service),
) -> Response:
    """Update active group column selection and reload subgroups."""
    session = service.update_group_col(session_id, group_column if group_column else None)
    store = get_session_store(request)
    return render_step(request, session, store)


@router.post("/sessions/{session_id}/update-cluster-col")
def update_cluster_col(
    session_id: str,
    request: Request,
    cluster_col: str = Form(...),
    service: WizardService = Depends(get_wizard_service),
) -> Response:
    """Update active cluster selection and reload clusters."""
    session = service.update_cluster_col(session_id, cluster_col if cluster_col else None)
    store = get_session_store(request)
    return render_step(request, session, store)


@router.post("/sessions/{session_id}/submit-dataset-config")
def submit_dataset_config(
    session_id: str,
    request: Request,
    group_column: str = Form(...),
    selected_groups: list[str] = Form(default=[]),
    selected_value_columns: list[str] = Form(default=[]),
    selected_discrete_columns: list[str] = Form(default=[]),
    cluster_col: str | None = Form(default=None),
    selected_clusters: list[str] = Form(default=[]),
    unit_col: str | None = Form(default=None),
    x_col: str | None = Form(default=None),
    y_col: str | None = Form(default=None),
    service: WizardService = Depends(get_wizard_service),
) -> Response:
    """Submit the dataset setup choices and transition to Step 2 (Filters)."""
    session = service.submit_dataset_config(
        session_id=session_id,
        group_column=group_column,
        selected_groups=selected_groups,
        selected_value_columns=selected_value_columns,
        selected_discrete_columns=selected_discrete_columns,
        cluster_col=cluster_col,
        selected_clusters=selected_clusters,
        unit_col=unit_col,
        x_col=x_col,
        y_col=y_col,
    )
    store = get_session_store(request)
    return render_step(request, session, store)
