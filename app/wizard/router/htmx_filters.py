"""HTMX routers for Step 2: Preprocessing filters."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request, Response
from fastapi.responses import HTMLResponse

from app.wizard.router.dependencies import get_session_store, get_wizard_service, render_step, templates
from app.wizard.service import WizardService

router = APIRouter()


@router.post("/sessions/{session_id}/update-filter-fields")
def update_filter_fields(
    session_id: str,
    request: Request,
    filter_type: str = Form(...),
) -> Response:
    """Render the filter fields sub-partial based on filter type."""
    context = {"request": request}
    if filter_type == "category_filter":
        tpl = "partials/filter_fields_category.html"
    elif filter_type == "cluster_exclusion":
        tpl = "partials/filter_fields_cluster.html"
    else:
        tpl = "partials/filter_fields_numeric.html"

    html = templates.get_template(tpl).render(context)
    return HTMLResponse(content=html)


@router.post("/sessions/{session_id}/add-filter")
def add_filter(
    session_id: str,
    request: Request,
    filter_type: str = Form(...),
    column: str = Form(...),
    min_val: str | None = Form(default=None),
    max_val: str | None = Form(default=None),
    values: str | None = Form(default=None),
    exclude: bool = Form(default=False),
    cluster_id: str | None = Form(default=None),
    reason: str | None = Form(default=None),
    service: WizardService = Depends(get_wizard_service),
) -> Response:
    """Add a new filter and validate it by dry-running the filter pipeline."""
    session = service.add_filter(
        session_id=session_id,
        filter_type=filter_type,
        column=column,
        min_val=min_val,
        max_val=max_val,
        values=values,
        exclude=exclude,
        cluster_id=cluster_id,
        reason=reason,
    )
    store = get_session_store(request)
    return render_step(request, session, store)


@router.delete("/sessions/{session_id}/filters/{index}")
def delete_filter(
    session_id: str,
    index: int,
    request: Request,
    service: WizardService = Depends(get_wizard_service),
) -> Response:
    """Remove a filter by its index position."""
    session = service.delete_filter(session_id, index)
    store = get_session_store(request)
    return render_step(request, session, store)


@router.post("/sessions/{session_id}/submit-filters")
def submit_filters(
    session_id: str,
    request: Request,
    service: WizardService = Depends(get_wizard_service),
) -> Response:
    """Submit the filters and move to Step 3 (Choose Method)."""
    session = service.submit_filters(session_id)
    store = get_session_store(request)
    return render_step(request, session, store)
