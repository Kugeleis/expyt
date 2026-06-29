"""HTMX routers for statistical results, plots, and navigation steps."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response

from app.wizard.router.dependencies import get_session_store, get_wizard_service, render_step
from app.wizard.service import WizardService

router = APIRouter()


@router.post("/sessions/{session_id}/update-method")
def update_method(
    session_id: str,
    request: Request,
    selected_method: str | None = Form(default=None),
    selected_discrete_method: str | None = Form(default=None),
    service: WizardService = Depends(get_wizard_service),
) -> Response:
    """Update selected statistical methods in session and re-render step 3."""
    session = service.update_method(session_id, selected_method, selected_discrete_method)
    store = get_session_store(request)
    return render_step(request, session, store)


@router.post("/sessions/{session_id}/submit-method")
def submit_method(
    session_id: str,
    request: Request,
    selected_method: str | None = Form(default=None),
    selected_discrete_method: str | None = Form(default=None),
    service: WizardService = Depends(get_wizard_service),
) -> Response:
    """Verify methods, execute the evaluations, and transition to Step 4 (Results)."""
    session = service.submit_method(session_id, selected_method, selected_discrete_method)
    store = get_session_store(request)
    return render_step(request, session, store)


@router.post("/sessions/{session_id}/update-sig-limit")
def update_sig_limit(
    session_id: str,
    request: Request,
    plots_sig_filter: float = Form(...),
    service: WizardService = Depends(get_wizard_service),
) -> Response:
    """Update plots significance p-value limit dynamically."""
    if not 0.0 <= plots_sig_filter <= 1.0:
        raise HTTPException(status_code=400, detail="plots_sig_filter must be between 0.0 and 1.0")

    session = service.get_session(session_id)
    store = get_session_store(request)
    safe_plots_sig_filter = float(plots_sig_filter)
    res = render_step(request, session, store, plots_sig_filter=safe_plots_sig_filter)
    res.set_cookie("plots_sig_filter", str(safe_plots_sig_filter))
    return res


@router.post("/sessions/{session_id}/update-sort")
def update_sort(
    session_id: str,
    request: Request,
    field: str,
    service: WizardService = Depends(get_wizard_service),
) -> Response:
    """Sort results table by clicked column header."""
    available_sort_fields = {"column_name", "method_name", "test_statistic", "p_value", "effect_size", "icc", "power"}
    safe_field = field if field in available_sort_fields else "column_name"

    current_sort_field = request.cookies.get("sort_field", "column_name")
    current_sort_asc_str = request.cookies.get("sort_asc", "true")

    sort_asc = True
    if current_sort_field == safe_field:
        sort_asc = current_sort_asc_str != "true"

    limit = float(request.cookies.get("plots_sig_filter", 0.05))

    session = service.get_session(session_id)
    store = get_session_store(request)
    res = render_step(request, session, store, plots_sig_filter=limit, sort_field=safe_field, sort_asc=sort_asc)
    res.set_cookie("sort_field", safe_field)
    res.set_cookie("sort_asc", "true" if sort_asc else "false")
    return res


@router.post("/sessions/{session_id}/submit-results")
def submit_results(
    session_id: str,
    request: Request,
    service: WizardService = Depends(get_wizard_service),
) -> Response:
    """Submit the statistical evaluation and transition to Step 5 (Visualizations)."""
    session = service.submit_results(session_id)
    store = get_session_store(request)
    return render_step(request, session, store)


@router.post("/sessions/{session_id}/generate-plots")
def generate_plots_htmx(
    session_id: str,
    request: Request,
    selected_plots: list[str] = Form(default=[]),
    service: WizardService = Depends(get_wizard_service),
) -> Response:
    """Execute plot generation via matplotlib backend and render Step 5."""
    limit = float(request.cookies.get("plots_sig_filter", 0.05))
    session = service.generate_plots(session_id, selected_plots, limit)
    store = get_session_store(request)
    return render_step(request, session, store)


@router.post("/sessions/{session_id}/submit-plots")
def submit_plots(
    session_id: str,
    request: Request,
    service: WizardService = Depends(get_wizard_service),
) -> Response:
    """Submit visualizations and transition to Step 6 (Export)."""
    session = service.submit_plots(session_id)
    store = get_session_store(request)
    return render_step(request, session, store)


@router.post("/sessions/{session_id}/navigate")
def navigate(
    session_id: str,
    direction: str,
    request: Request,
    service: WizardService = Depends(get_wizard_service),
) -> Response:
    """General Back / Next navigation handler."""
    session = service.navigate(session_id, direction)
    store = get_session_store(request)
    return render_step(request, session, store)
