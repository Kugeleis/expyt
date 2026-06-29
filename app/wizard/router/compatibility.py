"""HTML and JSON compatibility routers for step navigation."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from app.core.session import WizardSession
from app.wizard.router.dependencies import get_session, get_session_store, get_wizard_service, render_step
from app.wizard.service import WizardService
from app.wizard.steps import WizardStep

router = APIRouter()


async def _handle_compatibility_form(
    session_id: str,
    current: WizardStep,
    is_forward: bool,
    request: Request,
    service: WizardService,
) -> None:
    """Helper to process and submit form configurations during compatibility step switching."""
    session = service.get_session(session_id)
    form_data = await request.form()

    if current == WizardStep.DATASET_SELECTION and session.dataset_id:
        group_column_val = form_data.get("group_column")
        if group_column_val:
            service.submit_dataset_config(
                session_id=session_id,
                group_column=str(group_column_val),
                selected_groups=[str(x) for x in form_data.getlist("selected_groups")],
                selected_value_columns=[str(x) for x in form_data.getlist("selected_value_columns")],
                selected_discrete_columns=[str(x) for x in form_data.getlist("selected_discrete_columns")],
                cluster_col=str(form_data.get("cluster_col")) if form_data.get("cluster_col") is not None else None,
                selected_clusters=[str(x) for x in form_data.getlist("selected_clusters")],
                unit_col=str(form_data.get("unit_col")) if form_data.get("unit_col") else None,
                x_col=str(form_data.get("x_col")) if form_data.get("x_col") else None,
                y_col=str(form_data.get("y_col")) if form_data.get("y_col") else None,
            )

    elif current == WizardStep.STAT_METHOD:
        selected_method_val = form_data.get("selected_method")
        selected_discrete_method_val = form_data.get("selected_discrete_method")

        selected_method = str(selected_method_val) if selected_method_val else None
        selected_discrete_method = str(selected_discrete_method_val) if selected_discrete_method_val else None

        if is_forward:
            service.submit_method(session_id, selected_method, selected_discrete_method)
        else:
            service.update_method(session_id, selected_method, selected_discrete_method)

    elif current == WizardStep.PLOT_SELECTION:
        selected_plots = [str(x) for x in form_data.getlist("selected_plots")]
        limit = float(request.cookies.get("plots_sig_filter", 0.05))
        if is_forward:
            service.generate_plots(session_id, selected_plots, limit)
        else:
            session.selected_plots = selected_plots
            service.store.save(session)


@router.post("/sessions/{session_id}/go-to/{step}")
async def go_to_step_compatibility(
    session_id: str,
    step: str,
    request: Request,
    session: WizardSession = Depends(get_session),
    service: WizardService = Depends(get_wizard_service),
) -> Response:
    """HTML and JSON compatibility route: Navigate back to a completed wizard step."""
    try:
        target = WizardStep(step)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown wizard step {step!r}",
        ) from None

    current = WizardStep(session.current_step)

    steps_list = list(WizardStep)
    current_idx = steps_list.index(current)
    target_idx = steps_list.index(target)
    is_forward = target_idx > current_idx

    if "hx-request" in request.headers:
        await _handle_compatibility_form(session_id, current, is_forward, request, service)

    # Perform final navigation state transition
    updated_session = service.go_to_step(session_id, step)

    if "hx-request" in request.headers:
        store = get_session_store(request)
        return render_step(request, updated_session, store)
    else:
        import json

        return JSONResponse(content=json.loads(updated_session.model_dump_json()))
