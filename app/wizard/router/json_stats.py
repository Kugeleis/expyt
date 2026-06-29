"""JSON routers for preprocessing filters, statistical runs, and visualizations."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response

from app.core.session import WizardSession
from app.datasets.repository import DatasetRepository
from app.exporters.base import exporter_registry
from app.plots.base import PlotResult, plot_registry
from app.stats.base import StatResult, stat_registry
from app.stats.properties import compute_properties, compute_properties_for_columns
from app.wizard.router.dependencies import get_dataset_repository, get_filtered_dataset, get_session, get_wizard_service
from app.wizard.router.guards import require_step
from app.wizard.schemas import ExportRequest, FiltersConfigRequest, MethodSelectionRequest, PlotSelectionRequest
from app.wizard.service import WizardService
from app.wizard.steps import WizardStep

router = APIRouter()

_stat_method_guard = require_step(WizardStep.STAT_METHOD)
_plot_selection_guard = require_step(WizardStep.PLOT_SELECTION)
_export_guard = require_step(WizardStep.EXPORT)


@router.post("/sessions/{session_id}/filters", response_model=WizardSession)
def configure_filters(
    session_id: str,
    req: FiltersConfigRequest,
    service: WizardService = Depends(get_wizard_service),
) -> WizardSession:
    """JSON compatibility route: Configure and apply preprocessing filters."""
    filter_configs = [f.model_dump() for f in req.filters_config]
    return service.configure_filters(session_id, filter_configs)


@router.get("/sessions/{session_id}/methods", response_model=list[dict[str, str]])
def list_applicable_methods(
    session: WizardSession = Depends(_stat_method_guard),
    repo: DatasetRepository = Depends(get_dataset_repository),
) -> list[dict[str, str]]:
    """JSON compatibility route: List statistical methods applicable."""

    if session.group_column is None or (not session.selected_value_columns and not session.selected_discrete_columns):
        raise HTTPException(status_code=400, detail="Incomplete setup")

    filtered_df = get_filtered_dataset(session, repo)

    results = []
    if session.selected_value_columns:
        props_map = compute_properties_for_columns(session, filtered_df, session.selected_value_columns)
        applicable = stat_registry.get_applicable_intersect(props_map)
        for name, inst in applicable.items():
            results.append(
                {
                    "name": name,
                    "description": inst.description,
                    "variable_type": "continuous",
                }
            )

    if session.selected_discrete_columns:
        props_map_discrete = compute_properties_for_columns(session, filtered_df, session.selected_discrete_columns)
        applicable_discrete = stat_registry.get_applicable_intersect(props_map_discrete)
        for name, inst in applicable_discrete.items():
            results.append(
                {
                    "name": name,
                    "description": inst.description,
                    "variable_type": "discrete",
                }
            )

    return results


@router.post("/sessions/{session_id}/method", response_model=WizardSession)
def select_method(
    session_id: str,
    req: MethodSelectionRequest,
    service: WizardService = Depends(get_wizard_service),
) -> WizardSession:
    """JSON compatibility route: Select a statistical method."""
    service.update_method(session_id, req.selected_method, req.selected_discrete_method)
    return service.go_to_step(session_id, WizardStep.RESULTS.value)


@router.get("/sessions/{session_id}/results", response_model=list[StatResult])
def run_statistical_evaluation(
    session_id: str,
    session: WizardSession = Depends(get_session),
    service: WizardService = Depends(get_wizard_service),
) -> list[StatResult]:
    """JSON compatibility route: Run statistical evaluations."""
    updated_session = service.submit_method(
        session_id=session_id,
        selected_method=session.selected_method,
        selected_discrete_method=session.selected_discrete_method,
    )
    # Move to the next step
    updated_session = service.go_to_step(session_id, WizardStep.PLOT_SELECTION.value)
    return [StatResult.model_validate(res) for res in updated_session.stat_results]


@router.get("/sessions/{session_id}/plots", response_model=list[dict[str, str]])
def list_applicable_plots(
    session: WizardSession = Depends(_plot_selection_guard),
    repo: DatasetRepository = Depends(get_dataset_repository),
) -> list[dict[str, str]]:
    """JSON compatibility route: List applicable plot generators."""

    if session.group_column is None or (not session.selected_value_columns and not session.selected_discrete_columns):
        raise HTTPException(status_code=400, detail="Incomplete setup")

    filtered_df = get_filtered_dataset(session, repo)

    if session.selected_value_columns:
        props = compute_properties(session, filtered_df, session.selected_value_columns[0])
        applicable = plot_registry.get_applicable(props)
        return [{"name": name, "description": inst.description} for name, inst in applicable.items()]
    return []


@router.post("/sessions/{session_id}/plots", response_model=WizardSession)
def generate_plots_json(
    session_id: str,
    req: PlotSelectionRequest,
    service: WizardService = Depends(get_wizard_service),
) -> WizardSession:
    """JSON compatibility route: Generate visualizations."""
    service.generate_plots(
        session_id=session_id,
        selected_plots=req.selected_plots,
        plots_sig_filter=0.05,
        top_n_columns=req.top_n_columns,
    )
    return service.go_to_step(session_id, WizardStep.EXPORT.value)


@router.post("/sessions/{session_id}/export")
def export_results_json(
    session_id: str,
    req: ExportRequest,
    session: WizardSession = Depends(_export_guard),
    repo: DatasetRepository = Depends(get_dataset_repository),
    service: WizardService = Depends(get_wizard_service),
) -> Response:
    """JSON compatibility route: Export the report."""

    try:
        exporter = exporter_registry.get(req.export_format)
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Exporter {req.export_format!r} is not registered",
        ) from None

    if session.group_column is None:
        raise HTTPException(status_code=400, detail="Incomplete setup")

    filtered_df = get_filtered_dataset(session, repo)

    stat_results = [StatResult.model_validate(res) for res in session.stat_results]
    plots = [PlotResult.model_validate(p) for p in session.plot_results]

    export_res = exporter.export(stat_results, plots, filtered_df)

    # Save export choice in service
    session.export_format = req.export_format
    service.store.save(session)

    return Response(
        content=export_res.content,
        media_type=export_res.content_type,
        headers={"Content-Disposition": f"attachment; filename={export_res.filename}"},
    )
