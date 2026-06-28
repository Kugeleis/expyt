from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response

from app.core.session import ClusterExclusion, SessionStore, WizardSession
from app.datasets.repository import DatasetRepository
from app.exporters.base import exporter_registry
from app.filters.base import apply_filter_pipeline
from app.plots.base import PlotResult, plot_registry
from app.stats.base import StatResult, stat_registry
from app.stats.properties import compute_properties, compute_properties_for_columns
from app.wizard.router.dependencies import (
    get_dataset_repository,
    get_filtered_dataset,
    get_session,
    get_session_store,
)
from app.wizard.router.htmx_results import _run_stat_for_column
from app.wizard.schemas import (
    ExportRequest,
    FiltersConfigRequest,
    MethodSelectionRequest,
    PlotSelectionRequest,
)
from app.wizard.steps import WizardStep, validate_step_transition

router = APIRouter()


@router.post("/sessions/{session_id}/filters", response_model=WizardSession)
def configure_filters(
    req: FiltersConfigRequest,
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
    store: SessionStore = Depends(get_session_store),
) -> WizardSession:
    """JSON compatibility route: Configure and apply preprocessing filters."""
    validate_step_transition(session, WizardStep.FILTERS)

    if session.dataset_id is None:
        raise HTTPException(status_code=400, detail="Dataset not selected")

    try:
        df = repo.load_dataset(session.dataset_id)
    except KeyError:
        raise HTTPException(status_code=400, detail="Dataset missing") from None

    filter_configs = [f.model_dump() for f in req.filters_config]
    try:
        apply_filter_pipeline(df, filter_configs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    session.filters_config = filter_configs

    excluded = []
    for f in filter_configs:
        if f.get("name") == "cluster_exclusion":
            exclusions_list = f.get("params", {}).get("exclusions", [])
            for item in exclusions_list:
                excluded.append(ClusterExclusion(cluster_id=str(item["cluster_id"]), reason=str(item["reason"])))
    session.excluded_clusters = excluded

    session.current_step = WizardStep.STAT_METHOD.value
    store.save(session)
    return session


@router.get("/sessions/{session_id}/methods", response_model=list[dict[str, str]])
def list_applicable_methods(
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
) -> list[dict[str, str]]:
    """JSON compatibility route: List statistical methods applicable."""
    validate_step_transition(session, WizardStep.STAT_METHOD)

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
    req: MethodSelectionRequest,
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
    store: SessionStore = Depends(get_session_store),
) -> WizardSession:
    """JSON compatibility route: Select a statistical method."""
    validate_step_transition(session, WizardStep.STAT_METHOD)

    if session.group_column is None or (not session.selected_value_columns and not session.selected_discrete_columns):
        raise HTTPException(status_code=400, detail="Incomplete setup")

    get_filtered_dataset(session, repo)

    if session.selected_value_columns:
        if not req.selected_method:
            raise HTTPException(status_code=400, detail="Method for continuous columns must be selected")
        if req.selected_method not in stat_registry.list_all():
            raise HTTPException(status_code=400, detail=f"Method {req.selected_method!r} is not registered")
        session.selected_method = req.selected_method
    else:
        session.selected_method = None

    if session.selected_discrete_columns:
        if not req.selected_discrete_method:
            raise HTTPException(status_code=400, detail="Method for discrete columns must be selected")
        if req.selected_discrete_method not in stat_registry.list_all():
            raise HTTPException(status_code=400, detail=f"Method {req.selected_discrete_method!r} is not registered")
        session.selected_discrete_method = req.selected_discrete_method
    else:
        session.selected_discrete_method = None

    session.current_step = WizardStep.RESULTS.value
    store.save(session)
    return session


@router.get("/sessions/{session_id}/results", response_model=list[StatResult])
def run_statistical_evaluation(
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
    store: SessionStore = Depends(get_session_store),
) -> list[StatResult]:
    """JSON compatibility route: Run statistical evaluations."""
    validate_step_transition(session, WizardStep.RESULTS)

    if session.group_column is None or (not session.selected_value_columns and not session.selected_discrete_columns):
        raise HTTPException(status_code=400, detail="Incomplete setup")

    filtered_df = get_filtered_dataset(session, repo)
    results: list[StatResult] = []

    if session.selected_value_columns:
        if session.selected_method is None:
            raise HTTPException(status_code=400, detail="Continuous method not selected")
        method = stat_registry.get(session.selected_method)
        for value_col in session.selected_value_columns:
            results.append(_run_stat_for_column(filtered_df, value_col, method, session))

    if session.selected_discrete_columns:
        if session.selected_discrete_method is None:
            raise HTTPException(status_code=400, detail="Discrete method not selected")
        discrete_method = stat_registry.get(session.selected_discrete_method)
        for value_col in session.selected_discrete_columns:
            results.append(_run_stat_for_column(filtered_df, value_col, discrete_method, session))

    results.sort(key=lambda r: r.p_value if r.p_value is not None else 1.0)

    session.stat_results = [res.model_dump() for res in results]
    session.current_step = WizardStep.PLOT_SELECTION.value
    store.save(session)
    return results


@router.get("/sessions/{session_id}/plots", response_model=list[dict[str, str]])
def list_applicable_plots(
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
) -> list[dict[str, str]]:
    """JSON compatibility route: List applicable plot generators."""
    validate_step_transition(session, WizardStep.PLOT_SELECTION)

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
    req: PlotSelectionRequest,
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
    store: SessionStore = Depends(get_session_store),
) -> WizardSession:
    """JSON compatibility route: Generate visualizations."""
    validate_step_transition(session, WizardStep.PLOT_SELECTION)

    if session.group_column is None or (not session.selected_value_columns and not session.selected_discrete_columns):
        raise HTTPException(status_code=400, detail="Incomplete setup")

    filtered_df = get_filtered_dataset(session, repo)

    if session.stat_results:
        ranked_cols = [res["column_name"] for res in session.stat_results if "column_name" in res]
        top_columns = [col for col in ranked_cols if col and col in session.selected_value_columns][: req.top_n_columns]
    else:
        top_columns = session.selected_value_columns[: req.top_n_columns]

    plot_results: list[PlotResult] = []

    for value_col in top_columns:
        props = compute_properties(session, filtered_df, value_col)
        applicable = plot_registry.get_applicable(props)

        for name in req.selected_plots:
            if name not in applicable:
                raise HTTPException(
                    status_code=400,
                    detail=f"Plot generator {name!r} is not applicable for column {value_col!r}",
                )
            generator = plot_registry.get(name)
            import inspect

            sig = inspect.signature(generator.generate)
            kwargs: dict[str, Any] = {}
            is_hier = "hierarchy" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if is_hier:
                kwargs["hierarchy"] = session.hierarchy
                kwargs["excluded_clusters"] = [ex.cluster_id for ex in session.excluded_clusters]

            plot_result = generator.generate(filtered_df, session.group_column or "", value_col, **kwargs)
            plot_result.column_name = value_col
            plot_results.append(plot_result)

    session.selected_plots = req.selected_plots
    session.top_n_columns = req.top_n_columns
    session.plot_results = [p.model_dump() for p in plot_results]
    session.current_step = WizardStep.EXPORT.value
    store.save(session)
    return session


@router.post("/sessions/{session_id}/export")
def export_results_json(
    req: ExportRequest,
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
    store: SessionStore = Depends(get_session_store),
) -> Response:
    """JSON compatibility route: Export the report."""
    validate_step_transition(session, WizardStep.EXPORT)

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

    session.export_format = req.export_format
    session.current_step = WizardStep.EXPORT.value
    store.save(session)

    return Response(
        content=export_res.content,
        media_type=export_res.content_type,
        headers={"Content-Disposition": f"attachment; filename={export_res.filename}"},
    )
