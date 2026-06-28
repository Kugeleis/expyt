from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response

from app.core.session import SessionStore, WizardSession
from app.datasets.hierarchical import HierarchicalData
from app.datasets.repository import DatasetRepository
from app.plots.base import PlotResult, plot_registry
from app.stats.base import StatMethod, StatResult, stat_registry
from app.stats.properties import build_cluster_aggregates, compute_properties, compute_quick_icc
from app.wizard.router.dependencies import (
    _get_grouped_data,
    get_dataset_repository,
    get_filtered_dataset,
    get_session,
    get_session_store,
    render_step,
)
from app.wizard.steps import WizardStep, reset_to_step, validate_step_transition

router = APIRouter()


def _run_stat_for_column(
    filtered_df: pd.DataFrame,
    value_col: str,
    method: StatMethod,
    session: WizardSession,
) -> StatResult:
    """Run statistics on a single column (continuous or discrete)."""
    if session.hierarchy is not None:
        unique_vals = set(filtered_df[value_col].dropna().unique())
        is_bin = unique_vals.issubset({0, 1}) and len(unique_vals) > 0
        is_num = pd.api.types.is_numeric_dtype(filtered_df[value_col]) or pd.api.types.is_bool_dtype(
            filtered_df[value_col]
        )
        metric_kind: Literal["continuous", "binary_proportion", "unsupported"] = (
            "binary_proportion" if is_bin else ("continuous" if is_num else "unsupported")
        )
        if metric_kind == "unsupported":
            raise HTTPException(
                status_code=400,
                detail=f"Column {value_col!r} is not supported in hierarchical mode.",
            )
        excluded_ids = [ex.cluster_id for ex in session.excluded_clusters]
        cluster_agg = build_cluster_aggregates(filtered_df, session.hierarchy, excluded_ids, value_col, metric_kind)
        clean_unit = filtered_df[~filtered_df[session.hierarchy.cluster_col].astype(str).isin(excluded_ids)]
        icc = compute_quick_icc(clean_unit, session.hierarchy.cluster_col, value_col)
        h_data = HierarchicalData(
            unit_df=filtered_df,
            cluster_agg=cluster_agg,
            config=session.hierarchy,
            excluded_clusters=excluded_ids,
            metric=value_col,
            metric_kind=metric_kind,
            icc=icc,
        )
        try:
            res = method.run(h_data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
    else:
        group_data = _get_grouped_data(filtered_df, session.group_column or "", value_col)
        try:
            res = method.run(group_data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None

    res.column_name = value_col
    return res


@router.post("/sessions/{session_id}/update-method")
def update_method(
    session_id: str,
    request: Request,
    selected_method: str | None = Form(default=None),
    selected_discrete_method: str | None = Form(default=None),
    session: WizardSession = Depends(get_session),
    store: SessionStore = Depends(get_session_store),
) -> Response:
    """Update selected statistical methods in session and re-render step 3."""
    session.selected_method = selected_method
    session.selected_discrete_method = selected_discrete_method
    store.save(session)
    return render_step(request, session, store)


@router.post("/sessions/{session_id}/submit-method")
def submit_method(  # noqa: C901
    session_id: str,
    request: Request,
    selected_method: str | None = Form(default=None),
    selected_discrete_method: str | None = Form(default=None),
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
    store: SessionStore = Depends(get_session_store),
) -> Response:
    """Verify methods, execute the evaluations, and transition to Step 4 (Results)."""
    validate_step_transition(session, WizardStep.STAT_METHOD)

    if session.group_column is None or (not session.selected_value_columns and not session.selected_discrete_columns):
        raise HTTPException(status_code=400, detail="Incomplete setup")

    filtered_df = get_filtered_dataset(session, repo)

    if not selected_method and not selected_discrete_method:
        raise HTTPException(status_code=400, detail="At least one method must be selected")

    if selected_method:
        if selected_method not in stat_registry.list_all():
            raise HTTPException(status_code=400, detail=f"Method {selected_method!r} is not registered")
        session.selected_method = selected_method
    else:
        session.selected_method = None

    if selected_discrete_method:
        if selected_discrete_method not in stat_registry.list_all():
            raise HTTPException(status_code=400, detail=f"Method {selected_discrete_method!r} is not registered")
        session.selected_discrete_method = selected_discrete_method
    else:
        session.selected_discrete_method = None

    results: list[StatResult] = []
    if session.selected_value_columns and session.selected_method:
        method = stat_registry.get(session.selected_method)
        for val_col in session.selected_value_columns:
            results.append(_run_stat_for_column(filtered_df, val_col, method, session))

    if session.selected_discrete_columns and session.selected_discrete_method:
        discrete_method = stat_registry.get(session.selected_discrete_method)
        for disc_col in session.selected_discrete_columns:
            results.append(_run_stat_for_column(filtered_df, disc_col, discrete_method, session))

    results.sort(key=lambda r: r.p_value if r.p_value is not None else 1.0)
    session.stat_results = [res.model_dump() for res in results]
    session.current_step = WizardStep.RESULTS.value
    store.save(session)

    return render_step(request, session, store)


@router.post("/sessions/{session_id}/update-sig-limit")
def update_sig_limit(
    session_id: str,
    request: Request,
    plots_sig_filter: float = Form(...),
    session: WizardSession = Depends(get_session),
    store: SessionStore = Depends(get_session_store),
) -> Response:
    """Update plots significance p-value limit dynamically."""
    if not 0.0 <= plots_sig_filter <= 1.0:
        raise HTTPException(status_code=400, detail="plots_sig_filter must be between 0.0 and 1.0")

    safe_plots_sig_filter = float(plots_sig_filter)
    res = render_step(request, session, store, plots_sig_filter=safe_plots_sig_filter)
    res.set_cookie("plots_sig_filter", str(safe_plots_sig_filter))
    return res


@router.post("/sessions/{session_id}/update-sort")
def update_sort(
    session_id: str,
    request: Request,
    field: str,
    session: WizardSession = Depends(get_session),
    store: SessionStore = Depends(get_session_store),
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

    res = render_step(request, session, store, plots_sig_filter=limit, sort_field=safe_field, sort_asc=sort_asc)
    res.set_cookie("sort_field", safe_field)
    res.set_cookie("sort_asc", "true" if sort_asc else "false")
    return res


@router.post("/sessions/{session_id}/submit-results")
def submit_results(
    session_id: str,
    request: Request,
    session: WizardSession = Depends(get_session),
    store: SessionStore = Depends(get_session_store),
) -> Response:
    """Submit the statistical evaluation and transition to Step 5 (Visualizations)."""
    validate_step_transition(session, WizardStep.RESULTS)
    session.current_step = WizardStep.PLOT_SELECTION.value
    store.save(session)
    return render_step(request, session, store)


@router.post("/sessions/{session_id}/generate-plots")
def generate_plots_htmx(
    session_id: str,
    request: Request,
    selected_plots: list[str] = Form(default=[]),
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
    store: SessionStore = Depends(get_session_store),
) -> Response:
    """Execute plot generation via matplotlib backend and render Step 5."""
    validate_step_transition(session, WizardStep.PLOT_SELECTION)

    if session.group_column is None or (not session.selected_value_columns and not session.selected_discrete_columns):
        raise HTTPException(status_code=400, detail="Incomplete setup")

    filtered_df = get_filtered_dataset(session, repo)

    limit = float(request.cookies.get("plots_sig_filter", 0.05))
    if session.stat_results:
        top_columns = [
            res["column_name"]
            for res in session.stat_results
            if "column_name" in res and res.get("p_value") is not None and res["p_value"] <= limit
        ]
    else:
        top_columns = []

    plot_results: list[PlotResult] = []

    for value_col in top_columns:
        props = compute_properties(session, filtered_df, value_col)
        applicable = plot_registry.get_applicable(props)

        for name in selected_plots:
            if name not in applicable:
                raise HTTPException(
                    status_code=400,
                    detail=f"Plot {name!r} not applicable for column {value_col!r}",
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

    session.selected_plots = selected_plots
    session.top_n_columns = len(top_columns)
    session.plot_results = [p.model_dump() for p in plot_results]
    session.current_step = WizardStep.PLOT_SELECTION.value
    store.save(session)

    return render_step(request, session, store)


@router.post("/sessions/{session_id}/submit-plots")
def submit_plots(
    session_id: str,
    request: Request,
    session: WizardSession = Depends(get_session),
    store: SessionStore = Depends(get_session_store),
) -> Response:
    """Submit visualizations and transition to Step 6 (Export)."""
    validate_step_transition(session, WizardStep.PLOT_SELECTION)
    session.current_step = WizardStep.EXPORT.value
    store.save(session)
    return render_step(request, session, store)


@router.post("/sessions/{session_id}/navigate")
def navigate(
    session_id: str,
    direction: str,
    request: Request,
    session: WizardSession = Depends(get_session),
    store: SessionStore = Depends(get_session_store),
) -> Response:
    """General Back / Next navigation handler."""
    steps_list = list(WizardStep)
    current_idx = steps_list.index(WizardStep(session.current_step))

    if direction == "back":
        if current_idx == 0:
            raise HTTPException(status_code=400, detail="Cannot navigate back from first step")
        target = steps_list[current_idx - 1]
        reset_to_step(session, target)
    else:
        if current_idx == len(steps_list) - 1:
            raise HTTPException(status_code=400, detail="Cannot navigate forward from last step")
        target = steps_list[current_idx + 1]
        validate_step_transition(session, target)
        session.current_step = target.value

    store.save(session)
    return render_step(request, session, store)
