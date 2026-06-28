from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from app.core.session import SessionStore, WizardSession
from app.datasets.repository import DatasetRepository
from app.datasets.utils import resolve_selected_discrete_columns, resolve_selected_value_columns
from app.plots.base import plot_registry
from app.stats.base import stat_registry
from app.stats.properties import compute_properties
from app.wizard.router.dependencies import (
    get_dataset_repository,
    get_filtered_dataset,
    get_session,
    get_session_store,
    render_step,
)
from app.wizard.router.htmx_results import _run_stat_for_column
from app.wizard.steps import WizardStep, reset_to_step, validate_step_transition

router = APIRouter()


@router.post("/sessions/{session_id}/go-to/{step}")
async def go_to_step_compatibility(  # noqa: C901
    session_id: str,
    step: str,
    request: Request,
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
    store: SessionStore = Depends(get_session_store),
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
        form_data = await request.form()

        if current == WizardStep.DATASET_SELECTION:
            if session.dataset_id:
                group_column_val = form_data.get("group_column")
                if group_column_val:
                    group_column = str(group_column_val)
                    selected_groups = [str(x) for x in form_data.getlist("selected_groups")]
                    selected_value_columns = [str(x) for x in form_data.getlist("selected_value_columns")]
                    selected_discrete_columns = [str(x) for x in form_data.getlist("selected_discrete_columns")]
                    cluster_col_val = form_data.get("cluster_col")
                    cluster_col = str(cluster_col_val) if cluster_col_val is not None else None
                    selected_clusters = [str(x) for x in form_data.getlist("selected_clusters")]
                    unit_col_val = form_data.get("unit_col")
                    unit_col = str(unit_col_val) if unit_col_val else None
                    x_col_val = form_data.get("x_col")
                    x_col = str(x_col_val) if x_col_val else None
                    y_col_val = form_data.get("y_col")
                    y_col = str(y_col_val) if y_col_val else None

                    if is_forward:
                        try:
                            schema = repo.get_schema(session.dataset_id)
                            df = repo.load_dataset(session.dataset_id)
                        except KeyError:
                            raise HTTPException(status_code=400, detail="Dataset not found") from None

                        group_col_info = next((col for col in schema.columns or [] if col.name == group_column), None)
                        if not group_col_info:
                            raise HTTPException(status_code=400, detail=f"Group column {group_column!r} not found")
                        if group_col_info.is_numeric:
                            raise HTTPException(
                                status_code=400,
                                detail=(
                                    f"Group column {group_column!r} must be discrete/categorical, but it is numeric."
                                ),
                            )

                        if session.hierarchy:
                            if not cluster_col:
                                raise HTTPException(
                                    status_code=400, detail="Cluster column is required in hierarchical mode"
                                )
                            if cluster_col == group_column:
                                raise HTTPException(
                                    status_code=400,
                                    detail="Cluster column must not be the same as the group column.",
                                )
                            cluster_col_info = next(
                                (col for col in schema.columns or [] if col.name == cluster_col), None
                            )
                            if not cluster_col_info:
                                raise HTTPException(status_code=400, detail=f"Cluster column {cluster_col!r} not found")
                            if cluster_col_info.is_numeric:
                                raise HTTPException(
                                    status_code=400,
                                    detail=(
                                        f"Cluster column {cluster_col!r} must be discrete/categorical, "
                                        "but it is numeric."
                                    ),
                                )

                        # Resolve dependent columns
                        if not selected_value_columns and not selected_discrete_columns:
                            selected_value_columns = resolve_selected_value_columns(df, group_column, [])
                            selected_discrete_columns = resolve_selected_discrete_columns(df, group_column, [])
                        else:
                            selected_value_columns = resolve_selected_value_columns(
                                df, group_column, selected_value_columns
                            )
                            selected_discrete_columns = resolve_selected_discrete_columns(
                                df, group_column, selected_discrete_columns
                            )

                        if not selected_value_columns and not selected_discrete_columns:
                            raise HTTPException(
                                status_code=400,
                                detail="Select at least one dependent column to analyze.",
                            )

                    # Save config
                    session.group_column = group_column
                    session.selected_groups = selected_groups
                    session.selected_value_columns = selected_value_columns
                    session.selected_discrete_columns = selected_discrete_columns
                    if session.hierarchy:
                        session.hierarchy.group_col = group_column
                        session.hierarchy.cluster_col = cluster_col or ""
                        session.hierarchy.selected_clusters = selected_clusters
                        session.hierarchy.unit_col = unit_col if unit_col else None
                        session.hierarchy.x_col = x_col if x_col else None
                        session.hierarchy.y_col = y_col if y_col else None

        elif current == WizardStep.STAT_METHOD:
            selected_method_val = form_data.get("selected_method")
            selected_discrete_method_val = form_data.get("selected_discrete_method")

            selected_method = str(selected_method_val) if selected_method_val else None
            selected_discrete_method = str(selected_discrete_method_val) if selected_discrete_method_val else None

            session.selected_method = selected_method
            session.selected_discrete_method = selected_discrete_method

            if is_forward:
                is_incomplete = session.group_column is None or (
                    not session.selected_value_columns and not session.selected_discrete_columns
                )
                if is_incomplete:
                    raise HTTPException(status_code=400, detail="Incomplete setup")
                dataset_id = session.dataset_id
                if not dataset_id:
                    raise HTTPException(status_code=400, detail="Dataset not selected")
                try:
                    df = repo.load_dataset(dataset_id)
                except KeyError:
                    raise HTTPException(status_code=400, detail="Dataset not found") from None
                filtered_df = get_filtered_dataset(session, repo)

                if not selected_method and not selected_discrete_method:
                    raise HTTPException(status_code=400, detail="At least one method must be selected")

                if selected_method and selected_method not in stat_registry.list_all():
                    raise HTTPException(status_code=400, detail=f"Method {selected_method!r} is not registered")
                if selected_discrete_method and selected_discrete_method not in stat_registry.list_all():
                    raise HTTPException(
                        status_code=400, detail=f"Method {selected_discrete_method!r} is not registered"
                    )

                results = []
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

        elif current == WizardStep.PLOT_SELECTION:
            selected_plots = [str(x) for x in form_data.getlist("selected_plots")]

            session.selected_plots = selected_plots

            if is_forward:
                is_incomplete = session.group_column is None or (
                    not session.selected_value_columns and not session.selected_discrete_columns
                )
                if is_incomplete:
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

                session.top_n_columns = len(top_columns)

                plot_results = []
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

                        plot_res_obj = generator.generate(filtered_df, session.group_column or "", value_col, **kwargs)
                        plot_res_obj.column_name = value_col
                        plot_results.append(plot_res_obj)

                session.plot_results = [p.model_dump() for p in plot_results]

    original_current = session.current_step
    session.current_step = target.value
    try:
        validate_step_transition(session, target)
    except Exception as e:
        session.current_step = original_current
        raise e

    session.current_step = original_current
    reset_to_step(session, target)
    store.save(session)

    if "hx-request" in request.headers:
        return render_step(request, session, store)
    else:
        import json

        return JSONResponse(content=json.loads(session.model_dump_json()))
