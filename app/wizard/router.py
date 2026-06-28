"""Wizard orchestration API router returning HTML templates for HTMX and JSON for REST clients."""

from __future__ import annotations

import base64
import io
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.core.session import (
    ClusterExclusion,
    HierarchyConfig,
    InMemorySessionStore,
    SessionStore,
    WizardSession,
)
from app.datasets.hierarchical import HierarchicalData
from app.datasets.models import DatasetInfo
from app.datasets.repository import (
    DatasetRepository,
    MultiFormatDatasetRepository,
)
from app.datasets.utils import resolve_selected_discrete_columns, resolve_selected_value_columns
from app.exporters.base import exporter_registry
from app.filters.base import apply_filter_pipeline
from app.plots.base import PlotResult, plot_registry
from app.stats.base import (
    StatMethod,
    StatResult,
    compute_properties,
    compute_properties_for_columns,
    stat_registry,
)
from app.stats.properties import build_cluster_aggregates, compute_quick_icc
from app.wizard.schemas import (
    DatasetSelectionRequest,
    ExportRequest,
    FiltersConfigRequest,
    MethodSelectionRequest,
    PlotSelectionRequest,
)
from app.wizard.steps import WizardStep, _completed_steps, reset_to_step, validate_step_transition

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/wizard", tags=["wizard"])
templates = Jinja2Templates(directory="app/templates")

# Singletons/Defaults
_session_store = InMemorySessionStore()


def get_session_store() -> SessionStore:
    """Dependency provider for the SessionStore."""
    return _session_store


def get_dataset_repository() -> DatasetRepository:
    """Dependency provider for the DatasetRepository."""
    data_dir = Path(os.getenv("EXPYRI_DATA_DIR", "data"))
    return MultiFormatDatasetRepository(data_dir)


def get_session(
    session_id: str,
    store: SessionStore = Depends(get_session_store),
) -> WizardSession:
    """Fetch the wizard session by ID or raise 404."""
    session = store.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Wizard session {session_id!r} not found",
        )
    return session


def get_filtered_dataset(
    session: WizardSession,
    repo: DatasetRepository,
) -> pd.DataFrame:
    """Load the dataset and apply the session's filter pipeline."""
    if session.dataset_id is None:
        raise HTTPException(status_code=400, detail="Dataset not selected")

    try:
        df = repo.load_dataset(session.dataset_id)
    except KeyError:
        raise HTTPException(status_code=400, detail="Dataset missing") from None

    try:
        df = apply_filter_pipeline(df, session.filters_config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Filter registration missing: {e}") from None

    if session.group_column and session.selected_groups:
        df = df[df[session.group_column].astype(str).isin(session.selected_groups)]
    if session.hierarchy and session.hierarchy.selected_clusters:
        df = df[df[session.hierarchy.cluster_col].astype(str).isin(session.hierarchy.selected_clusters)]
    return df


def _get_grouped_data(df: pd.DataFrame, group_col: str, value_col: str) -> dict[str, list[Any]]:
    """Helper to group a DataFrame by a column and extract non-null value lists."""
    grouped = df.groupby(group_col)[value_col]
    return {str(name): list(group.dropna().values) for name, group in grouped}


def generate_significance_chart_base64(stat_results: list[dict[str, Any]], limit: float) -> str | None:
    """Generate a significance scatter plot of p-values using matplotlib and encode it to base64."""
    valid_results = [res for res in stat_results if res.get("p_value") is not None]
    if not valid_results:
        return None

    valid_results.sort(key=lambda x: x["p_value"])
    labels = [res.get("column_name") or "Unknown" for res in valid_results]
    p_values = [res["p_value"] for res in valid_results]
    strict_limit = limit * 0.2

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 3.5))

    ax.axhline(y=limit, color=(250 / 255, 204 / 255, 21 / 255, 0.5), linestyle="--", label=f"p-value Limit ({limit})")
    ax.axhline(
        y=strict_limit,
        color=(16 / 255, 185 / 255, 129 / 255, 0.5),
        linestyle="--",
        label=f"Strict Limit ({strict_limit:.3f})",
    )

    ax.axhspan(0, strict_limit, color="#10b981", alpha=0.1)
    ax.axhspan(strict_limit, limit, color="#facc15", alpha=0.1)

    colors = []
    for p in p_values:
        if p <= strict_limit:
            colors.append("#10b981")
        elif p <= limit:
            colors.append("#facc15")
        else:
            colors.append("#ffffff")

    x_indices = np.arange(len(labels))
    ax.scatter(x_indices, p_values, color=colors, edgecolor=(1, 1, 1, 0.3), s=60, zorder=5)

    ax.set_xticks(x_indices)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("p-value", fontsize=10)
    ax.set_title("Statistical Significance by Column", fontsize=11)
    ax.set_ylim(-0.02, 1.02)

    ax.grid(True, linestyle=":", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
    finally:
        plt.close(fig)

    return img_str


def render_step(  # noqa: C901
    request: Request,
    session: WizardSession,
    store: SessionStore,
    plots_sig_filter: float = 0.05,
    sort_field: str = "column_name",
    sort_asc: bool = True,
) -> Response:
    """Helper to render either the full workspace or HTMX out-of-band updates."""
    completed = _completed_steps(session)
    completed_names = {s.value for s in completed}

    # Sort results if present
    if session.stat_results:

        def sort_key(x: dict[str, Any]) -> tuple[bool, Any]:
            val = x.get(sort_field)
            return (val is None, val)

        session.stat_results.sort(key=sort_key, reverse=not sort_asc)

    columns: list[Any] = []
    available_groups: list[str] = []
    available_clusters: list[str] = []
    applicable_continuous: dict[str, Any] = {}
    applicable_discrete: dict[str, Any] = {}
    applicable_plots: dict[str, Any] = {}
    sig_chart_base64: str | None = None
    matched_count = 0

    repo = get_dataset_repository()

    if session.dataset_id:
        try:
            columns = repo.get_schema(session.dataset_id).columns or []
            df = repo.load_dataset(session.dataset_id)
            if session.group_column:
                available_groups = sorted(df[session.group_column].dropna().astype(str).unique().tolist())
            if session.hierarchy and session.hierarchy.cluster_col:
                available_clusters = sorted(df[session.hierarchy.cluster_col].dropna().astype(str).unique().tolist())
        except Exception:
            pass

    if session.current_step == "stat_method":
        try:
            filtered_df = get_filtered_dataset(session, repo)
            if session.selected_value_columns:
                props_map = compute_properties_for_columns(session, filtered_df, session.selected_value_columns)
                applicable_continuous = stat_registry.get_applicable_intersect(props_map)
            if session.selected_discrete_columns:
                props_map_discrete = compute_properties_for_columns(
                    session, filtered_df, session.selected_discrete_columns
                )
                applicable_discrete = stat_registry.get_applicable_intersect(props_map_discrete)
        except Exception:
            pass

    elif session.current_step in ("results", "plot_selection", "export"):
        if session.stat_results:
            matched_count = sum(
                1
                for res in session.stat_results
                if res.get("p_value") is not None and res["p_value"] <= plots_sig_filter
            )
            sig_chart_base64 = generate_significance_chart_base64(session.stat_results, plots_sig_filter)

        if session.current_step == "plot_selection":
            try:
                filtered_df = get_filtered_dataset(session, repo)
                if session.selected_value_columns:
                    props = compute_properties(session, filtered_df, session.selected_value_columns[0])
                    applicable_plots = plot_registry.get_applicable(props)
            except Exception:
                pass

    context = {
        "request": request,
        "session": session,
        "completed_steps": completed_names,
        "is_step_completed": session.current_step in completed_names,
        "columns": columns,
        "available_groups": available_groups,
        "available_clusters": available_clusters,
        "continuous_methods": [
            (name, method) for name, method in stat_registry.list_all().items() if name != "chi_square"
        ],
        "discrete_methods": [
            (name, method) for name, method in stat_registry.list_all().items() if name == "chi_square"
        ],
        "applicable_continuous": applicable_continuous,
        "applicable_discrete": applicable_discrete,
        "applicable_plots": applicable_plots,
        "sig_chart_base64": sig_chart_base64,
        "plots_sig_filter": plots_sig_filter,
        "matched_count": matched_count,
        "sort_field": sort_field,
        "sort_asc": sort_asc,
        "datasets": repo.list_datasets(),
    }

    if "hx-request" in request.headers:
        workspace_html = templates.get_template("partials/workspace.html").render(context)
        session_info_html = templates.get_template("partials/session_info.html").render(context)
        sidebar_actions_html = templates.get_template("partials/sidebar_actions.html").render(context)
        steps_nav_html = templates.get_template("partials/steps_nav.html").render(context)

        full_html = f"{workspace_html}\n{session_info_html}\n{sidebar_actions_html}\n{steps_nav_html}"
        return HTMLResponse(content=full_html)
    else:
        return templates.TemplateResponse(request=request, name="base.html", context=context)


# ==========================================
# 1. HTML / HTMX VIEW ENDPOINTS
# ==========================================


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    store: SessionStore = Depends(get_session_store),
) -> Response:
    """Home route that automatically initializes a wizard session."""
    session = store.create()
    return render_step(request, session, store)


@router.post("/sessions/{session_id}/restart")
def restart_session(
    session_id: str,
    request: Request,
    store: SessionStore = Depends(get_session_store),
) -> Response:
    """Restart and destroy the current session by redirecting to root or returning JSON."""
    session = store.create()

    if "hx-request" in request.headers:
        from fastapi.responses import HTMLResponse

        return HTMLResponse(content="", headers={"HX-Redirect": "/"})

    accept = request.headers.get("accept", "")
    if "application/json" in accept or "json" in accept:
        import json

        return JSONResponse(content=json.loads(session.model_dump_json()))

    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/", status_code=303)


@router.post("/sessions/{session_id}/select-dataset-id")
def select_dataset_id(
    session_id: str,
    request: Request,
    dataset_id: str = Form(...),
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
    store: SessionStore = Depends(get_session_store),
) -> Response:
    """Select an existing dataset from the repository list."""
    reset_to_step(session, WizardStep.DATASET_SELECTION)
    session.dataset_id = dataset_id if dataset_id else None
    if session.dataset_id:
        try:
            df = repo.load_dataset(session.dataset_id)
            session.selected_value_columns = resolve_selected_value_columns(df, "", [])
            session.selected_discrete_columns = resolve_selected_discrete_columns(df, "", [])
        except Exception:
            session.selected_value_columns = []
            session.selected_discrete_columns = []
    else:
        session.selected_value_columns = []
        session.selected_discrete_columns = []
    store.save(session)
    return render_step(request, session, store)


@router.post("/sessions/{session_id}/toggle-hierarchy")
def toggle_hierarchy_htmx(
    session_id: str,
    request: Request,
    enabled: str,
    session: WizardSession = Depends(get_session),
    store: SessionStore = Depends(get_session_store),
) -> Response:
    """Toggle hierarchical configuration mode (HTMX view)."""
    if enabled == "true":
        session.hierarchy = HierarchyConfig(group_col="", cluster_col="")
    else:
        session.hierarchy = None
    store.save(session)
    return render_step(request, session, store)


@router.post("/sessions/{session_id}/update-group-col")
def update_group_col(
    session_id: str,
    request: Request,
    group_column: str = Form(...),
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
    store: SessionStore = Depends(get_session_store),
) -> Response:
    """Update active group column selection and reload subgroups."""
    session.group_column = group_column if group_column else None
    if session.group_column:
        if session.group_column in session.selected_value_columns:
            session.selected_value_columns.remove(session.group_column)
        if session.group_column in session.selected_discrete_columns:
            session.selected_discrete_columns.remove(session.group_column)

    if session.group_column and session.dataset_id:
        try:
            df = repo.load_dataset(session.dataset_id)
            session.selected_groups = sorted(df[session.group_column].dropna().astype(str).unique().tolist())
        except Exception:
            session.selected_groups = []
    else:
        session.selected_groups = []
    store.save(session)
    return render_step(request, session, store)


@router.post("/sessions/{session_id}/update-cluster-col")
def update_cluster_col(
    session_id: str,
    request: Request,
    cluster_col: str = Form(...),
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
    store: SessionStore = Depends(get_session_store),
) -> Response:
    """Update active cluster selection and reload clusters."""
    if session.hierarchy:
        session.hierarchy.cluster_col = cluster_col
        if cluster_col:
            if cluster_col in session.selected_value_columns:
                session.selected_value_columns.remove(cluster_col)
            if cluster_col in session.selected_discrete_columns:
                session.selected_discrete_columns.remove(cluster_col)

        if cluster_col and session.dataset_id:
            try:
                df = repo.load_dataset(session.dataset_id)
                session.hierarchy.selected_clusters = sorted(df[cluster_col].dropna().astype(str).unique().tolist())
            except Exception:
                session.hierarchy.selected_clusters = []
        else:
            session.hierarchy.selected_clusters = []
    store.save(session)
    return render_step(request, session, store)


@router.post("/sessions/{session_id}/submit-dataset-config")
def submit_dataset_config(  # noqa: C901
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
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
    store: SessionStore = Depends(get_session_store),
) -> Response:
    """Submit the dataset setup choices and transition to Step 2 (Filters)."""
    validate_step_transition(session, WizardStep.DATASET_SELECTION)

    if not session.dataset_id:
        raise HTTPException(status_code=400, detail="Dataset not selected")

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
            detail=f"Group column {group_column!r} must be discrete/categorical, but it is numeric.",
        )

    session.group_column = group_column
    session.selected_groups = selected_groups

    if session.hierarchy:
        if not cluster_col:
            raise HTTPException(status_code=400, detail="Cluster column is required in hierarchical mode")
        if cluster_col == group_column:
            raise HTTPException(
                status_code=400,
                detail="Cluster column must not be the same as the group column.",
            )
        cluster_col_info = next((col for col in schema.columns or [] if col.name == cluster_col), None)
        if not cluster_col_info:
            raise HTTPException(status_code=400, detail=f"Cluster column {cluster_col!r} not found")
        if cluster_col_info.is_numeric:
            raise HTTPException(
                status_code=400,
                detail=f"Cluster column {cluster_col!r} must be discrete/categorical, but it is numeric.",
            )
        session.hierarchy = HierarchyConfig(
            group_col=group_column,
            cluster_col=cluster_col,
            selected_clusters=selected_clusters,
            unit_col=unit_col if unit_col else None,
            x_col=x_col if x_col else None,
            y_col=y_col if y_col else None,
        )
    else:
        session.hierarchy = None

    is_empty_val = not selected_value_columns
    is_empty_disc = not selected_discrete_columns

    try:
        if is_empty_val and is_empty_disc:
            session.selected_value_columns = resolve_selected_value_columns(df, group_column, [])
            session.selected_discrete_columns = resolve_selected_discrete_columns(df, group_column, [])
        else:
            session.selected_value_columns = (
                resolve_selected_value_columns(df, group_column, selected_value_columns) if not is_empty_val else []
            )
            session.selected_discrete_columns = (
                resolve_selected_discrete_columns(df, group_column, selected_discrete_columns)
                if not is_empty_disc
                else []
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    if not session.selected_value_columns and not session.selected_discrete_columns:
        raise HTTPException(status_code=400, detail="Select at least one dependent column to analyze.")

    session.current_step = WizardStep.FILTERS.value
    store.save(session)
    return render_step(request, session, store)


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
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
    store: SessionStore = Depends(get_session_store),
) -> Response:
    """Add a new filter and validate it by dry-running the filter pipeline."""
    validate_step_transition(session, WizardStep.FILTERS)

    if session.dataset_id is None:
        raise HTTPException(status_code=400, detail="Dataset not selected")

    try:
        df = repo.load_dataset(session.dataset_id)
    except KeyError:
        raise HTTPException(status_code=400, detail="Dataset missing") from None

    params: dict[str, Any] = {}
    if filter_type == "numeric_range":
        params["min_val"] = float(min_val) if min_val else None
        params["max_val"] = float(max_val) if max_val else None
    elif filter_type == "category_filter":
        params["values"] = [v.strip() for v in values.split(",")] if values else []
        params["exclude"] = bool(exclude)
    elif filter_type == "cluster_exclusion":
        if not cluster_id or not reason:
            raise HTTPException(status_code=400, detail="Cluster ID and Reason are required")
        params["exclusions"] = [{"cluster_id": cluster_id, "reason": reason}]
    else:
        raise HTTPException(status_code=400, detail="Invalid filter type")

    new_filter = {
        "name": filter_type,
        "column": column,
        "params": params,
    }

    test_configs = session.filters_config + [new_filter]
    try:
        apply_filter_pipeline(df, test_configs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid filter parameters: {e}") from None

    session.filters_config = test_configs

    if filter_type == "cluster_exclusion" and cluster_id and reason:
        session.excluded_clusters.append(ClusterExclusion(cluster_id=cluster_id, reason=reason))

    store.save(session)
    return render_step(request, session, store)


@router.delete("/sessions/{session_id}/filters/{index}")
def delete_filter(
    session_id: str,
    index: int,
    request: Request,
    session: WizardSession = Depends(get_session),
    store: SessionStore = Depends(get_session_store),
) -> Response:
    """Remove a filter by its index position."""
    validate_step_transition(session, WizardStep.FILTERS)

    if index < 0 or index >= len(session.filters_config):
        raise HTTPException(status_code=400, detail="Filter index out of range")

    removed = session.filters_config.pop(index)

    if removed.get("name") == "cluster_exclusion":
        exclusions_list = removed.get("params", {}).get("exclusions", [])
        for item in exclusions_list:
            session.excluded_clusters = [
                ex for ex in session.excluded_clusters if ex.cluster_id != str(item["cluster_id"])
            ]

    store.save(session)
    return render_step(request, session, store)


@router.post("/sessions/{session_id}/submit-filters")
def submit_filters(
    session_id: str,
    request: Request,
    session: WizardSession = Depends(get_session),
    store: SessionStore = Depends(get_session_store),
) -> Response:
    """Submit the filters and move to Step 3 (Choose Method)."""
    validate_step_transition(session, WizardStep.FILTERS)
    session.current_step = WizardStep.STAT_METHOD.value
    store.save(session)
    return render_step(request, session, store)


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
    """Update selected statistical methods in session and re-render step 3 (to update sidebar buttons)."""
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


# ==========================================
# 2. BACKWARDS-COMPATIBLE JSON API ENDPOINTS
# ==========================================


@router.get("/sessions/{session_id}", response_model=WizardSession, include_in_schema=False)
def get_session_state_json(
    session: WizardSession = Depends(get_session),
) -> WizardSession:
    """JSON compatibility route: Get session state."""
    return session


@router.post("/sessions", response_model=WizardSession)
def create_session(
    store: SessionStore = Depends(get_session_store),
) -> WizardSession:
    """JSON compatibility route: Create new session."""
    return store.create()


@router.post("/upload", response_model=DatasetInfo)
def upload_dataset_anon(
    file: UploadFile = File(...),
    repo: DatasetRepository = Depends(get_dataset_repository),
) -> DatasetInfo:
    """JSON compatibility route: Anonymous file upload."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename not provided")

    safe_filename = Path(file.filename).name
    if not safe_filename or safe_filename in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")

    data_dir = getattr(repo, "_data_dir", Path("data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / safe_filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    dataset_id = file_path.stem
    try:
        return repo.get_schema(dataset_id)
    except KeyError:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Unsupported file format") from None


@router.post("/sessions/{session_id}/upload", include_in_schema=False)
def upload_dataset_session_json(
    session_id: str,
    request: Request,
    file: UploadFile = File(...),
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
    store: SessionStore = Depends(get_session_store),
) -> Response:
    """JSON compatibility and HTMX route: Upload dataset to specific session."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename not provided")

    safe_filename = Path(file.filename).name
    if not safe_filename or safe_filename in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")

    data_dir = getattr(repo, "_data_dir", Path("data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / safe_filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    dataset_id = file_path.stem
    try:
        repo.get_schema(dataset_id)
    except KeyError:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Unsupported file format") from None

    reset_to_step(session, WizardStep.DATASET_SELECTION)
    session.dataset_id = dataset_id
    store.save(session)

    if "hx-request" in request.headers:
        return render_step(request, session, store)
    else:
        import json

        return JSONResponse(content=json.loads(session.model_dump_json()))


@router.get("/datasets", response_model=list[DatasetInfo])
def list_available_datasets(
    repo: DatasetRepository = Depends(get_dataset_repository),
) -> list[DatasetInfo]:
    """JSON compatibility route: Retrieve all available datasets."""
    return repo.list_datasets()


@router.get("/datasets/{dataset_id}/columns/{column_name}/unique")
def get_column_unique_values(
    dataset_id: str,
    column_name: str,
    repo: DatasetRepository = Depends(get_dataset_repository),
) -> list[str]:
    """JSON compatibility route: Retrieve unique sorted values of a column."""
    try:
        df = repo.load_dataset(dataset_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id!r} not found",
        ) from None

    if column_name not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column {column_name!r} not found in dataset",
        )

    unique_vals = df[column_name].dropna().unique()
    try:
        sorted_vals = sorted(unique_vals)
    except TypeError:
        sorted_vals = sorted(unique_vals, key=str)

    return [str(v) for v in sorted_vals]


@router.post("/sessions/{session_id}/dataset", response_model=WizardSession)
def select_dataset(
    req: DatasetSelectionRequest,
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
    store: SessionStore = Depends(get_session_store),
) -> WizardSession:
    """JSON compatibility route: Select a dataset and map columns."""
    validate_step_transition(session, WizardStep.DATASET_SELECTION)

    try:
        schema = repo.get_schema(req.dataset_id)
        df = repo.load_dataset(req.dataset_id)
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset {req.dataset_id!r} not found",
        ) from None

    group_col_info = next((col for col in schema.columns or [] if col.name == req.group_column), None)
    if not group_col_info:
        raise HTTPException(status_code=400, detail=f"Group column {req.group_column!r} not found")
    if group_col_info.is_numeric:
        raise HTTPException(
            status_code=400,
            detail=f"Group column {req.group_column!r} must be discrete/categorical, but it is numeric.",
        )

    is_empty_val = not req.selected_value_columns
    is_empty_disc = not req.selected_discrete_columns

    try:
        if is_empty_val and is_empty_disc:
            session.selected_value_columns = resolve_selected_value_columns(df, req.group_column, [])
            session.selected_discrete_columns = resolve_selected_discrete_columns(df, req.group_column, [])
        else:
            session.selected_value_columns = (
                resolve_selected_value_columns(df, req.group_column, req.selected_value_columns)
                if not is_empty_val
                else []
            )
            session.selected_discrete_columns = (
                resolve_selected_discrete_columns(df, req.group_column, req.selected_discrete_columns)
                if not is_empty_disc
                else []
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    if not session.selected_value_columns and not session.selected_discrete_columns:
        raise HTTPException(status_code=400, detail="Select at least one dependent column to analyze.")

    session.dataset_id = req.dataset_id
    session.group_column = req.group_column
    session.selected_groups = req.selected_groups
    session.hierarchy = None
    session.excluded_clusters = []
    session.current_step = WizardStep.FILTERS.value
    store.save(session)
    return session


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


class HierarchyRequest(BaseModel):
    """Hierarchy configuration request."""

    group_col: str
    cluster_col: str
    selected_clusters: list[str] = Field(default_factory=list)
    unit_col: str | None = None
    x_col: str | None = None
    y_col: str | None = None


class HierarchyResponse(BaseModel):
    """Hierarchy configuration response."""

    session: WizardSession
    metric_kinds: dict[str, str]


@router.post("/sessions/{session_id}/hierarchy", response_model=HierarchyResponse)
def set_hierarchy(  # noqa: C901
    req: HierarchyRequest,
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
    store: SessionStore = Depends(get_session_store),
) -> HierarchyResponse:
    """JSON compatibility route: Configure hierarchical settings."""
    if session.dataset_id is None:
        raise HTTPException(status_code=400, detail="Dataset not selected")

    try:
        df = repo.load_dataset(session.dataset_id)
    except KeyError:
        raise HTTPException(status_code=400, detail="Dataset missing") from None

    required_cols = [req.group_col, req.cluster_col]
    if req.unit_col:
        required_cols.append(req.unit_col)
    for col in required_cols:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column {col!r} not found in dataset")

    if req.cluster_col == req.group_col:
        raise HTTPException(
            status_code=400,
            detail="Cluster column must not be the same as the group column.",
        )

    if pd.api.types.is_numeric_dtype(df[req.group_col]):
        raise HTTPException(
            status_code=400,
            detail=f"Group column {req.group_col!r} must be discrete/categorical, but it is numeric.",
        )

    if pd.api.types.is_numeric_dtype(df[req.cluster_col]):
        raise HTTPException(
            status_code=400,
            detail=f"Cluster column {req.cluster_col!r} must be discrete/categorical, but it is numeric.",
        )

    if req.x_col and req.x_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column {req.x_col!r} not found in dataset")
    if req.y_col and req.y_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column {req.y_col!r} not found in dataset")

    session.hierarchy = HierarchyConfig(
        group_col=req.group_col,
        cluster_col=req.cluster_col,
        selected_clusters=req.selected_clusters,
        unit_col=req.unit_col,
        x_col=req.x_col,
        y_col=req.y_col,
    )

    ignored_cols = {req.group_col, req.cluster_col}
    if req.unit_col:
        ignored_cols.add(req.unit_col)
    if req.x_col:
        ignored_cols.add(req.x_col)
    if req.y_col:
        ignored_cols.add(req.y_col)

    new_value_cols = []
    for col in session.selected_value_columns:
        if col in ignored_cols:
            continue
        if col in df.columns and (pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])):
            new_value_cols.append(col)
    session.selected_value_columns = new_value_cols

    new_discrete_cols = []
    for col in session.selected_discrete_columns:
        if col in ignored_cols:
            continue
        if col in df.columns and (pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])):
            unique_vals = set(df[col].dropna().unique())
            if unique_vals.issubset({0, 1}):
                new_discrete_cols.append(col)
    session.selected_discrete_columns = new_discrete_cols

    store.save(session)

    metric_kinds = {}
    for col in df.columns:
        if col in ignored_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            unique_vals = set(df[col].dropna().unique())
            if unique_vals.issubset({0, 1}) and len(unique_vals) > 0:
                metric_kinds[col] = "binary_proportion"
            else:
                metric_kinds[col] = "continuous"
        else:
            metric_kinds[col] = "unsupported"

    return HierarchyResponse(session=session, metric_kinds=metric_kinds)


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
