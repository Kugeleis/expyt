"""Wizard orchestration API router."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from pydantic import BaseModel

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
from app.wizard.steps import WizardStep, reset_to_step, validate_step_transition

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/wizard", tags=["wizard"])

# Singletons/Defaults
_session_store = InMemorySessionStore()


def get_session_store() -> SessionStore:
    """Dependency provider for the SessionStore."""
    return _session_store


def get_dataset_repository() -> DatasetRepository:
    """Dependency provider for the DatasetRepository."""
    data_dir = Path(os.getenv("EXPYT_DATA_DIR", "data"))
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
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
) -> pd.DataFrame:
    """Dependency to load the dataset and apply the session's filter pipeline."""
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
    return df


def _get_grouped_data(df: pd.DataFrame, group_col: str, value_col: str) -> dict[str, list[Any]]:
    """Helper to group a DataFrame by a column and extract non-null value lists."""
    grouped = df.groupby(group_col)[value_col]
    return {str(name): list(group.dropna().values) for name, group in grouped}


@router.post("/sessions", response_model=WizardSession)
def create_session(
    store: SessionStore = Depends(get_session_store),
) -> WizardSession:
    """Create a new experiment evaluation wizard session."""
    return store.create()


@router.post("/sessions/{session_id}/go-to/{step}", response_model=WizardSession)
def go_to_step(
    step: str,
    session: WizardSession = Depends(get_session),
    store: SessionStore = Depends(get_session_store),
) -> WizardSession:
    """Navigate back to a previously completed wizard step.

    Resets all session state for steps after the target step so the
    user can redo them.
    """
    try:
        target = WizardStep(step)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown wizard step {step!r}",
        ) from None

    # Only allow navigating to the current step or an already-completed one
    validate_step_transition(session, target)

    reset_to_step(session, target)
    store.save(session)
    return session


@router.get("/sessions/{session_id}", response_model=WizardSession)
def get_session_state(
    session: WizardSession = Depends(get_session),
) -> WizardSession:
    """Get the current state of a wizard session."""
    return session


@router.get("/datasets", response_model=list[DatasetInfo])
def list_available_datasets(
    repo: DatasetRepository = Depends(get_dataset_repository),
) -> list[DatasetInfo]:
    """Retrieve all available datasets and their metadata schemas."""
    return repo.list_datasets()


@router.post("/upload", response_model=DatasetInfo)
def upload_dataset(
    file: UploadFile = File(...),
    repo: DatasetRepository = Depends(get_dataset_repository),
) -> DatasetInfo:
    """Upload a dataset file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename not provided")

    # Sanitize the filename to prevent path traversal
    safe_filename = Path(file.filename).name
    if not safe_filename or safe_filename in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Access the configured data directory correctly through the repository
    # The repository must support this attribute
    data_dir = getattr(repo, "_data_dir", Path(os.getenv("EXPYT_DATA_DIR", "data")))
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


@router.get("/datasets/{dataset_id}/columns/{column_name}/unique")
def get_column_unique_values(
    dataset_id: str,
    column_name: str,
    repo: DatasetRepository = Depends(get_dataset_repository),
) -> list[str]:
    """Retrieve unique non-null values of a column in a dataset, sorted as strings."""
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
    """Step 1: Select a dataset and map group/value columns."""
    validate_step_transition(session, WizardStep.DATASET_SELECTION)

    try:
        schema = repo.get_schema(req.dataset_id)
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset {req.dataset_id!r} not found",
        ) from None

    group_col_info = next((col for col in schema.columns or [] if col.name == req.group_column), None)
    if not group_col_info:
        raise HTTPException(
            status_code=400,
            detail=f"Group column {req.group_column!r} not found in dataset schema",
        )
    if group_col_info.is_numeric:
        raise HTTPException(
            status_code=400,
            detail=(f"Group column {req.group_column!r} must be discrete/categorical, but it is numeric."),
        )
    try:
        df = repo.load_dataset(req.dataset_id)
    except KeyError:
        raise HTTPException(status_code=400, detail="Dataset missing") from None

    logger.debug(
        "req.group_column=%r, req.selected_value_columns=%s, req.selected_discrete_columns=%s",
        req.group_column,
        req.selected_value_columns,
        req.selected_discrete_columns,
    )
    is_empty_val = not req.selected_value_columns
    is_empty_disc = not req.selected_discrete_columns

    try:
        if is_empty_val and is_empty_disc:
            selected_columns = resolve_selected_value_columns(df, req.group_column, [])
            selected_discrete = resolve_selected_discrete_columns(df, req.group_column, [])
        else:
            selected_columns = (
                resolve_selected_value_columns(df, req.group_column, req.selected_value_columns)
                if not is_empty_val
                else []
            )
            selected_discrete = (
                resolve_selected_discrete_columns(df, req.group_column, req.selected_discrete_columns)
                if not is_empty_disc
                else []
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    if not selected_columns and not selected_discrete:
        raise HTTPException(
            status_code=400,
            detail="Dataset must contain at least one dependent column to analyze (continuous or discrete).",
        )

    session.dataset_id = req.dataset_id
    session.group_column = req.group_column
    session.selected_value_columns = selected_columns
    session.selected_discrete_columns = selected_discrete
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
    """Step 2: Configure and apply preprocessing filters."""
    validate_step_transition(session, WizardStep.FILTERS)

    if session.dataset_id is None:
        raise HTTPException(status_code=400, detail="Dataset not selected")

    try:
        df = repo.load_dataset(session.dataset_id)
    except KeyError:
        raise HTTPException(status_code=400, detail="Dataset missing") from None

    filter_configs = [f.model_dump() for f in req.filters_config]

    # Validate by dry-running the filter pipeline
    try:
        apply_filter_pipeline(df, filter_configs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Filter registration missing: {e}") from None

    session.filters_config = filter_configs

    # Extract exclusions to session.excluded_clusters
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


@router.get("/sessions/{session_id}/methods")
def list_applicable_methods(
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
) -> list[dict[str, str]]:
    """Helper: Retrieve statistical methods applicable to the filtered dataset."""
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
    """Step 3: Select a statistical method."""
    validate_step_transition(session, WizardStep.STAT_METHOD)

    if session.group_column is None or (not session.selected_value_columns and not session.selected_discrete_columns):
        raise HTTPException(status_code=400, detail="Incomplete setup")

    filtered_df = get_filtered_dataset(session, repo)

    # Validate continuous method if continuous columns are selected
    if session.selected_value_columns:
        if not req.selected_method:
            raise HTTPException(status_code=400, detail="Method for continuous columns must be selected")
        props_map = compute_properties_for_columns(session, filtered_df, session.selected_value_columns)
        applicable = stat_registry.get_applicable_intersect(props_map)
        if req.selected_method not in applicable:
            raise HTTPException(
                status_code=400,
                detail=(f"Method {req.selected_method!r} is not applicable to the dataset properties"),
            )
        session.selected_method = req.selected_method
    else:
        session.selected_method = None

    # Validate discrete method if discrete columns are selected
    if session.selected_discrete_columns:
        if not req.selected_discrete_method:
            raise HTTPException(status_code=400, detail="Method for discrete columns must be selected")
        props_map_discrete = compute_properties_for_columns(session, filtered_df, session.selected_discrete_columns)
        applicable_discrete = stat_registry.get_applicable_intersect(props_map_discrete)
        if req.selected_discrete_method not in applicable_discrete:
            raise HTTPException(
                status_code=400,
                detail=(f"Method {req.selected_discrete_method!r} is not applicable to the dataset properties"),
            )
        session.selected_discrete_method = req.selected_discrete_method
    else:
        session.selected_discrete_method = None

    session.current_step = WizardStep.RESULTS.value
    store.save(session)
    return session


def _run_stat_for_column(
    filtered_df: pd.DataFrame,
    value_col: str,
    method: StatMethod,
    session: WizardSession,
) -> StatResult:
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
                detail=f"Column {value_col!r} is not supported in hierarchical mode. Please deselect it.",
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


@router.get("/sessions/{session_id}/results", response_model=list[StatResult])
def run_statistical_evaluation(
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
    store: SessionStore = Depends(get_session_store),
) -> list[StatResult]:
    """Step 4: Execute the selected statistical test on the filtered dataset.

    Loops over all selected value columns and ranks them by p-value.
    """
    validate_step_transition(session, WizardStep.RESULTS)

    if session.group_column is None or (not session.selected_value_columns and not session.selected_discrete_columns):
        raise HTTPException(status_code=400, detail="Incomplete setup")

    filtered_df = get_filtered_dataset(session, repo)
    results: list[StatResult] = []

    # Process continuous columns
    if session.selected_value_columns:
        if session.selected_method is None:
            raise HTTPException(status_code=400, detail="Continuous method not selected")
        method = stat_registry.get(session.selected_method)
        for value_col in session.selected_value_columns:
            results.append(_run_stat_for_column(filtered_df, value_col, method, session))

    # Process discrete columns
    if session.selected_discrete_columns:
        if session.selected_discrete_method is None:
            raise HTTPException(status_code=400, detail="Discrete method not selected")
        discrete_method = stat_registry.get(session.selected_discrete_method)
        for value_col in session.selected_discrete_columns:
            results.append(_run_stat_for_column(filtered_df, value_col, discrete_method, session))

    results.sort(key=lambda r: r.p_value)

    session.stat_results = [res.model_dump() for res in results]
    session.current_step = WizardStep.PLOT_SELECTION.value
    store.save(session)
    return results


@router.get("/sessions/{session_id}/plots")
def list_applicable_plots(
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
) -> list[dict[str, str]]:
    """Helper: Retrieve plots applicable to the filtered dataset."""
    validate_step_transition(session, WizardStep.PLOT_SELECTION)

    if session.group_column is None or (not session.selected_value_columns and not session.selected_discrete_columns):
        raise HTTPException(status_code=400, detail="Incomplete setup")

    filtered_df = get_filtered_dataset(session, repo)
    if session.selected_value_columns:
        props_map = compute_properties_for_columns(session, filtered_df, session.selected_value_columns)
        applicable = plot_registry.get_applicable_intersect(props_map)
        return [{"name": name, "description": inst.description} for name, inst in applicable.items()]
    return []


@router.post("/sessions/{session_id}/plots", response_model=WizardSession)
def generate_plots(
    req: PlotSelectionRequest,
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
    store: SessionStore = Depends(get_session_store),
) -> WizardSession:
    """Step 5: Select and generate visualizations.

    Generates plots for the top N ranked value columns based on statistical results.
    """
    validate_step_transition(session, WizardStep.PLOT_SELECTION)

    if session.group_column is None or (not session.selected_value_columns and not session.selected_discrete_columns):
        raise HTTPException(status_code=400, detail="Incomplete setup")

    filtered_df = get_filtered_dataset(session, repo)

    # Identify top N columns based on existing statistical results p-values
    # If no stat_results, fallback to just selecting the top N columns arbitrarily
    if session.stat_results:
        ranked_cols = [res["column_name"] for res in session.stat_results if "column_name" in res]
        top_columns = [col for col in ranked_cols if col and col in session.selected_value_columns][: req.top_n_columns]
    else:
        top_columns = session.selected_value_columns[: req.top_n_columns]

    plot_results: list[PlotResult] = []

    for value_col in top_columns:
        props = compute_properties(session, filtered_df, value_col)
        applicable = plot_registry.get_applicable(props)

        # Generate selected plots
        for name in req.selected_plots:
            if name not in applicable:
                raise HTTPException(
                    status_code=400,
                    detail=(f"Plot generator {name!r} is not applicable or not registered for column {value_col!r}"),
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
def export_results(
    req: ExportRequest,
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
    store: SessionStore = Depends(get_session_store),
) -> Response:
    """Step 6: Export the evaluation report or dataset."""
    validate_step_transition(session, WizardStep.EXPORT)

    try:
        exporter = exporter_registry.get(req.export_format)
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Exporter {req.export_format!r} is not registered",
        ) from None

    if session.group_column is None or not session.selected_value_columns:
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
    """Step 1b: Configure hierarchical settings."""
    if session.dataset_id is None:
        raise HTTPException(status_code=400, detail="Dataset not selected")

    try:
        df = repo.load_dataset(session.dataset_id)
    except KeyError:
        raise HTTPException(status_code=400, detail="Dataset missing") from None

    # Validate that columns exist in the dataset
    required_cols = [req.group_col, req.cluster_col]
    if req.unit_col:
        required_cols.append(req.unit_col)
    for col in required_cols:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column {col!r} not found in dataset")

    if req.x_col and req.x_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column {req.x_col!r} not found in dataset")
    if req.y_col and req.y_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column {req.y_col!r} not found in dataset")

    # Set session hierarchy
    session.hierarchy = HierarchyConfig(
        group_col=req.group_col,
        cluster_col=req.cluster_col,
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

    # Clean up selected columns for hierarchical mode
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

    # Detect metric kinds for numeric columns
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
