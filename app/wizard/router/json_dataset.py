import shutil
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.session import HierarchyConfig, SessionStore, WizardSession
from app.datasets.models import DatasetInfo
from app.datasets.repository import DatasetRepository
from app.datasets.utils import resolve_selected_discrete_columns, resolve_selected_value_columns
from app.wizard.router.dependencies import (
    get_dataset_repository,
    get_session,
    get_session_store,
    render_step,
)
from app.wizard.schemas import DatasetSelectionRequest
from app.wizard.steps import WizardStep, reset_to_step, validate_step_transition

router = APIRouter()


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
