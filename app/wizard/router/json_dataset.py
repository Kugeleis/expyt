"""JSON routers for Step 1: Dataset selection and configuration."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.session import SessionStore, WizardSession
from app.datasets.models import DatasetInfo
from app.datasets.repository import DatasetRepository
from app.wizard.router.dependencies import (
    get_dataset_repository,
    get_session,
    get_session_store,
    get_wizard_service,
    render_step,
)
from app.wizard.schemas import DatasetSelectionRequest
from app.wizard.service import WizardService

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
    service: WizardService = Depends(get_wizard_service),
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

    session = service.select_dataset_id(session_id, dataset_id)

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
    session_id: str,
    req: DatasetSelectionRequest,
    service: WizardService = Depends(get_wizard_service),
) -> WizardSession:
    """JSON compatibility route: Select a dataset and map columns."""
    service.select_dataset_id(session_id, req.dataset_id)
    return service.submit_dataset_config(
        session_id=session_id,
        group_column=req.group_column,
        selected_groups=req.selected_groups,
        selected_value_columns=req.selected_value_columns,
        selected_discrete_columns=req.selected_discrete_columns,
    )


@router.post("/sessions/{session_id}/hierarchy", response_model=HierarchyResponse)
def set_hierarchy(
    session_id: str,
    req: HierarchyRequest,
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
    service: WizardService = Depends(get_wizard_service),
) -> HierarchyResponse:
    """JSON compatibility route: Configure hierarchical settings."""
    if session.dataset_id is None:
        raise HTTPException(status_code=400, detail="Dataset not selected")

    try:
        df = repo.load_dataset(session.dataset_id)
    except KeyError:
        raise HTTPException(status_code=400, detail="Dataset missing") from None

    service.toggle_hierarchy(session_id, enabled=True)
    updated_session = service.submit_dataset_config(
        session_id=session_id,
        group_column=req.group_col,
        selected_groups=session.selected_groups,
        selected_value_columns=session.selected_value_columns,
        selected_discrete_columns=session.selected_discrete_columns,
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

    return HierarchyResponse(session=updated_session, metric_kinds=metric_kinds)
