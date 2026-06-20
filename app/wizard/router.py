"""Wizard orchestration API router."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile

from app.core.session import (
    InMemorySessionStore,
    SessionStore,
    WizardSession,
)
from app.datasets.models import DatasetInfo
from app.datasets.repository import (
    DatasetRepository,
    MultiFormatDatasetRepository,
)
from app.exporters.base import exporter_registry
from app.filters.base import apply_filter_pipeline
from app.plots.base import PlotResult, plot_registry
from app.stats.base import (
    StatResult,
    compute_data_properties,
    stat_registry,
)
from app.wizard.schemas import (
    DatasetSelectionRequest,
    ExportRequest,
    FiltersConfigRequest,
    MethodSelectionRequest,
    PlotSelectionRequest,
)
from app.wizard.steps import WizardStep, validate_step_transition

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


@router.post("/sessions", response_model=WizardSession)
def create_session(
    store: SessionStore = Depends(get_session_store),
) -> WizardSession:
    """Create a new experiment evaluation wizard session."""
    return store.create()


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

    column_names = {col.name for col in schema.columns or []}
    if req.group_column not in column_names:
        raise HTTPException(
            status_code=400,
            detail=f"Group column {req.group_column!r} not found in dataset schema",
        )
    if req.value_column not in column_names:
        raise HTTPException(
            status_code=400,
            detail=f"Value column {req.value_column!r} not found in dataset schema",
        )

    session.dataset_id = req.dataset_id
    session.group_column = req.group_column
    session.value_column = req.value_column
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
        raise HTTPException(
            status_code=400, detail=f"Filter registration missing: {e}"
        ) from None

    session.filters_config = filter_configs
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

    if (
        session.dataset_id is None
        or session.group_column is None
        or session.value_column is None
    ):
        raise HTTPException(status_code=400, detail="Incomplete setup")

    try:
        df = repo.load_dataset(session.dataset_id)
    except KeyError:
        raise HTTPException(status_code=400, detail="Dataset missing") from None

    filtered_df = apply_filter_pipeline(df, session.filters_config)
    props = compute_data_properties(
        filtered_df, session.group_column, session.value_column
    )

    applicable = stat_registry.get_applicable(**props.model_dump())
    return [
        {"name": name, "description": inst.description}
        for name, inst in applicable.items()
    ]


@router.post("/sessions/{session_id}/method", response_model=WizardSession)
def select_method(
    req: MethodSelectionRequest,
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
    store: SessionStore = Depends(get_session_store),
) -> WizardSession:
    """Step 3: Select a statistical method."""
    validate_step_transition(session, WizardStep.STAT_METHOD)

    if (
        session.dataset_id is None
        or session.group_column is None
        or session.value_column is None
    ):
        raise HTTPException(status_code=400, detail="Incomplete setup")

    try:
        df = repo.load_dataset(session.dataset_id)
    except KeyError:
        raise HTTPException(status_code=400, detail="Dataset missing") from None

    filtered_df = apply_filter_pipeline(df, session.filters_config)
    props = compute_data_properties(
        filtered_df, session.group_column, session.value_column
    )

    applicable = stat_registry.get_applicable(**props.model_dump())
    if req.selected_method not in applicable:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Method {req.selected_method!r} is not applicable "
                "to the dataset properties"
            ),
        )

    session.selected_method = req.selected_method
    session.current_step = WizardStep.RESULTS.value
    store.save(session)
    return session


@router.get("/sessions/{session_id}/results", response_model=StatResult)
def run_statistical_evaluation(
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
    store: SessionStore = Depends(get_session_store),
) -> StatResult:
    """Step 4: Execute the selected statistical test on the filtered dataset."""
    validate_step_transition(session, WizardStep.RESULTS)

    if (
        session.dataset_id is None
        or session.group_column is None
        or session.value_column is None
        or session.selected_method is None
    ):
        raise HTTPException(status_code=400, detail="Incomplete setup")

    try:
        df = repo.load_dataset(session.dataset_id)
    except KeyError:
        raise HTTPException(status_code=400, detail="Dataset missing") from None

    filtered_df = apply_filter_pipeline(df, session.filters_config)
    grouped = filtered_df.groupby(session.group_column)[session.value_column]
    group_data = {str(name): list(group.dropna().values) for name, group in grouped}

    method = stat_registry.get(session.selected_method)
    try:
        res = method.run(group_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    session.stat_result = res.model_dump()
    session.current_step = WizardStep.PLOT_SELECTION.value
    store.save(session)
    return res


@router.get("/sessions/{session_id}/plots")
def list_applicable_plots(
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
) -> list[dict[str, str]]:
    """Helper: Retrieve plots applicable to the filtered dataset."""
    validate_step_transition(session, WizardStep.PLOT_SELECTION)

    if (
        session.dataset_id is None
        or session.group_column is None
        or session.value_column is None
    ):
        raise HTTPException(status_code=400, detail="Incomplete setup")

    try:
        df = repo.load_dataset(session.dataset_id)
    except KeyError:
        raise HTTPException(status_code=400, detail="Dataset missing") from None

    filtered_df = apply_filter_pipeline(df, session.filters_config)
    props = compute_data_properties(
        filtered_df, session.group_column, session.value_column
    )

    applicable = plot_registry.get_applicable(**props.model_dump())
    return [
        {"name": name, "description": inst.description}
        for name, inst in applicable.items()
    ]


@router.post("/sessions/{session_id}/plots", response_model=WizardSession)
def generate_plots(
    req: PlotSelectionRequest,
    session: WizardSession = Depends(get_session),
    repo: DatasetRepository = Depends(get_dataset_repository),
    store: SessionStore = Depends(get_session_store),
) -> WizardSession:
    """Step 5: Select and generate visualizations."""
    validate_step_transition(session, WizardStep.PLOT_SELECTION)

    if (
        session.dataset_id is None
        or session.group_column is None
        or session.value_column is None
    ):
        raise HTTPException(status_code=400, detail="Incomplete setup")

    try:
        df = repo.load_dataset(session.dataset_id)
    except KeyError:
        raise HTTPException(status_code=400, detail="Dataset missing") from None

    filtered_df = apply_filter_pipeline(df, session.filters_config)
    props = compute_data_properties(
        filtered_df, session.group_column, session.value_column
    )

    applicable = plot_registry.get_applicable(**props.model_dump())

    # Generate selected plots
    plot_results: list[PlotResult] = []
    for name in req.selected_plots:
        if name not in applicable:
            raise HTTPException(
                status_code=400,
                detail=f"Plot generator {name!r} is not applicable or not registered",
            )
        generator = plot_registry.get(name)
        plot_results.append(
            generator.generate(filtered_df, session.group_column, session.value_column)
        )

    session.selected_plots = req.selected_plots
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

    if (
        session.dataset_id is None
        or session.group_column is None
        or session.value_column is None
    ):
        raise HTTPException(status_code=400, detail="Incomplete setup")

    try:
        df = repo.load_dataset(session.dataset_id)
    except KeyError:
        raise HTTPException(status_code=400, detail="Dataset missing") from None

    filtered_df = apply_filter_pipeline(df, session.filters_config)

    stat_res = (
        StatResult.model_validate(session.stat_result) if session.stat_result else None
    )
    plots = [PlotResult.model_validate(p) for p in session.plot_results]

    export_res = exporter.export(stat_res, plots, filtered_df)

    session.export_format = req.export_format
    session.current_step = WizardStep.EXPORT.value
    store.save(session)

    return Response(
        content=export_res.content,
        media_type=export_res.content_type,
        headers={"Content-Disposition": f"attachment; filename={export_res.filename}"},
    )
