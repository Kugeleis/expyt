from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from app.core.session import HierarchyConfig, SessionStore, WizardSession
from app.datasets.repository import DatasetRepository
from app.datasets.utils import resolve_selected_discrete_columns, resolve_selected_value_columns
from app.wizard.router.dependencies import (
    get_dataset_repository,
    get_session,
    get_session_store,
    render_step,
)
from app.wizard.steps import WizardStep, reset_to_step, validate_step_transition

router = APIRouter()


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
