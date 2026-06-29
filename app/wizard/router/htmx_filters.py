from typing import Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse

from app.core.session import ClusterExclusion, SessionStore, WizardSession
from app.datasets.repository import DatasetRepository
from app.filters.base import apply_filter_pipeline
from app.wizard.router.dependencies import (
    get_dataset_repository,
    get_session,
    get_session_store,
    render_step,
    templates,
)
from app.wizard.steps import WizardStep, validate_step_transition

router = APIRouter()


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
        params["column"] = column
        params["min"] = float(min_val) if min_val else None
        params["max"] = float(max_val) if max_val else None
    elif filter_type == "category_filter":
        params["column"] = column
        params["values"] = [v.strip() for v in values.split(",")] if values else []
        params["mode"] = "exclude" if exclude else "include"
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
