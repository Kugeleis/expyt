from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import HTMLResponse

from app.core.session import SessionStore
from app.wizard.router.compatibility import router as compatibility_router
from app.wizard.router.dependencies import (
    get_dataset_repository,
    get_session_store,
    render_step,
)
from app.wizard.router.htmx_dataset import router as htmx_dataset_router
from app.wizard.router.htmx_filters import router as htmx_filters_router
from app.wizard.router.htmx_results import router as htmx_results_router
from app.wizard.router.json_dataset import router as json_dataset_router
from app.wizard.router.json_stats import router as json_stats_router

router = APIRouter(prefix="/wizard", tags=["wizard"])


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    store: SessionStore = Depends(get_session_store),
) -> Response:
    """Home route that automatically initializes a wizard session."""
    session = store.create()
    return render_step(request, session, store)


router.include_router(htmx_dataset_router)
router.include_router(htmx_filters_router)
router.include_router(htmx_results_router)
router.include_router(json_dataset_router)
router.include_router(json_stats_router)
router.include_router(compatibility_router)

__all__ = [
    "router",
    "get_session_store",
    "render_step",
    "get_dataset_repository",
]
