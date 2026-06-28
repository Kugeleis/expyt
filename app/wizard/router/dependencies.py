import base64
import io
import os
from pathlib import Path
from typing import Any

import matplotlib
from fastapi import Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.core.session import InMemorySessionStore, SessionStore, WizardSession
from app.datasets.repository import DatasetRepository, MultiFormatDatasetRepository
from app.filters.base import apply_filter_pipeline
from app.plots.base import plot_registry
from app.stats.base import (
    compute_properties_for_columns,
    stat_registry,
)
from app.stats.properties import compute_properties
from app.wizard.steps import _completed_steps

templates = Jinja2Templates(directory="app/templates")

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
