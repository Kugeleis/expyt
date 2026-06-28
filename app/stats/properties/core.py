import pandas as pd

from app.core.session import WizardSession
from app.stats.models import DataProperties
from app.stats.properties.flat import compute_data_properties
from app.stats.properties.hierarchical import compute_hierarchical_properties


def compute_properties(
    session: WizardSession,
    df: pd.DataFrame,
    value_col: str,
) -> DataProperties:
    """Top-level dispatcher to compute data properties, supporting hierarchical and flat data."""
    if session.hierarchy is not None:
        excluded_ids = [ex.cluster_id for ex in session.excluded_clusters]
        props = compute_hierarchical_properties(df, session.hierarchy, excluded_ids, value_col)
        props.has_hierarchy = True
        props.has_spatial_coords = session.hierarchy.x_col is not None and session.hierarchy.y_col is not None
    else:
        props = compute_data_properties(df, value_col, session.group_column or "")
        props.has_hierarchy = False
    return props


def compute_properties_for_columns(
    session: WizardSession,
    df: pd.DataFrame,
    value_columns: list[str],
) -> dict[str, DataProperties]:
    """Compute data properties for multiple columns under hierarchical or flat session configuration."""
    return {col: compute_properties(session, df, col) for col in value_columns}
