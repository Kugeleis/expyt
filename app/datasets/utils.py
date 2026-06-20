"""Dataset utilities."""

from __future__ import annotations

import pandas as pd


def resolve_selected_value_columns(
    df: pd.DataFrame, group_column: str, selected_value_columns: list[str]
) -> list[str]:
    """Resolve and validate the columns to analyze for the dataset."""

    if selected_value_columns:
        missing = [col for col in selected_value_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Selected value columns not found in dataset: {missing}")
        if group_column in selected_value_columns:
            raise ValueError("Group column must not appear in selected value columns.")
        non_numeric = [
            col
            for col in selected_value_columns
            if not pd.api.types.is_numeric_dtype(df[col])
        ]
        if non_numeric:
            raise ValueError(
                "Selected value columns must be numeric. "
                f"Non-numeric columns: {non_numeric}"
            )
        return selected_value_columns

    numeric_columns = [
        col
        for col in df.select_dtypes(include=["number"]).columns
        if col != group_column
    ]
    if not numeric_columns:
        raise ValueError(
            "Dataset contains no numeric value columns apart from the group column. "
            "Please choose a different group column or provide "
            "explicit numeric value columns."
        )
    return numeric_columns
