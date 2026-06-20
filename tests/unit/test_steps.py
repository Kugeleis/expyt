"""Unit tests for wizard step state machine and guards."""

from __future__ import annotations

import pytest

from app.core.session import WizardSession
from app.wizard.steps import (
    StepGuardError,
    WizardStep,
    reset_to_step,
    validate_step_transition,
)


def test_dataset_selection_always_allowed() -> None:
    """First step has no prerequisites."""
    session = WizardSession()
    validate_step_transition(session, WizardStep.DATASET_SELECTION)


def test_filters_requires_dataset() -> None:
    """Filters step requires dataset to be selected."""
    session = WizardSession()
    with pytest.raises(StepGuardError, match="dataset_selection"):
        validate_step_transition(session, WizardStep.FILTERS)


def test_filters_allowed_after_dataset() -> None:
    """Filters step is allowed when dataset is selected."""
    session = WizardSession(dataset_id="ds1", current_step="filters")
    validate_step_transition(session, WizardStep.FILTERS)


def test_stat_method_requires_dataset_and_filters() -> None:
    """Stat method step requires both dataset and filters steps completed."""
    session = WizardSession(dataset_id="ds1", current_step="filters")
    with pytest.raises(StepGuardError, match="filters"):
        validate_step_transition(session, WizardStep.STAT_METHOD)


def test_stat_method_allowed_after_filters() -> None:
    """Stat method step is allowed after dataset + filters."""
    session = WizardSession(
        dataset_id="ds1",
        current_step="stat_method",
    )
    validate_step_transition(session, WizardStep.STAT_METHOD)


def test_results_requires_method() -> None:
    """Results step requires a statistical method to be selected."""
    session = WizardSession(
        dataset_id="ds1",
        current_step="stat_method",
    )
    with pytest.raises(StepGuardError, match="stat_method"):
        validate_step_transition(session, WizardStep.RESULTS)


def test_results_allowed_after_method() -> None:
    """Results step is allowed after method selection."""
    session = WizardSession(
        dataset_id="ds1",
        current_step="results",
        selected_method="ttest",
    )
    validate_step_transition(session, WizardStep.RESULTS)


def test_skip_to_export_from_start_fails() -> None:
    """Cannot jump to export from the beginning."""
    session = WizardSession()
    with pytest.raises(StepGuardError) as exc_info:
        validate_step_transition(session, WizardStep.EXPORT)
    assert len(exc_info.value.missing) > 0


def test_plot_selection_requires_results() -> None:
    """Plot selection requires stat results to be computed."""
    session = WizardSession(
        dataset_id="ds1",
        current_step="results",
        selected_method="ttest",
    )
    with pytest.raises(StepGuardError, match="results"):
        validate_step_transition(session, WizardStep.PLOT_SELECTION)


def test_plot_selection_allowed_after_results() -> None:
    """Plot selection is allowed after results are computed."""
    session = WizardSession(
        dataset_id="ds1",
        current_step="plot_selection",
        selected_method="ttest",
        stat_results=[{"p_value": 0.05}],
    )
    validate_step_transition(session, WizardStep.PLOT_SELECTION)


def test_export_allowed_after_full_flow() -> None:
    """Export is allowed once all preceding steps are complete."""
    session = WizardSession(
        dataset_id="ds1",
        current_step="export",
        selected_method="ttest",
        stat_results=[{"p_value": 0.05}],
        plot_results=[{"type": "boxplot"}],
    )
    validate_step_transition(session, WizardStep.EXPORT)


def test_step_guard_error_attributes() -> None:
    """StepGuardError exposes target and missing steps."""
    error = StepGuardError(
        WizardStep.EXPORT,
        {WizardStep.DATASET_SELECTION, WizardStep.FILTERS},
    )
    assert error.target == WizardStep.EXPORT
    assert WizardStep.DATASET_SELECTION in error.missing
    assert "dataset_selection" in str(error)


def test_wizard_step_values() -> None:
    """WizardStep enum has the expected values."""
    assert WizardStep.DATASET_SELECTION.value == "dataset_selection"
    assert WizardStep.FILTERS.value == "filters"
    assert WizardStep.STAT_METHOD.value == "stat_method"
    assert WizardStep.RESULTS.value == "results"
    assert WizardStep.PLOT_SELECTION.value == "plot_selection"
    assert WizardStep.EXPORT.value == "export"


# --- Back-navigation / reset_to_step tests ---


def test_reset_to_filters_clears_downstream() -> None:
    """Resetting to filters clears method, results, plots, and export."""
    session = WizardSession(
        dataset_id="ds1",
        group_column="g",
        selected_value_columns=["v"],
        filters_config=[{"name": "numeric_range", "params": {"column": "v", "min": 1}}],
        selected_method="ttest",
        stat_results=[{"p_value": 0.05}],
        selected_plots=["boxplot"],
        plot_results=[{"type": "boxplot"}],
        export_format="pdf",
        current_step="export",
    )
    reset_to_step(session, WizardStep.FILTERS)

    # Fields for filters step itself and dataset should be preserved
    assert session.dataset_id == "ds1"
    assert session.group_column == "g"
    assert session.selected_value_columns == ["v"]
    expected_filters = [{"name": "numeric_range", "params": {"column": "v", "min": 1}}]
    assert session.filters_config == expected_filters

    # Everything after filters should be cleared
    assert session.selected_method is None
    assert not session.stat_results
    assert session.selected_plots == []
    assert session.plot_results == []
    assert session.export_format is None
    assert session.current_step == "filters"


def test_reset_to_dataset_selection_clears_all_downstream() -> None:
    """Resetting to dataset_selection clears everything except dataset fields."""
    session = WizardSession(
        dataset_id="ds1",
        group_column="g",
        selected_value_columns=["v"],
        filters_config=[{"name": "f"}],
        selected_method="ttest",
        stat_results=[{"p_value": 0.05}],
        current_step="export",
    )
    reset_to_step(session, WizardStep.DATASET_SELECTION)

    # Dataset fields are kept (they belong to dataset_selection step)
    assert session.dataset_id == "ds1"
    # Everything after dataset_selection is cleared
    assert session.filters_config == []
    assert session.selected_method is None
    assert not session.stat_results
    assert session.current_step == "dataset_selection"


def test_reset_to_stat_method_preserves_filters() -> None:
    """Resetting to stat_method keeps dataset and filters intact."""
    session = WizardSession(
        dataset_id="ds1",
        group_column="g",
        selected_value_columns=["v"],
        filters_config=[{"name": "f"}],
        selected_method="ttest",
        stat_results=[{"p_value": 0.05}],
        selected_plots=["boxplot"],
        plot_results=[{"type": "boxplot"}],
        current_step="export",
    )
    reset_to_step(session, WizardStep.STAT_METHOD)

    assert session.dataset_id == "ds1"
    assert session.filters_config == [{"name": "f"}]
    assert session.selected_method == "ttest"  # kept — it's the target step
    assert not session.stat_results  # cleared — after target
    assert session.selected_plots == []
    assert session.plot_results == []
    assert session.current_step == "stat_method"


def test_reset_to_step_gives_fresh_mutable_defaults() -> None:
    """Each reset yields independent list instances."""
    s1 = WizardSession(dataset_id="ds1", current_step="export")
    s2 = WizardSession(dataset_id="ds2", current_step="export")
    reset_to_step(s1, WizardStep.FILTERS)
    reset_to_step(s2, WizardStep.FILTERS)
    s1.selected_plots.append("boxplot")
    assert s2.selected_plots == []  # must be independent


def test_validate_allows_revisiting_completed_step() -> None:
    """Validation passes for a step whose prerequisites are still completed."""
    session = WizardSession(
        dataset_id="ds1",
        current_step="stat_method",
        selected_method="ttest",
    )
    # Going back to filters — dataset_selection prerequisite is met
    validate_step_transition(session, WizardStep.FILTERS)
