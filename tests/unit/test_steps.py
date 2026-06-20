"""Unit tests for wizard step state machine and guards."""

from __future__ import annotations

import pytest

from app.core.session import WizardSession
from app.wizard.steps import StepGuardError, WizardStep, validate_step_transition


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
        stat_result={"p_value": 0.05},
    )
    validate_step_transition(session, WizardStep.PLOT_SELECTION)


def test_export_allowed_after_full_flow() -> None:
    """Export is allowed once all preceding steps are complete."""
    session = WizardSession(
        dataset_id="ds1",
        current_step="export",
        selected_method="ttest",
        stat_result={"p_value": 0.05},
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
