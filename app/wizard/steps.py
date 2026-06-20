"""Wizard step enum and state-machine guards.

Each step has a set of prerequisite steps that must be completed
before the session may advance to it.
"""

from __future__ import annotations

from enum import StrEnum

from app.core.session import WizardSession


class WizardStep(StrEnum):
    """Ordered steps in the experiment evaluation wizard."""

    DATASET_SELECTION = "dataset_selection"
    FILTERS = "filters"
    STAT_METHOD = "stat_method"
    RESULTS = "results"
    PLOT_SELECTION = "plot_selection"
    EXPORT = "export"


# Each step maps to the set of steps that must already be completed.
_PREREQUISITES: dict[WizardStep, set[WizardStep]] = {
    WizardStep.DATASET_SELECTION: set(),
    WizardStep.FILTERS: {WizardStep.DATASET_SELECTION},
    WizardStep.STAT_METHOD: {WizardStep.DATASET_SELECTION, WizardStep.FILTERS},
    WizardStep.RESULTS: {
        WizardStep.DATASET_SELECTION,
        WizardStep.FILTERS,
        WizardStep.STAT_METHOD,
    },
    WizardStep.PLOT_SELECTION: {
        WizardStep.DATASET_SELECTION,
        WizardStep.FILTERS,
        WizardStep.STAT_METHOD,
        WizardStep.RESULTS,
    },
    WizardStep.EXPORT: {
        WizardStep.DATASET_SELECTION,
        WizardStep.FILTERS,
        WizardStep.STAT_METHOD,
        WizardStep.RESULTS,
        WizardStep.PLOT_SELECTION,
    },
}

_STEP_ORDER = list(WizardStep)


def _completed_steps(session: WizardSession) -> set[WizardStep]:
    """Derive which steps have been completed from session state."""
    completed: set[WizardStep] = set()
    if session.dataset_id is not None:
        completed.add(WizardStep.DATASET_SELECTION)
    # Filters step is considered complete once dataset is chosen
    # (filters may be empty — that's a valid choice).
    if (
        session.dataset_id is not None
        and session.current_step != WizardStep.DATASET_SELECTION.value
    ):
        current_idx = _STEP_ORDER.index(WizardStep(session.current_step))
        filter_idx = _STEP_ORDER.index(WizardStep.FILTERS)
        if current_idx > filter_idx:
            completed.add(WizardStep.FILTERS)
    if session.selected_method is not None:
        completed.add(WizardStep.STAT_METHOD)
    if session.stat_result is not None:
        completed.add(WizardStep.RESULTS)
    if (
        session.current_step
        in {
            WizardStep.EXPORT.value,
        }
        or len(session.plot_results) > 0
    ):
        completed.add(WizardStep.PLOT_SELECTION)
    if session.export_format is not None:
        completed.add(WizardStep.EXPORT)
    return completed


class StepGuardError(Exception):
    """Raised when a step transition is not allowed."""

    def __init__(self, target: WizardStep, missing: set[WizardStep]) -> None:
        """Initialize the error.

        Args:
            target: The step the session attempted to move to.
            missing: The prerequisite steps that have not been completed.
        """
        self.target = target
        self.missing = missing
        names = ", ".join(sorted(s.value for s in missing))
        super().__init__(
            f"Cannot advance to {target.value!r}: "
            f"prerequisite steps not completed: {names}"
        )


def validate_step_transition(session: WizardSession, target: WizardStep) -> None:
    """Raise ``StepGuardError`` if *session* may not advance to *target*.

    Args:
        session: Current wizard session state.
        target: The step the caller wants to execute.

    Raises:
        StepGuardError: If prerequisite steps are not yet completed.
    """
    completed = _completed_steps(session)
    required = _PREREQUISITES[target]
    missing = required - completed
    if missing:
        raise StepGuardError(target, missing)
