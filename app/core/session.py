"""Wizard session model and in-memory store.

The ``SessionStore`` protocol abstracts storage so it can be swapped
(e.g., for Redis) without touching wizard logic.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field


class WizardSession(BaseModel):
    """Holds the state of a single wizard run."""

    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    current_step: str = "dataset_selection"
    dataset_id: str | None = None
    group_column: str | None = None
    selected_value_columns: list[str] = Field(default_factory=list)
    filters_config: list[dict[str, Any]] = Field(default_factory=list)
    selected_method: str | None = None
    stat_results: list[dict[str, Any]] = Field(default_factory=list)
    selected_plots: list[str] = Field(default_factory=list)
    plot_results: list[dict[str, Any]] = Field(default_factory=list)
    top_n_columns: int = 1
    export_format: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SessionStore(Protocol):
    """Protocol for wizard session persistence."""

    def create(self) -> WizardSession:
        """Create and store a new session."""
        ...  # pragma: no cover

    def get(self, session_id: str) -> WizardSession | None:
        """Return the session or ``None`` if not found."""
        ...  # pragma: no cover

    def save(self, session: WizardSession) -> None:
        """Persist updates to an existing session."""
        ...  # pragma: no cover


class InMemorySessionStore:
    """In-memory implementation of ``SessionStore``."""

    def __init__(self) -> None:
        """Initialize an empty store."""
        self._sessions: dict[str, WizardSession] = {}

    def create(self) -> WizardSession:
        """Create and store a new session."""
        session = WizardSession()
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> WizardSession | None:
        """Return the session or ``None`` if not found."""
        return self._sessions.get(session_id)

    def save(self, session: WizardSession) -> None:
        """Persist updates to an existing session."""
        session.updated_at = datetime.now(UTC)
        self._sessions[session.session_id] = session
