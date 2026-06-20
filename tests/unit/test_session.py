"""Unit tests for wizard session management."""

from __future__ import annotations

from app.core.session import InMemorySessionStore, WizardSession


def test_session_defaults() -> None:
    """New session has sensible defaults."""
    session = WizardSession()
    assert session.session_id
    assert session.current_step == "dataset_selection"
    assert session.dataset_id is None
    assert session.filters_config == []
    assert session.selected_method is None
    assert session.stat_result is None
    assert session.selected_plots == []
    assert session.export_format is None


def test_store_create_and_get() -> None:
    """Creating a session stores it retrievably."""
    store = InMemorySessionStore()
    session = store.create()
    retrieved = store.get(session.session_id)
    assert retrieved is not None
    assert retrieved.session_id == session.session_id


def test_store_get_missing_returns_none() -> None:
    """Getting a non-existent session returns None."""
    store = InMemorySessionStore()
    assert store.get("nonexistent") is None


def test_store_save_updates() -> None:
    """Saving a session persists modifications."""
    store = InMemorySessionStore()
    session = store.create()
    old_updated = session.updated_at
    session.dataset_id = "test_dataset"
    store.save(session)
    retrieved = store.get(session.session_id)
    assert retrieved is not None
    assert retrieved.dataset_id == "test_dataset"
    assert retrieved.updated_at >= old_updated
