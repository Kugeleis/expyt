"""Unit tests for the generic plugin registry."""

from __future__ import annotations

from typing import Any

import pytest

from app.core.registry import Registry


class _DummyPlugin:
    """A minimal plugin for testing registration."""

    pass


class _ApplicablePlugin:
    """A plugin that implements is_applicable."""

    def __init__(self, *, accept: bool = True) -> None:
        self._accept = accept

    def is_applicable(self, **properties: Any) -> bool:
        """Return a fixed value for testing."""
        return self._accept


def test_register_and_get() -> None:
    """Registering a plugin makes it retrievable by name."""
    reg: Registry[_DummyPlugin] = Registry("test")

    @reg.register("alpha")
    class AlphaPlugin(_DummyPlugin):
        pass

    result = reg.get("alpha")
    assert isinstance(result, AlphaPlugin)


def test_register_duplicate_raises() -> None:
    """Registering the same name twice raises ValueError."""
    reg: Registry[_DummyPlugin] = Registry("test")

    @reg.register("dup")
    class FirstPlugin(_DummyPlugin):
        pass

    with pytest.raises(ValueError, match="already registered"):

        @reg.register("dup")
        class SecondPlugin(_DummyPlugin):
            pass


def test_get_missing_raises() -> None:
    """Looking up an unregistered name raises KeyError."""
    reg: Registry[_DummyPlugin] = Registry("test")
    with pytest.raises(KeyError, match="No test plugin"):
        reg.get("nonexistent")


def test_list_all() -> None:
    """list_all returns all registered plugins."""
    reg: Registry[_DummyPlugin] = Registry("test")

    @reg.register("a")
    class PluginA(_DummyPlugin):
        pass

    @reg.register("b")
    class PluginB(_DummyPlugin):
        pass

    all_plugins = reg.list_all()
    assert set(all_plugins.keys()) == {"a", "b"}
    assert isinstance(all_plugins["a"], PluginA)
    assert isinstance(all_plugins["b"], PluginB)


def test_list_all_returns_copy() -> None:
    """list_all returns a copy, not a reference to internals."""
    reg: Registry[_DummyPlugin] = Registry("test")

    @reg.register("x")
    class PluginX(_DummyPlugin):
        pass

    copy = reg.list_all()
    copy["injected"] = _DummyPlugin()  # type: ignore[assignment]
    assert "injected" not in reg.list_all()


def test_get_applicable_filters_by_is_applicable() -> None:
    """get_applicable only returns plugins whose is_applicable returns True."""
    reg: Registry[_ApplicablePlugin] = Registry("test")

    # Manually register instances with different applicability
    reg._plugins["yes"] = _ApplicablePlugin(accept=True)
    reg._plugins["no"] = _ApplicablePlugin(accept=False)

    applicable = reg.get_applicable()
    assert "yes" in applicable
    assert "no" not in applicable


def test_get_applicable_includes_non_applicable_protocol() -> None:
    """Plugins without is_applicable are always included."""
    reg: Registry[_DummyPlugin] = Registry("test")

    @reg.register("always")
    class AlwaysPlugin(_DummyPlugin):
        pass

    applicable = reg.get_applicable()
    assert "always" in applicable


def test_kind_property() -> None:
    """Registry exposes its kind label."""
    reg: Registry[_DummyPlugin] = Registry("filter")
    assert reg.kind == "filter"
