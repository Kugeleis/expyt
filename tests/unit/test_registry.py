"""Unit tests for the generic plugin registry."""

from __future__ import annotations

import pytest

from app.core.registry import Registry
from app.stats.base import DataProperties


class _DummyPlugin:
    """A minimal plugin for testing registration."""


class _ApplicablePlugin:
    """A plugin that implements is_applicable."""

    def __init__(self, *, accept: bool = True) -> None:
        self._accept = accept

    def is_applicable(self, properties: DataProperties) -> bool:
        """Return a fixed value for testing."""
        return self._accept


def test_register_and_get() -> None:
    """Registering a plugin makes it retrievable by name."""
    reg: Registry[_DummyPlugin] = Registry("test")

    @reg.register("alpha")
    class AlphaPlugin(_DummyPlugin): ...

    result = reg.get("alpha")
    assert isinstance(result, AlphaPlugin)


def test_register_duplicate_raises() -> None:
    """Registering the same name twice raises ValueError."""
    reg: Registry[_DummyPlugin] = Registry("test")

    @reg.register("dup")
    class FirstPlugin(_DummyPlugin): ...

    with pytest.raises(ValueError, match="already registered"):

        @reg.register("dup")
        class SecondPlugin(_DummyPlugin): ...


def test_get_missing_raises() -> None:
    """Looking up an unregistered name raises KeyError."""
    reg: Registry[_DummyPlugin] = Registry("test")
    with pytest.raises(KeyError, match="No test plugin"):
        reg.get("nonexistent")


def test_list_all() -> None:
    """list_all returns all registered plugins."""
    reg: Registry[_DummyPlugin] = Registry("test")

    @reg.register("a")
    class PluginA(_DummyPlugin): ...

    @reg.register("b")
    class PluginB(_DummyPlugin): ...

    all_plugins = reg.list_all()
    assert set(all_plugins.keys()) == {"a", "b"}
    assert isinstance(all_plugins["a"], PluginA)
    assert isinstance(all_plugins["b"], PluginB)


def test_list_all_returns_copy() -> None:
    """list_all returns a copy, not a reference to internals."""
    reg: Registry[_DummyPlugin] = Registry("test")

    @reg.register("x")
    class PluginX(_DummyPlugin): ...

    copy = reg.list_all()
    copy["injected"] = _DummyPlugin()
    assert "injected" not in reg.list_all()


def _make_dummy_properties(n_groups: int = 2) -> DataProperties:
    return DataProperties(
        outcome_type_guess="continuous",
        n_groups=n_groups,
        group_sizes={"A": 10, "B": 10},
        normality={},
        all_groups_normal=True,
        missing={
            "outcome_missing": {"count": 0, "percentage": 0.0},
            "group_missing": {"count": 0, "percentage": 0.0},
            "association": {
                "test_used": "Chi-Square",
                "statistic": None,
                "p_value": None,
                "significant": None,
                "note": "Default values",
            },
        },
        outliers={},
        sampled=False,
    )


def test_get_applicable_filters_by_is_applicable() -> None:
    """get_applicable only returns plugins whose is_applicable returns True."""
    reg: Registry[_ApplicablePlugin] = Registry("test")

    # Manually register instances with different applicability
    reg._plugins["yes"] = _ApplicablePlugin(accept=True)
    reg._plugins["no"] = _ApplicablePlugin(accept=False)

    applicable = reg.get_applicable(_make_dummy_properties())
    assert "yes" in applicable
    assert "no" not in applicable


def test_get_applicable_includes_non_applicable_protocol() -> None:
    """Plugins without is_applicable are always included."""
    reg: Registry[_DummyPlugin] = Registry("test")

    @reg.register("always")
    class AlwaysPlugin(_DummyPlugin): ...

    applicable = reg.get_applicable(_make_dummy_properties())
    assert "always" in applicable


def test_kind_property() -> None:
    """Registry exposes its kind label."""
    reg: Registry[_DummyPlugin] = Registry("filter")
    assert reg.kind == "filter"


def test_get_applicable_intersect() -> None:
    """Test get_applicable_intersect returns common plugins or breaks early."""
    reg: Registry[_ApplicablePlugin] = Registry("test")

    class _CustomPlugin:
        def __init__(self, accept_n_groups: list[int]) -> None:
            self._accept_n_groups = accept_n_groups

        def is_applicable(self, properties: DataProperties) -> bool:
            return properties.n_groups in self._accept_n_groups

    reg._plugins["p1"] = _CustomPlugin([2])  # type: ignore[assignment]
    reg._plugins["p2"] = _CustomPlugin([3])  # type: ignore[assignment]
    reg._plugins["p3"] = _CustomPlugin([2, 3])  # type: ignore[assignment]

    # Map of column names to properties
    properties_map = {
        "col1": _make_dummy_properties(n_groups=2),
        "col2": _make_dummy_properties(n_groups=3),
    }

    # Intersect of {"p1", "p3"} and {"p2", "p3"} is {"p3"}
    intersect = reg.get_applicable_intersect(properties_map)
    assert set(intersect.keys()) == {"p3"}

    # Test early break where intersection is empty
    properties_map_empty = {
        "col1": _make_dummy_properties(n_groups=2),
        "col2": _make_dummy_properties(n_groups=3),
        "col3": _make_dummy_properties(n_groups=4),
    }
    intersect_empty = reg.get_applicable_intersect(properties_map_empty)
    assert intersect_empty == {}
