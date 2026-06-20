"""Generic plugin registry.

Provides a single ``Registry[T]`` class used by all four plugin types
(filters, statistical methods, plots, exporters). Plugins register
themselves via the ``@registry.register("name")`` decorator.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Applicable(Protocol):
    """Protocol for plugins that declare applicability based on data properties."""

    def is_applicable(self, **properties: Any) -> bool:
        """Return whether this plugin is applicable given data properties."""
        ...  # pragma: no cover


class Registry[T]:
    """A generic registry for named plugin instances.

    Usage::

        filter_registry: Registry[Filter] = Registry("filter")

        @filter_registry.register("numeric_range")
        class NumericRangeFilter(Filter):
            ...
    """

    def __init__(self, kind: str) -> None:
        """Initialize the registry.

        Args:
            kind: Human-readable label for the type of plugin (used in error messages).
        """
        self._kind = kind
        self._plugins: dict[str, T] = {}

    @property
    def kind(self) -> str:
        """Return the human-readable label for this registry's plugin type."""
        return self._kind

    def register(self, name: str) -> type[T] | Any:
        """Decorator that registers a plugin class under *name*.

        The class is instantiated with no arguments and stored.

        Args:
            name: Unique name for the plugin.

        Returns:
            A decorator that registers and returns the original class unchanged.

        Raises:
            ValueError: If *name* is already registered.
        """

        def decorator(cls: type[T]) -> type[T]:
            if name in self._plugins:
                msg = f"{self._kind} plugin {name!r} is already registered"
                raise ValueError(msg)
            self._plugins[name] = cls()
            return cls

        return decorator

    def get(self, name: str) -> T:
        """Look up a registered plugin by name.

        Args:
            name: The registered name.

        Returns:
            The plugin instance.

        Raises:
            KeyError: If *name* is not registered.
        """
        try:
            return self._plugins[name]
        except KeyError:
            msg = f"No {self._kind} plugin registered with name {name!r}"
            raise KeyError(msg) from None

    def list_all(self) -> dict[str, T]:
        """Return a copy of all registered plugins as ``{name: instance}``."""
        return dict(self._plugins)

    def get_applicable(self, **properties: Any) -> dict[str, T]:
        """Return plugins whose ``is_applicable`` returns ``True``.

        Plugins that don't implement the ``Applicable`` protocol are
        always included (they are assumed to be universally applicable).

        Args:
            **properties: Keyword arguments forwarded to each plugin's
                ``is_applicable`` method.

        Returns:
            A dict of ``{name: instance}`` for applicable plugins.
        """
        result: dict[str, T] = {}
        for name, plugin in self._plugins.items():
            if isinstance(plugin, Applicable):
                if plugin.is_applicable(**properties):
                    result[name] = plugin
            else:
                result[name] = plugin
        return result
