"""Top-level package for the SIERRA project."""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

__all__ = ["agent", "environment", "tests"]


def __getattr__(name: str) -> ModuleType:
    """Lazily import subpackages to provide convenient access.

    This keeps import times low while still allowing ``sierra.environment``
    style imports once the package is imported.  Attribute errors will be
    propagated in the usual way if the requested submodule does not exist.
    """

    if name in __all__:
        return import_module(f"sierra.{name}")
    raise AttributeError(f"module 'sierra' has no attribute '{name}'")


if TYPE_CHECKING:  # pragma: no cover - used only for static analyzers
    from . import agent, environment, tests  # noqa: F401
