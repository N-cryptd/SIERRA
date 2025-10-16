"""Top level package for the Sierra project.

This module ensures that the ``sierra`` directory is treated as a
package during test discovery.  It also exposes the most commonly used
submodules so ``from sierra.environment import core`` style imports keep
working when the package is installed in different environments.
"""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

__all__ = [
    "environment",
    "agent",
]


def __getattr__(name: str) -> ModuleType:
    """Dynamically expose known subpackages.

    Python will call this ``__getattr__`` when an attribute is not found
    on the module.  We forward the request to ``importlib`` so that
    ``sierra.environment`` and ``sierra.agent`` continue to work as
    expected while keeping the attribute list small.  ``ImportError`` is
    raised for unknown attributes to match the normal module behaviour.
    """

    if name in __all__:
        return import_module(f"sierra.{name}")
    raise AttributeError(f"module 'sierra' has no attribute {name!r}")


if TYPE_CHECKING:
    # Provide type-checker visibility for the subpackages without
    # importing them at runtime unless explicitly requested.
    from . import environment, agent  # noqa: F401
