"""Pytest configuration for the Sierra project.

Ensures that the project root is available on ``sys.path`` so imports of
``sierra`` resolve correctly during test discovery and execution.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
