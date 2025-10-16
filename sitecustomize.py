"""Ensure the repository root is available on sys.path during tests."""

import os
import sys

repo_root = os.path.dirname(__file__)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
