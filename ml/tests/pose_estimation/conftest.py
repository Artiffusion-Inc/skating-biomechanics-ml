"""Pytest configuration for pose_estimation tests."""

import sys
from pathlib import Path

# Ensure ml/src is on path before any test imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
