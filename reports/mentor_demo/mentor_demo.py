"""Reproducible Mentor Demonstration Runner.

Direct entry point located in reports/mentor_demo/ to rerun the fresh QSVM
and classical benchmark demonstration.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.mentor_demo import main

if __name__ == "__main__":
    main()
