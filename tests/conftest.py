"""Make the src-layout package importable without an editable install.

An editable install (`pip install -e .`) is the supported path and is what the
reproducibility instructions in the README use; this shim only exists so the
test suite also runs on a bare checkout, which is how the judges are most
likely to try it first.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
