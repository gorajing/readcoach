"""Weave doctor — print tracing diagnostics for the current environment.

Usage
-----
    uv run python scripts/weave_doctor.py

Output format: one "key: value" line per diagnostic field.  Values are
booleans only — this script never prints API keys or credential values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on the path when run directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evals.tracing import weave_doctor  # noqa: E402


def main() -> int:
    diag = weave_doctor()
    for key, value in diag.items():
        print(f"{key}: {json.dumps(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
