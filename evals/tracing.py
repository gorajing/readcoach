"""Weave tracing — scoped, optional, never imported on the disabled path.

T2.3: Weave scoped — tracing optional, benchmark path provably credless.

Public surface
--------------
weave_status()      -> str   "disabled:<reason>" or "enabled:<project>"
initialize_weave()  -> str   same as weave_status(); side-effects on enabled path
weave_doctor()      -> dict  diagnostic dict (booleans only — never leaks values)

Design notes
------------
* Lazy import: `import weave` is executed ONLY inside initialize_weave() when the
  enabled path is taken.  The disabled path (WEAVE_DISABLED=1 or missing key) never
  touches the weave package — proven by the sys.modules test in test_tracing.py.
* Fail loud: if weave.init() raises on the enabled path, the exception propagates
  uncaught.  A half-configured tracing setup must never silently degrade — internal
  evals require Weave.
* WEAVE_DISABLED=1 wins unconditionally over WANDB_API_KEY presence.
* weave_doctor() emits only boolean flags, never env values.
"""

from __future__ import annotations

import os


# ---------------------------------------------------------------------------
# Public: status
# ---------------------------------------------------------------------------

def weave_status() -> str:
    """Return "disabled:<reason>" or "enabled:<project>".

    Disabled when:
    - WEAVE_DISABLED env var is set to "1" (or any truthy non-empty value)
    - WANDB_API_KEY is absent or empty
    """
    if os.environ.get("WEAVE_DISABLED", "").strip():
        return "disabled:WEAVE_DISABLED"

    api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if not api_key:
        return "disabled:no_wandb_api_key"

    project = os.environ.get("WEAVE_PROJECT", "readcoach-evals").strip() or "readcoach-evals"
    return f"enabled:{project}"


# ---------------------------------------------------------------------------
# Public: initialize
# ---------------------------------------------------------------------------

def initialize_weave() -> str:
    """Initialize Weave when enabled; return the status string either way.

    On the disabled path this function never imports the weave package.
    On the enabled path it imports weave lazily and calls weave.init(project).
    If weave.init() raises, the exception propagates — no silent fallback.
    """
    status = weave_status()
    if status.startswith("disabled"):
        return status

    # Enabled path — lazy import: this line is never reached on the disabled path.
    import weave as _weave  # noqa: PLC0415

    project = status[len("enabled:"):]
    _weave.init(project)
    return status


# ---------------------------------------------------------------------------
# Public: doctor
# ---------------------------------------------------------------------------

def weave_doctor() -> dict:
    """Return a diagnostic dict for the current environment.

    CONTRACT: this dict contains ONLY booleans and strings that do NOT contain
    environment variable values.  Never leak API keys or secrets.

    Keys
    ----
    weave_disabled_flag     bool  — WEAVE_DISABLED is set to a truthy value
    wandb_api_key_present   bool  — WANDB_API_KEY is set and non-empty
    weave_project_set       bool  — WEAVE_PROJECT is set and non-empty
    weave_importable        bool  — the weave package can be imported
    status                  str   — result of weave_status()
    """
    weave_disabled_flag = bool(os.environ.get("WEAVE_DISABLED", "").strip())
    wandb_api_key_present = bool(os.environ.get("WANDB_API_KEY", "").strip())
    weave_project_set = bool(os.environ.get("WEAVE_PROJECT", "").strip())

    weave_importable: bool
    try:
        import weave as _  # noqa: F401, PLC0415
        weave_importable = True
    except ImportError:
        weave_importable = False

    return {
        "weave_disabled_flag": weave_disabled_flag,
        "wandb_api_key_present": wandb_api_key_present,
        "weave_project_set": weave_project_set,
        "weave_importable": weave_importable,
        "status": weave_status(),
    }
