"""Hermetic test environment — zero credentials.

CONTRACT:
  Tests and CI must run with NO network credentials of any kind.  A test
  that genuinely needs a real key must set it explicitly in its own body
  (and must be marked to skip on CI).  Any developer's exported shell
  variables or .env sourcing must NOT leak into this suite.

ENFORCEMENT:
  We iterate os.environ *before* any project import and delete (not blank)
  every variable whose name starts with WANDB_, ANTHROPIC_, or GOOGLE_.
  Deletion is chosen over assignment to "" so that code using
  os.environ.get("KEY") and code using os.environ["KEY"] both hit
  the missing-key path consistently.

  WANDB_MODE is then forced to "disabled" as belt-and-suspenders: even if
  wandb itself re-sets something during import, it will not attempt a
  network call.
"""
import os

# Force-clear credential namespaces BEFORE any project import.
_PREFIXES = ("WANDB_", "ANTHROPIC_", "GOOGLE_")
for _key in list(os.environ):
    if _key.startswith(_PREFIXES):
        del os.environ[_key]

# Belt-and-suspenders: wandb fully offline regardless of library state.
os.environ["WANDB_MODE"] = "disabled"
