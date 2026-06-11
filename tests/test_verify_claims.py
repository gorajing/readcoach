"""Tests for the claim-ledger verifier (T7.1).

The ledger (``docs/claims.md``) is only honest if a regression in any claimed
number turns the verifier red. These tests pin three properties:

  1. ``scripts/verify_claims.py`` PASSES over the committed ledger (exit 0).
  2. Synthetic drift — flip one observed value in a COPY of the ledger — makes
     the verifier FAIL (exit 1) and name the drifting claim. This proves the
     check has teeth; a verifier that cannot fail is decoration.
  3. The new ``docs/claims.md`` passes the language-boundary firewall (no
     job-search / private language leaks into the public repo).
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_LEDGER = _PROJECT_ROOT / "docs" / "claims.md"


def _load_verifier():
    """Import scripts/verify_claims.py as a module (it is not a package)."""
    path = _PROJECT_ROOT / "scripts" / "verify_claims.py"
    spec = importlib.util.spec_from_file_location("verify_claims", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses defined/used inside resolve their
    # owning module via sys.modules (otherwise cls.__module__ lookups fail).
    sys.modules["verify_claims"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_verifier_passes_over_committed_ledger():
    """The verifier exits 0 over the real, committed ledger."""
    verify = _load_verifier()
    rc = verify.main(["--ledger", str(_LEDGER)])
    assert rc == 0, "verify_claims must pass over the committed docs/claims.md"


def test_synthetic_drift_fails(tmp_path: Path, capsys):
    """Flipping one observed value makes the verifier fail with exit 1.

    We target the WAIT-rate row (0.4352 → 0.9999); the recomputation reads the
    real artifact and the mismatch must surface as DRIFT.
    """
    verify = _load_verifier()
    original = _LEDGER.read_text(encoding="utf-8")
    assert "0.4352 (in band)" in original, "fixture assumes the WAIT-rate cell text"
    drifted = original.replace("0.4352 (in band)", "0.9999 (in band)")
    assert drifted != original, "drift injection did not change the ledger"

    drifted_path = tmp_path / "claims_drifted.md"
    drifted_path.write_text(drifted, encoding="utf-8")

    rc = verify.main(["--ledger", str(drifted_path)])
    assert rc == 1, "verifier must FAIL on a drifted observed value"

    out = capsys.readouterr().out
    assert "DRIFT" in out, "the failure must be reported as DRIFT"
    assert "WAIT rate" in out, "the drifting claim must be named in the failure"


def test_missing_check_for_rerun_row_fails(tmp_path: Path):
    """A RERUN row with no registered check is a hard failure.

    We add a brand-new RERUN row the registry knows nothing about; the verifier
    must refuse it (every RERUN claim must be mechanically checkable).
    """
    verify = _load_verifier()
    original = _LEDGER.read_text(encoding="utf-8")
    # Inject the rogue row INSIDE an existing table (right after a real row), so
    # it sits under a parsed header — otherwise the parser would (correctly)
    # ignore a stray line with no table header above it.
    anchor = "| The default rule is never reached"
    assert anchor in original, "fixture assumes the default-rule row exists"
    rogue = (
        "| A wholly unregistered RERUN claim with no check. | nothing | "
        "`true` | 42 | nowhere | RERUN |\n"
    )
    drifted = original.replace(anchor, rogue + anchor, 1)
    assert drifted != original

    rogue_path = tmp_path / "claims_rogue.md"
    rogue_path.write_text(drifted, encoding="utf-8")

    rc = verify.main(["--ledger", str(rogue_path)])
    assert rc == 1, "an unchecked RERUN row must fail the verifier"


def test_pipe_inside_code_span_is_not_a_column_break():
    """The parser must not split on ``|`` inside inline-code spans.

    Two ledger rows carry ``|fit − true|`` and ``|RMSE_naive − RMSE_soft|``;
    they must parse as single rows, not be dropped.
    """
    verify = _load_verifier()
    rows = verify.parse_ledger(_LEDGER)
    claims = [r.claim for r in rows]
    assert any("parameter recovery error is at or below 0.06" in c for c in claims), (
        "the BKT recovery row (carries a code-span pipe) must parse"
    )
    assert any("break-even is channel accuracy a = 0.90" in c for c in claims), (
        "the break-even row (carries a code-span pipe) must parse"
    )


# ---------------------------------------------------------------------------
# Language-boundary firewall over the new doc (no private language leaks).
# Mirrors tests/test_language_boundary.py's intent for docs/claims.md.
# ---------------------------------------------------------------------------
_BANNED: list[tuple[str, re.RegexFlag]] = [
    (r"\bEllo\b", re.RegexFlag(0)),
    (r"\bJD\b", re.RegexFlag(0)),
    (r"\boutreach\b", re.IGNORECASE),
    (r"\bcold[ -]email\b", re.IGNORECASE),
    (r"\bhiring\b", re.IGNORECASE),
    (r"\binterviews?\b", re.IGNORECASE),
]


@pytest.mark.parametrize("doc", [_LEDGER, _PROJECT_ROOT / "scripts" / "verify_claims.py"])
def test_new_artifacts_pass_language_boundary(doc: Path):
    """docs/claims.md and the verifier carry no banned private language."""
    text = doc.read_text(encoding="utf-8")
    for pat, flags in _BANNED:
        m = re.search(pat, text, flags)
        assert m is None, f"{doc.name}: banned pattern {pat!r} found: {m.group(0)!r}"
