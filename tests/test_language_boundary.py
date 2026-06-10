"""Language-boundary test — the public/private firewall, in CI.

Scans every git-tracked *.md and *.py file for job-search-shaped language
that must never appear in this public repo.  Any violation fails the suite
with the file path, line number, and matched pattern.

NOTE: This file is explicitly excluded from its own scan (see SELF_PATH
below) because it necessarily contains the banned phrases as data.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Self-exclusion: this file must not be scanned because it contains the banned
# phrases as test data.  We use a path relative to the repo root.
# ---------------------------------------------------------------------------
SELF_PATH = Path(__file__).resolve().relative_to(
    Path(__file__).parent.parent.resolve()
).as_posix()

# ---------------------------------------------------------------------------
# Banned patterns.
# DOCTRINE: this list may GROW but must NEVER shrink.  Every pattern added
# here is a permanent commitment.  Removing or weakening a pattern defeats
# the firewall that keeps job-search language out of this public repo.
# ---------------------------------------------------------------------------
BANNED_PATTERNS: list[tuple[str, re.RegexFlag]] = [
    # Company name — case-SENSITIVE: "hello", "cello", "ello" must pass.
    (r"\bEllo\b", re.RegexFlag(0)),
    # Abbreviation — case-sensitive, word-bounded.
    (r"\bJD\b", re.RegexFlag(0)),
    # Job-search phrases — case-INSENSITIVE, word-bounded.
    # \b ensures "shiring", "outreaching" etc. do NOT match if they lack a
    # boundary — actually word-boundaries handle the starts/ends correctly.
    (r"\boutreach\b", re.IGNORECASE),
    (r"\bcold[ -]email\b", re.IGNORECASE),
    (r"\bhiring\b", re.IGNORECASE),
    (r"\binterviews?\b", re.IGNORECASE),
]

# Compile once for performance.
_COMPILED: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pat, flags), pat) for pat, flags in BANNED_PATTERNS
]

# ---------------------------------------------------------------------------
# Per-file allowlist (Day-7 mechanism).
#
# Maps a repo-relative file path → frozenset of pattern strings that are
# explicitly permitted in that file.  Ships EMPTY today.
#
# Day-7 mechanism: README.md may be allowlisted for 'Ello' in one conscious
# commit once the public announcement is ready; the pattern list itself never
# shrinks, so every other file stays protected through the end of the build.
# ---------------------------------------------------------------------------
ALLOWLIST: dict[str, frozenset[str]] = {}


def _get_tracked_files() -> list[Path]:
    """Return all git-tracked *.md and *.py files as Path objects."""
    repo_root = Path(__file__).parent.parent
    result = subprocess.run(
        ["git", "ls-files", "*.md", "*.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    paths = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped and stripped != SELF_PATH:
            paths.append(repo_root / stripped)
    return paths


def test_no_job_search_language() -> None:
    """Fail if any tracked *.md or *.py file contains job-search language."""
    violations: list[str] = []
    repo_root = Path(__file__).parent.parent

    files = _get_tracked_files()
    assert files, "No tracked *.md/*.py files found — the boundary scan would be vacuous"

    for filepath in files:
        rel = filepath.relative_to(repo_root).as_posix()
        allowed_for_file = ALLOWLIST.get(rel, frozenset())

        try:
            lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            violations.append(f"  {rel}: could not read file — {exc}")
            continue

        for lineno, line in enumerate(lines, start=1):
            for compiled, pat_str in _COMPILED:
                if pat_str in allowed_for_file:
                    continue
                if compiled.search(line):
                    violations.append(
                        f"  {rel}:{lineno}: pattern {pat_str!r} matched: {line.rstrip()!r}"
                    )

    assert not violations, (
        "Language-boundary violations found — job-search language must not appear "
        "in tracked files (see BANNED_PATTERNS in tests/test_language_boundary.py):\n"
        + "\n".join(violations)
    )
