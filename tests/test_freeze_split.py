"""T5.3 — freeze_split.py: one-way, hash-locked dev/held-out split (red first).

These tests run the freeze machinery against a TEMP golden dir (the module's
paths are monkeypatched) so they never touch the committed
``evals/golden/holdout.lock`` and never trip the one-way refusal against the real
freeze.  They pin: split disjointness + coverage (every persona in both halves),
lock verify green, tamper -> loud, and the freeze REFUSAL on a second
(differing) run.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "freeze_split", _PROJECT_ROOT / "scripts" / "freeze_split.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["freeze_split"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """Run a fresh freeze into a temp golden dir; yield (mod, golden_dir)."""
    mod = _load_script()
    golden = tmp_path / "golden"
    monkeypatch.setattr(mod, "GOLDEN_DIR", golden)
    monkeypatch.setattr(mod, "DEV_FILE", golden / "persona_sessions_dev.jsonl")
    monkeypatch.setattr(mod, "HOLDOUT_FILE", golden / "persona_sessions_holdout.jsonl")
    monkeypatch.setattr(mod, "LOCK_FILE", golden / "holdout.lock")
    rc = mod.freeze()
    assert rc == 0, "initial freeze should succeed"
    return mod, golden


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_freeze_writes_three_artifacts(frozen):
    mod, golden = frozen
    assert mod.DEV_FILE.exists()
    assert mod.HOLDOUT_FILE.exists()
    assert mod.LOCK_FILE.exists()


def test_split_is_disjoint_and_total_matches(frozen):
    mod, golden = frozen
    dev = _read_jsonl(mod.DEV_FILE)
    holdout = _read_jsonl(mod.HOLDOUT_FILE)
    dev_ids = {d["id"] for d in dev}
    holdout_ids = {h["id"] for h in holdout}
    # Disjoint: no session is in both halves.
    assert dev_ids.isdisjoint(holdout_ids)
    # Cover the whole corpus exactly once.
    lock = json.loads(mod.LOCK_FILE.read_text())
    assert len(dev_ids) + len(holdout_ids) == lock["n_total"]
    assert len(dev_ids) == lock["n_dev"]
    assert len(holdout_ids) == lock["n_holdout"]
    # 50/50 within one item.
    assert abs(len(dev_ids) - len(holdout_ids)) <= 1


def test_every_persona_appears_in_both_halves(frozen):
    mod, golden = frozen
    dev = _read_jsonl(mod.DEV_FILE)
    holdout = _read_jsonl(mod.HOLDOUT_FILE)
    dev_personas = {d["persona_id"] for d in dev}
    holdout_personas = {h["persona_id"] for h in holdout}
    expected = {"emergent", "ell_profile", "dyslexic_profile"}
    assert dev_personas == expected
    assert holdout_personas == expected


def test_lock_records_seeds_and_per_persona_counts(frozen):
    mod, golden = frozen
    lock = json.loads(mod.LOCK_FILE.read_text())
    assert lock["generation_seed"] == mod.GENERATION_SEED
    assert lock["split_seed"] == mod.SPLIT_SEED
    assert "created_utc" in lock
    assert set(lock["per_persona_counts"]) == {"total", "dev", "holdout"}
    assert set(lock["files"]) == {
        mod.DEV_FILE.name,
        mod.HOLDOUT_FILE.name,
    }


def test_verify_passes_on_a_clean_freeze(frozen):
    mod, golden = frozen
    assert mod.verify() == 0


def test_verify_fails_loud_on_tamper(frozen):
    mod, golden = frozen
    # Mutate one byte of the held-out file -> hash mismatch -> non-zero.
    content = mod.HOLDOUT_FILE.read_text()
    mod.HOLDOUT_FILE.write_text(content + "\n")  # extra newline changes the hash
    assert mod.verify() == 1


def test_reproducible_refreeze_is_noop(frozen):
    mod, golden = frozen
    before = mod.LOCK_FILE.read_text()
    # A second freeze with the SAME seeds/corpus is identical -> no-op, returns 0.
    rc = mod.freeze()
    assert rc == 0
    # The lock file's hashes are unchanged (timestamp may differ but files match).
    after_files = json.loads(mod.LOCK_FILE.read_text())["files"]
    before_files = json.loads(before)["files"]
    assert after_files == before_files


def test_freeze_refuses_when_corpus_differs(frozen):
    mod, golden = frozen
    # Tamper the committed dev file so a fresh generation differs from disk.
    mod.DEV_FILE.write_text(mod.DEV_FILE.read_text() + "tampered\n")
    rc = mod.freeze()
    assert rc == 1, "freeze must REFUSE when the on-disk split differs from a fresh gen"
    # The tampered file is left as-is (the freeze did not overwrite it).
    assert mod.DEV_FILE.read_text().endswith("tampered\n")
