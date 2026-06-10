"""Tests for the speechocean762 loader.

Network-dependent test is marked ``@pytest.mark.network`` and deselected
by default (see ``pyproject.toml`` addopts).  CI never hits the network.
"""

from __future__ import annotations

import pytest

from readcoach.speechocean import (
    MISPRONOUNCED_MAX_ACCURACY,
    WordScore,
    prevalence_check,
)


# ---------------------------------------------------------------------------
# Unit tests — no network
# ---------------------------------------------------------------------------

def _ws(accuracy: int) -> WordScore:
    """Helper: build a WordScore with the given accuracy."""
    return WordScore(
        speaker="u0",
        word="test",
        accuracy=accuracy,
        mispronounced=(accuracy <= MISPRONOUNCED_MAX_ACCURACY),
    )


def test_prevalence_check_in_band():
    # 20% positive — well within [2%, 40%]
    scores = [_ws(3)] * 2 + [_ws(8)] * 8   # 2/10 = 0.20
    result = prevalence_check(scores)
    assert abs(result - 0.20) < 1e-9


def test_prevalence_check_out_of_band_raises():
    # 60% positive — outside the [2%, 40%] band
    scores = [_ws(2)] * 6 + [_ws(9)] * 4
    with pytest.raises(ValueError, match="outside the expected band"):
        prevalence_check(scores)


def test_prevalence_check_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        prevalence_check([])


def test_prevalence_check_boundary_low():
    # Exactly 2% positive (1 in 50) — just inside the band
    scores = [_ws(2)] * 1 + [_ws(9)] * 49
    result = prevalence_check(scores)
    assert abs(result - 0.02) < 1e-9


def test_prevalence_check_boundary_high():
    # Exactly 40% positive (2 in 5) — just inside the band
    scores = [_ws(1)] * 2 + [_ws(8)] * 3
    result = prevalence_check(scores)
    assert abs(result - 0.40) < 1e-9


def test_mispronounced_threshold_boundary():
    # score == MISPRONOUNCED_MAX_ACCURACY → mispronounced=True
    assert _ws(MISPRONOUNCED_MAX_ACCURACY).mispronounced is True
    # score == MISPRONOUNCED_MAX_ACCURACY + 1 → mispronounced=False
    assert _ws(MISPRONOUNCED_MAX_ACCURACY + 1).mispronounced is False


# ---------------------------------------------------------------------------
# Network smoke test — skipped by default, run with: pytest -m network
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_speechocean_smoke_network():
    """Live smoke test: stream up to 300 words from the test split.

    Validates:
    - At least one record is returned.
    - Every accuracy value is in [0, 10].
    - Prevalence is within [2%, 40%] (prevalence_check raises otherwise).

    Prints observed prevalence and word count so the number lands in the
    test report.
    """
    from readcoach.speechocean import iter_word_scores

    # Use the full test split (no limit) for a stable prevalence estimate.
    # Audio decode is disabled so this is metadata-only streaming; the test
    # split (~2500 utterances, ~15k words) finishes in under a minute.
    scores = list(iter_word_scores(split="test", limit=None))

    assert len(scores) >= 1, "Expected at least one WordScore from the test split"

    for ws in scores:
        assert 0 <= ws.accuracy <= 10, (
            f"accuracy {ws.accuracy} out of [0,10] for word {ws.word!r} "
            f"from speaker {ws.speaker!r}"
        )

    prevalence = prevalence_check(scores)

    n = len(scores)
    n_pos = sum(1 for ws in scores if ws.mispronounced)
    print(
        f"\n[speechocean762 smoke] n={n} words, "
        f"n_mispronounced={n_pos}, "
        f"prevalence={prevalence:.4f} ({prevalence*100:.1f}%)"
    )
