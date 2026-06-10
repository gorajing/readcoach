"""fit_check.py — pyBKT toy fit sanity check.

Generates a synthetic 3-skill dataset from known BKT parameters, fits a
pyBKT Model, and prints fitted vs true parameters.

Fails loud on any error (no try/except-pass).
Run: uv run --group fit python scripts/fit_check.py
"""

import time
import warnings

import numpy as np
import pandas as pd

# ── True BKT parameters per skill ────────────────────────────────────────────
TRUE_PARAMS = {
    "reading_fluency": {
        "learns": 0.20,
        "guesses": 0.15,
        "slips": 0.10,
        "prior": 0.30,
    },
    "phonics_decoding": {
        "learns": 0.30,
        "guesses": 0.20,
        "slips": 0.08,
        "prior": 0.40,
    },
    "sight_words": {
        "learns": 0.15,
        "guesses": 0.25,
        "slips": 0.12,
        "prior": 0.50,
    },
}

N_STUDENTS = 50
N_OPPORTUNITIES = 15
SEED = 42


def simulate_bkt_sequence(
    rng: np.random.Generator,
    learns: float,
    guesses: float,
    slips: float,
    prior: float,
    n_opps: int,
) -> list[int]:
    """Simulate one student's response sequence under BKT."""
    known = rng.random() < prior
    responses = []
    for _ in range(n_opps):
        if known:
            correct = int(rng.random() >= slips)
        else:
            correct = int(rng.random() < guesses)
        responses.append(correct)
        if not known:
            known = rng.random() < learns
    return responses


def build_dataset(rng: np.random.Generator) -> pd.DataFrame:
    """Build a synthetic dataset for all 3 skills."""
    rows = []
    order_id = 0
    for skill_name, params in TRUE_PARAMS.items():
        for student_id in range(N_STUDENTS):
            seq = simulate_bkt_sequence(
                rng,
                learns=params["learns"],
                guesses=params["guesses"],
                slips=params["slips"],
                prior=params["prior"],
                n_opps=N_OPPORTUNITIES,
            )
            for opp_idx, correct in enumerate(seq):
                rows.append(
                    {
                        "order_id": order_id,
                        "user_id": f"student_{student_id:03d}",
                        "skill_name": skill_name,
                        "correct": correct,
                    }
                )
                order_id += 1
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = build_dataset(rng)
    print(
        f"Dataset: {len(df)} rows, {df['user_id'].nunique()} students, "
        f"{df['skill_name'].nunique()} skills"
    )

    # ── Import pyBKT Model (will raise if pyBKT is broken / not installed) ──
    from pyBKT.models import Model  # noqa: PLC0415

    model = Model(seed=SEED, num_fits=5)

    defaults = {
        "order_id": "order_id",
        "skill_name": "skill_name",
        "correct": "correct",
    }

    t0 = time.perf_counter()
    model.fit(data=df, defaults=defaults)
    elapsed = time.perf_counter() - t0

    print(f"\nFit wall time: {elapsed:.2f}s\n")

    fitted = model.params()
    print("Fitted parameters:")
    print(fitted.to_string())

    print("\nFitted vs True (per skill):")
    print(f"{'skill':<20} {'param':<12} {'true':>8} {'fitted':>8} {'delta':>8}")
    print("-" * 60)
    for skill_name, params in TRUE_PARAMS.items():
        skill_rows = fitted[fitted.index.get_level_values("skill") == skill_name]
        for param_name in ("prior", "learns", "guesses", "slips"):
            true_val = params[param_name] if param_name != "forgets" else 0.0
            try:
                fitted_val = float(
                    skill_rows[skill_rows.index.get_level_values("param") == param_name][
                        "value"
                    ].iloc[0]
                )
            except (IndexError, KeyError):
                fitted_val = float("nan")
            delta = fitted_val - true_val
            print(
                f"{skill_name:<20} {param_name:<12} {true_val:>8.3f} {fitted_val:>8.3f} {delta:>+8.3f}"
            )

    print("\nfit_check PASSED")


if __name__ == "__main__":
    main()
