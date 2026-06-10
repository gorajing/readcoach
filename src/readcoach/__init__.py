"""ReadCoach — an eval-first agent harness for an AI reading tutor.

Modules:
    asr            ASR layer (swappable; supports the target-text prior)
    miscue         miscue detection (align to target text, classify deviations)
    learner_model  per-skill mastery (pyBKT) + review scheduling (FSRS), in Redis
    tutor          the tutoring decision policy (when to intervene / how to help)

The eval flywheel lives in the top-level ``evals`` package.
See ``docs/ARCHITECTURE.md``.
"""

__version__ = "0.0.0"
