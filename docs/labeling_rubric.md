# ReadCoach — Human Labeling Rubric (judged dimensions)

> **Why this exists.** Deterministic scorers cover everything ground truth can
> reach (miscue identification/location, the policy's discrete move, the
> invariant checks). Three things ground truth *cannot* reach — whether a turn's
> **guidance** is good, whether it is **actionable** for the child, and the
> **ICAP engagement level** it invites — are graded by a cross-family LLM judge.
> That judge is only trustworthy if it agrees with humans, so we hand-label a set
> of turns and measure agreement (TPR/TNR + Cohen's κ with bootstrap CIs, per
> dimension) before any dimension is allowed to gate. **This document is the
> instruction sheet for the human labelers** whose labels are that ground truth.
>
> A turn to label is one row: the reader context (miscue, page position), the
> policy's chosen **move**, and the tutor's **utterance** (the line said to the
> child). You are scoring the *utterance given the move and context*.

## What you are NOT judging (ignore these)

- **Whether the move was correct.** The move (`WAIT`, `SCAFFOLDED_HINT`, …) is
  chosen by a separate, deterministic policy and audited elsewhere. Judge the
  *line*, not the decision. If `MODEL_THE_WORD` was the move, the tutor *should*
  reveal the word — that is not "giving away the answer."
- **Safety/invariant violations.** "Says wrong," mid-page coaching, emotional
  intimacy, missing AI reminders, re-serving completed items — all caught by the
  policy compiler as hard, gated failures. Do **not** fold them into these 1–5
  scores. If you see one, flag it separately; it is not a low score, it is a
  build break.
- **Transcription/ASR errors, audio quality, the child's reading accuracy.**
- **Length for its own sake.** A short line can be a 5; a long one can be a 2.
- **Your personal style preference.** Score against the anchors, not your taste.

## Scale

Every dimension is **1–5**, integers only (no half-points). Use the behavioral
anchors below. When a turn sits between two anchors, score the *lower* one — the
judge must clear a real bar, not a generous one. `WAIT` turns with an empty or
purely-holding utterance are scored **N/A** for Actionability and ICAP (there is
no guidance to act on); still score Guidance quality (a `WAIT` line can still be
warm/appropriate or jarring).

---

## Dimension 1 — Guidance quality

*Is this a warm, pedagogically sound thing to say to a 5–8-year-old at this
moment?* Tone, age-appropriateness, motivation-protection, and fit to the move.

| Score | Anchor |
|------:|--------|
| **5** | Warm, age-perfect, motivation-protective, and exactly fits the move. A skilled reading teacher would say this. |
| **4** | Good and appropriate; minor stiffness or a slightly-too-adult word, but lands well. |
| **3** | Serviceable but flat or generic ("Good job.") — not harmful, not memorable, doesn't quite fit the moment. |
| **2** | Off-tone for a young child (clinical, condescending, or over-effusive), or only loosely related to the move. |
| **1** | Cold, discouraging, confusing, or contradicts the move (e.g. lectures during a `WAIT`). |

**Worked examples** (move in brackets):

- *5* — [SCAFFOLDED_HINT, hint=bounce] "Ooh, tricky one! Let's bounce through it
  sound by sound — what's the very first sound you see?" — warm, invites the
  child in, fits a bounce hint exactly.
- *2* — [ENCOURAGE, page-end] "Your decoding accuracy on that page was
  satisfactory." — clinical, adult vocabulary, no warmth for a 6-year-old.

---

## Dimension 2 — Actionability

*Can the child actually DO something with this line right now?* A good hint gives
a concrete, child-sized next step; a poor one is vague encouragement that leaves
the reader no handhold. (Score N/A for a silent/holding `WAIT`.)

| Score | Anchor |
|------:|--------|
| **5** | Gives one clear, concrete, child-doable next action ("look at the first two letters and blend them"). The child knows exactly what to try. |
| **4** | Actionable, but the step is slightly broad or assumes a skill not yet cued ("sound it out"). |
| **3** | Points in a direction but leaves the *how* unspecified ("try again"). |
| **2** | Mostly affect with a faint nudge ("you can do it, keep looking!") — little to act on. |
| **1** | No actionable content, or the action is wrong for the move (asks the child to re-decode a word the move just modeled). |

**Worked examples:**

- *5* — [SCAFFOLDED_HINT, hint=phonetic] "The 'ea' in here makes one long-e
  sound, like in 'eat' — now try the whole word." — one concrete, scoped step.
- *1* — [SCAFFOLDED_HINT] "Don't worry, you've got this!" — pure affect; the
  child has nothing new to try.

---

## Dimension 3 — ICAP engagement level

ICAP (Chi & Wylie, 2014) ranks the cognitive engagement a prompt *invites*:
**Passive < Active < Constructive < Interactive.** Score how high up that ladder
the utterance pushes the child for this move. (Score N/A for a silent `WAIT`.)
This is about the engagement the line *invites*, not whether the child succeeds.

| Score | ICAP level invited | Anchor |
|------:|--------------------|--------|
| **5** | Interactive / strongly Constructive | Invites the child to generate, explain, or reason ("Why do you think the wolf did that?", "What sound does this part make, and how do you know?"). |
| **4** | Constructive | Prompts the child to produce something new — predict, infer, build the word from parts. |
| **3** | Active | Asks the child to *do* the focused thing (blend these sounds, reread this line) without generating new reasoning. |
| **2** | Passive-leaning | Mostly tells; the child receives rather than acts (a bare model with no invitation to try). |
| **1** | Passive / disengaging | No cognitive invitation at all, or shuts engagement down. |

> Note: a *correct* `MODEL_THE_WORD` line is **expected** to be lower on ICAP —
> the move's job is to hand over a stuck word so the child can continue. Score it
> on whether it models cleanly and re-invites reading ("That word is *brave* —
> keep going!" rents one Active beat back), not penalize it for not being
> Interactive. The move sets the ceiling; judge against that ceiling.

**Worked examples:**

- *5* — [COMPREHENSION_PROMPT, page-end] "What do you think happens next, and what
  in the story makes you think that?" — invites a constructed, evidence-based
  prediction (Constructive→Interactive).
- *2* — [MODEL_THE_WORD] "That word is *thought*." — models cleanly but adds no
  re-invitation to read on; receptive only.

---

## Logistics

- **Target set size: n = 60** labeled turns (the held-out judge-validation split).
  Drawn across the three reader profiles (struggling-decoder,
  fluent-but-hesitant, self-corrector) and across moves so no single move or
  profile dominates. The split is content-hash-locked before any A/B.
- **Two labelers per turn**, independently; disagreements of ≥2 on any dimension
  are adjudicated by a third. Inter-rater agreement (Cohen's κ) is reported per
  dimension alongside the judge's agreement.
- **Tie to the gate.** A judged dimension is admitted to gating **only** if the
  judge clears the per-dimension κ floor against these labels (with bootstrap
  CIs); dimensions below the floor are reported as *untrusted* and excluded from
  the gate (never silently dropped). This is why the labels must be careful: a
  noisy label set raises the κ bar the judge cannot legitimately clear, and a
  generous one lets a weak judge gate. Score against the anchors.
- The invariant violations metric (`invariants.violations`, gated at 0) is
  **separate** from these dimensions and is computed deterministically by the
  policy compiler — labelers never score safety.
