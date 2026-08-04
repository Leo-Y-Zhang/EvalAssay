# SESSION HANDOFF

## AUTONOMOUS OVERNIGHT MODE - ACTIVE

Building EvalAssay to a complete, recruiter-grade v1 without further input.
Work continuously: increment -> gate -> commit targeted paths -> push -> next
increment. Do not pause between increments.

**NEXT ACTION:** build `src/evalassay/intervene/` and `src/evalassay/score/` -
the paired interventions (option permutation, hide-question, verified
meaning-preserving rewrite, distractor injection), the `Scorer` interface with
oracle / local / API backends, and the content-addressed score cache.

## The gate (must be green before every commit)

```
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy
python -m pytest -q
```

All four must pass. `mypy` runs in strict mode over `src` and `tests` both.

## What this project is

EvalAssay decomposes a reported multiple-choice benchmark score into measured
capability and measured artifacts. Artifacts are attributed with Shapley values,
so the shares sum exactly to the observed drop rather than double-counting
overlapping effects. Every attribution passes a default-deny gate: a component
that does not clear its pre-registered threshold is reported as not established
and contributes nothing.

The headline metric is **purity** - the fraction of a reported score that
survives the audit.

## Build order

1. [DONE] Scaffold: licence, gitignore, pyproject, CI, package skeleton.
2. [DONE] Core types (`types.py`) and content addressing (`hashing.py`).
3. [DONE] Statistics core (`stats/`): exact McNemar, BCa bootstrap, exact
   Shapley operator, Holm correction, default-deny gate, minimum detectable
   effect.
4. [DONE] Corpus layer: canonical format, synthetic generator with planted
   artifacts, benchmark loaders.
5. [DONE] Pathology detectors (model-free): choices-only solvability, position
   skew, longest-answer heuristic, near-duplicates. Each calibrated against
   planted defects and verified silent on clean corpora.
6. [NEXT] Interventions and scorers: option permutation, hide-question, verified
   meaning-preserving rewrite, distractor injection; oracle / local / API
   scorer backends behind one interface, with a content-addressed score cache.
7. Calibration harness: prove the instrument recovers planted effects within
   interval, and stays silent on clean controls. Measure the false-positive rate.
8. Reporting, run manifest, CLI, and the written-up measured findings.

## Design decisions already settled - do not relitigate

- **Shapley, not marginal effects.** Measuring each artifact alone and summing
  double-counts, because the artifacts overlap. Shapley's efficiency axiom makes
  the shares sum to the joint effect exactly. With a handful of players the
  lattice is small, and because the Shapley value is linear in the coalition
  values the whole attribution is one matrix applied to coalition accuracies -
  which also makes the bootstrap a single matrix product with no extra scoring.
- **Default-deny.** Alpha is 0.01, not 0.05, and a component must also clear a
  minimum effect size and have an interval excluding zero. Unestablished
  components are never charged, so the audit is deliberately generous to the
  model it audits.
- **MDE, not post-hoc power.** Null results report the smallest effect the
  sample size could have detected.
- **Exact log-likelihood scoring, not generation.** Option scoring by
  likelihood is deterministic and reproducible; sampling is not.
- **No benchmark data is vendored.** Loaders read data the user obtained
  themselves, under whatever licence it carries.
- **Escape sequences, not pasted characters,** for non-ASCII test fixtures.
  A normalising editor silently turned two Unicode tests into tautologies once;
  the guards in `tests/test_hashing.py` now detect that.
- **Detectors that assume independent items are handed a deduplicated corpus.**
  An exact repeat is not a second observation, and leaving repeats in shrinks the
  effective sample size without shrinking the nominal one. Measured, not
  theoretical: a corpus with a tenth of its items duplicated made the
  position-skew test report significance on a key that was in fact uniform.
- **Synthetic vocabulary words are all the same length.** The lexical marker is
  planted by substitution, so unequal lengths would make planted leakage
  register in the longest-answer detector too, and the two defects could not be
  calibrated apart. Padding likewise draws ordinary words rather than repeating
  one filler token, which the choices-only probe would learn instantly.
- **Duplicate candidates are indexed from rare tokens, not by excluding common
  ones.** The excluding version silently produced zero candidates on a
  small-vocabulary corpus and reported a corpus that was a tenth duplicates as
  clean.

## Repository conventions

- Private repository. Born clean, so it can be made public as-is with no
  history rewrite. Visibility is the owner's decision, never made here.
- Commit targeted paths only, never `git add -A`.
- Commit messages are plain ASCII and state the technical fact only.
- Linters are pinned exactly, so a linter release cannot turn a clean tree red.
