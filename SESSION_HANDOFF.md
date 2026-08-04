# SESSION HANDOFF

## AUTONOMOUS OVERNIGHT MODE - ACTIVE

Building EvalAssay to a complete, recruiter-grade v1 without further input.
Work continuously: increment -> gate -> commit targeted paths -> push -> next
increment. Do not pause between increments.

**NEXT ACTION:** the real audits are running in the background against ARC-Easy
and ARC-Challenge with Qwen2.5-0.5B-Instruct, writing to `runs/`. When they
finish: read the reports, write `docs/FINDINGS.md` and `docs/METHOD.md`, and
rewrite the README around the measured numbers. If the runs died, relaunch with
`python -m evalassay.cli audit data/arc-easy.jsonl --model Qwen/Qwen2.5-0.5B-Instruct
--items 250 --seed 7 --json runs/arc-easy.json`.

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
6. [DONE] Interventions and scorers, plus the audit engine (`audit.py`).
7. [DONE] Calibration harness (`tests/test_audit.py`, marked slow, run by CI):
   planted artifacts are recovered inside the audit's own intervals, and
   nothing is charged against a clean model or an inert guesser.
8. [DONE] Reporting (text + JSON), run manifest serialisation, and the `assay`
   CLI with a `demo` subcommand that needs no model, dataset or network.
9. [IN PROGRESS] The measured findings against a real benchmark, and the
   write-up. Audits of ARC-Easy and ARC-Challenge (250 items each,
   Qwen2.5-0.5B-Instruct, seed 7) are running and write to `runs/`.
   Remaining after they land: `docs/FINDINGS.md`, and a README rewritten
   around the measured numbers rather than around illustrative ones.

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
- **THREE Shapley players, not four. Hiding the question is NOT one of them.**
  The game attributes inflation, so every player must remove an artifact.
  Removing the question destroys the accuracy that *needed* the question, which
  is capability; charging it would invert the meaning of the report. Blind
  accuracy is measured separately and reported as a floor.
- **Positional preference is only an artifact on a skewed benchmark.** A model
  that always answers position one scores exactly chance with or without
  rotation when the key is uniform. Verified by calibration, and it is why the
  audit correctly charges nothing in that case.
- **Bootstrap p-values must treat near-zero effects as null.** An intervention
  that changes nothing produces shares that are algebraically zero but land on
  values like 1e-18, all one sign; counting the opposite tail then finds nothing
  and returns p near zero - a confident claim of a provably absent effect.
- **Scorers never read `answer_index`.** Enforced by a test that tampers with it
  and requires identical scores. The oracle locates the key by matching answer
  text, so it satisfies the same contract as the real backends.
- **Multiple-choice replies are parsed by token, never by scanning characters.**
  Character scanning made "The answer is B." resolve to A (from "answer") and
  made "I would rather not say" a confident answer rather than an abstention.
- **Model-side and benchmark-side findings are corrected as two separate
  families**, because they answer different questions; the combined error rate
  is therefore bounded by roughly the sum rather than by alpha, and METHOD.md
  says so.
- **The repo is born clean and has been scanned:** no machine paths, no personal
  identifiers, no other project names, no agent files tracked, correct identity
  on author and committer, no AI trailers. It can be flipped public as-is.

## Repository conventions

- Private repository. Born clean, so it can be made public as-is with no
  history rewrite. Visibility is the owner's decision, never made here.
- Commit targeted paths only, never `git add -A`.
- Commit messages are plain ASCII and state the technical fact only.
- Linters are pinned exactly, so a linter release cannot turn a clean tree red.
