# SESSION HANDOFF

## AUTONOMOUS OVERNIGHT MODE - ACTIVE

Building EvalAssay to a complete, recruiter-grade v1 without further input.
Work continuously: increment -> gate -> commit targeted paths -> push -> next
increment. Do not pause between increments.

**NEXT ACTION:** the four ARC audits are DONE and written up. An MMLU pair is
running (`runs/_mmlu.log`, prints MMLU_ALLDONE when finished). When it lands,
add an MMLU section to `docs/FINDINGS.md` covering whether the position artifact
is finally charged, then run `python verify.py` in full and close out.

MMLU is the run that matters most. Its answer key is skewed, which is the
condition under which a positional preference becomes a real artifact rather
than a wash - so it tests the confirming half of a prediction the calibration
made and that ARC could only test negatively.

**Banked and written up already - do not re-derive** (all in `docs/FINDINGS.md`):

- Model-free layer on MMLU test (14,042 items): all four detectors establish a
  defect. Longest-option scores 28.3% against 25.0% chance; 105 items are exact
  duplicates; key sits at position 3 for 26.8%; 1.4 points recoverable from
  options alone. ARC-Easy and ARC-Challenge establish nothing, with MDEs given.
- Presentation effect, replicated: ARC-Easy 0.5520 -> 0.7440 (+19.2 points) and
  ARC-Challenge 0.3720 -> 0.5600 (+18.8), continuation scoring against labelled.
- Blind accuracy across all four ARC runs; established only on ARC-Challenge
  labelled at 0.3480 [0.2680, 0.4240] against chance 0.2508. Below chance under
  continuation scoring, which is the evidence that presentation manufactures the
  question-independent signal rather than merely revealing it.
- Measured false-positive rate: 4 of 150 clean corpora fired some detector
  (2.67%, nominal 1%); 0 of 60 charged an artifact against a clean model.

To relaunch a died run:

    python -m evalassay.cli audit data/<corpus>.jsonl       --model Qwen/Qwen2.5-0.5B-Instruct --style <cloze|labelled>       --items 250 --seed 7 --json runs/<tag>.json > runs/<tag>.txt

Each ARC run takes 40 to 75 minutes on CPU and MMLU longer. `data/*.jsonl`
exists and the model is in the local cache; nothing needs downloading.

## The gate (must be green before every commit)

```
python verify.py            # everything, including the calibration sweep
python verify.py --fast     # same, minus the calibration, while iterating
```

Or the pieces directly:

```
python -m ruff check src tests verify.py
python -m ruff format --check src tests verify.py
python -m mypy
python -m pytest -q
```

`mypy` runs strict over `src`, `tests` and `verify.py`. **A locally green tree is
not a green build** - check `gh run list` too.

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
9. [DONE] Comparison of two audits (`assay compare`), which refuses runs whose
   manifests disagree, plus a lossless JSON round-trip.
10. [IN PROGRESS] Measured findings. The model-free results and the four ARC
    runs are written up in `docs/FINDINGS.md`; the MMLU pair is the remainder.

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
- **Option permutation is inert under `cloze` scoring and only measurable under
  `labelled`.** Cloze scores each option as a continuation of a prompt that never
  contains the others, so position never reaches the model. Measured: 0 of 12
  items changed answer under rotation with cloze, 3 of 12 with labelled. The
  audit detects an intervention that changed no outcome anywhere and reports it
  as *inert against this backend*, which is not the same as *not established*.
- **The report only asserts an interpretation when the interval supports it.**
  Blind accuracy of 0.316 with interval [0.244, 0.392] against chance 0.250 was
  being narrated as "the model is not answering the question"; the interval
  includes chance, so that was unsupported.
- **Do not watch long jobs with a harness background task; they get killed.** The
  waiter armed to notice the audits finishing was stopped mid-run, while the
  audits themselves - launched with `nohup bash -c '...' &` - kept going, along
  with the shell driving the loop. Verified by process inspection after the kill:
  the detached tree survived and only the tracked task died. Poll from the
  recurring heartbeat instead of arming a watcher.
- **A locally green tree is not a green build.** The type check was pinned to the
  minimum supported Python, which made mypy read the installed third-party stubs
  under older language rules; modern numpy stubs need 3.12 syntax, so the job on
  the newest interpreter failed inside numpy's own stub file while the oldest
  passed. Local runs missed it because the local numpy was older than the one CI
  installs. **Check `gh run list` before believing a tree is healthy.**
- **A count is a discontinuous function, so it must not be compared against a
  bare zero.** Bootstrap p-values differed between machines by exactly one
  replicate, because many replicates land algebraically on zero and a different
  linear-algebra backend nudges them either side. The comparison uses the
  numerical tolerance. Byte-identical reruns are a same-machine promise and the
  documents now say so; the committed example is compared to a relative
  tolerance, with a second test proving the tolerance still catches real drift.
- **Tie-breaking folds the option order into its hash**, so an exact tie can
  resolve differently after a rotation and the inert detection is slightly
  blunted under continuation scoring. Bounded by the runs at exactly zero on
  ARC-Easy and 0.0011 on ARC-Challenge. A tie-break keyed on the option's own
  text would remove it - the obvious next change, deliberately not made while
  runs were in flight because it would de-sync findings from the code.
- **The repo is born clean and has been scanned:** no machine paths, no personal
  identifiers, no other project names, no agent files tracked, correct identity
  on author and committer, no AI trailers. It can be flipped public as-is.

## Repository conventions

- Private repository. Born clean, so it can be made public as-is with no
  history rewrite. Visibility is the owner's decision, never made here.
- Commit targeted paths only, never `git add -A`.
- Commit messages are plain ASCII and state the technical fact only.
- Linters are pinned exactly, so a linter release cannot turn a clean tree red.
