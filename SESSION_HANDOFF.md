# SESSION HANDOFF

## STATUS: v1 COMPLETE AND PUBLIC. A model-size sweep is running.

The repository was made public on 2026-08-04 with the owner's authorisation,
after a full-history scan: no secrets across 39 commits, no machine paths, no
personal identifiers, no other repo names, correct identity on author and
committer throughout.

**IN FLIGHT:** `runs/size/_size.log`, printing SIZE_ALLDONE when done. Three
models - SmolLM2-135M, Qwen2.5-0.5B, Qwen2.5-1.5B - on ARC-Easy, 250 items,
seed 7, under both scoring styles. The question is whether presentation
dependence shrinks as models get larger. All three are re-run under current code
rather than reusing last night's 0.5B numbers, because the tie-break changed and
a cross-model comparison should come from one version.

Write each model's pair up as it lands:
`python -m evalassay.cli compare runs/size/<name>-cloze.json runs/size/<name>-labelled.json`

## v1, for reference

All six audits finished, all findings written up, `python verify.py` green on all
ten checks, continuous integration green, tree committed and pushed.

**Nothing is outstanding and nothing is running.** If you are resuming, there is
no work in flight to recover.

**The one thing needing the owner:** the repository is PRIVATE. It is born clean
- no machine paths, no personal identifiers, no other project names, no agent
files tracked, correct identity on author and committer, no AI trailers - so it
can be made public as-is with no history rewrite. That decision is the owner's
and was deliberately not made here.

**Fixed since v1:** tie-breaking now keys on each tied option's own text rather
than on the whole ordered option list, so it is order-invariant and the
permutation intervention can no longer see a positional artifact manufactured by
the instrument. The committed calibration example is unchanged, because the
oracle emits one-hot scores and so never ties.

## What was measured

Six audits: ARC-Easy, ARC-Challenge and MMLU, each under both scoring styles,
250 items apiece, Qwen2.5-0.5B-Instruct, seed 7. Reports in `runs/` (untracked),
write-up in `docs/FINDINGS.md`.

- **MMLU has all four benchmark defects**, measured with no model on the full
  14,042-item split: longest-option scores 28.3% against 25.0% chance, 105 items
  are exact duplicates, the key sits at position 3 for 26.8%, and 1.4 points are
  recoverable from the options alone. ARC establishes none, with MDEs given.
- **Presentation is worth 11 to 19 points** on every corpus tested.
- **Every question-independent result is a labelled run.** Both runs beating
  chance with no question are labelled; no continuation-scored run clears chance.
- **On MMLU the question adds 4.8 points, paired p 0.20** - not distinguished
  from zero at this sample size.
- **Measured false-positive rate:** 4 of 150 clean corpora, against a nominal 1%.
- **The position prediction could not be tested at n=250** and the reason is
  quantified: MMLU's key skew is 0.0180 while the sample gives the detector an
  MDE of 0.0677. Settling it needs about 2,400 items.

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
- **Type-check against the platform CI runs on, not just the local one.** A
  Windows-only call (`ctypes.windll`) passed mypy locally and failed on the
  Linux runner, because typeshed only declares it in the Windows-target stub.
  Two commits went red before anyone noticed. `verify.py` now runs
  `mypy --platform linux` as a separate check; measured, the same snippet passes
  under `--platform win32` and fails under `--platform linux`, so the gate would
  have caught it. **This is the second time the same class of error - platform
  or version dependent type checking - has turned CI red while local was
  green.**
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
