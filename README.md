# EvalAssay

**How much of a reported benchmark score is actually capability?**

An *assay* is the test that determines how much of an ore is really the metal.
EvalAssay does that to a leaderboard number: it applies controlled, paired
interventions to a multiple-choice benchmark, attributes the resulting accuracy
loss among named artifacts, and refuses to report any attribution it cannot
establish.

The headline metric is **purity** — the fraction of a reported score that
survives the audit.

```
Reported                                      0.824
──────────────────────────────────────────────────────────────────────
  choices-only solvable        −0.061  [0.042, 0.080]   question hidden
  option-position bias         −0.048  [0.031, 0.065]   options rotated
  surface-form memorisation      —      not established  MDE 0.019
  weak distractors             −0.027  [0.019, 0.036]   distractor injected
──────────────────────────────────────────────────────────────────────
  Assayed capability                            0.688
  Purity                                        83.5%
```

> Layout of the intended final report. The instrument is built and verified;
> measured results against public benchmarks are being generated now. Numbers
> above are illustrative and are **not** a finding — see *Status*.

---

## Why this is not just another robustness harness

Perturbation-robustness testing is well-trodden ground. What is missing from it
is arithmetic honesty, and that is the entire contribution here.

**1. Artifacts overlap, so you cannot add them up.**
The usual approach measures each artifact on its own and prints the list.
Option-position bias and choices-only solvability are not disjoint: an item
answerable from position heuristics is often the same item answerable from the
options alone. Measured separately, both effects claim it; summed, the total
exceeds the real drop and the implied "true capability" is too low.

EvalAssay treats the artifacts as players in a cooperative game and attributes
the joint drop with **Shapley values**. Shapley is the unique attribution
satisfying efficiency, symmetry, dummy and additivity — and efficiency is the one
that matters: the shares sum to the observed joint drop *exactly*, by
construction. This is checked on every run and in the test suite.

**2. Attribution is affordable because it is linear.**
The Shapley value is a linear functional of the coalition values, so the whole
attribution is one fixed matrix applied to the vector of coalition accuracies.
Two consequences: a handful of artifacts means only a small coalition lattice to
evaluate, and the bootstrap over items becomes a *single matrix product*. No
model is called during interval estimation at all.

**3. Silence is a result.**
Every quantity passes a **default-deny gate** before it is reported: family-wise
adjusted significance at α = 0.01 (Holm step-down), an interval excluding zero,
and a minimum effect size. A component that fails is printed as *not
established*, with the reason, and is **never** deducted from the score. The
audit is deliberately generous to the model it audits, because a false artifact
claim is an accusation.

**4. Null results report what they could have seen.**
Instead of post-hoc power — which is just a restatement of the p-value —
unestablished components report the **minimum detectable effect** at the achieved
sample size. That answers the question a reader of a null result actually has.

**5. A run is reproducible or it says so.**
Every audit emits a manifest: corpus content hash, item count and order, scorer
identity, config hash, seed, and library versions. Option scoring uses exact
log-likelihood rather than generation, so the local backend is deterministic.
Hosted-API backends are recorded as `deterministic: false` — a report that claims
reproducibility it cannot deliver is worse than one that admits the gap.

**6. The benchmark gets audited too.**
A separate, **zero-inference** layer measures defects in the benchmark itself:
whether the answer key is guessable from the options alone, whether answer
positions are uniform, whether "pick the longest option" beats chance, and
whether items are near-duplicates. A benchmark whose key leaks through surface
form is defective regardless of which model is pointed at it. These findings
describe the corpus, not the model, so they sit outside the decomposition and
never move the assayed score.

## How it is proved to work

The instrument is calibrated against **synthetic models with dialled-in
artifacts of known magnitude**. A benchmark is generated with, say, exactly eight
points of position bias planted in it, and the pipeline must recover eight points
within its interval — and must stay silent on a clean control. The measured
false-positive rate on clean controls is reported, not asserted.

That is the difference between a script that prints plausible numbers and an
instrument with a known error characteristic.

## Status

| Component | State |
|---|---|
| Core types, content addressing | Built, tested |
| Statistics: exact McNemar, BCa bootstrap, Holm, MDE | Built, tested |
| Shapley attribution (all four axioms tested) | Built, tested |
| Default-deny gate | Built, tested |
| Corpus layer and benchmark loaders | In progress |
| Model-free pathology detectors | In progress |
| Interventions and scorer backends | In progress |
| Calibration harness | In progress |
| CLI, run manifest, measured findings | In progress |

Gate on every commit: `ruff check`, `ruff format --check`, `mypy` (strict, over
source *and* tests), `pytest`.

## Running it

```bash
python -m pip install -e ".[probe,dev]"
python -m pytest -q          # test suite
python -m mypy               # strict type check
```

No benchmark data ships with this repository. Loaders read datasets you have
obtained yourself, under whatever licence those datasets carry.

## Licence

Proprietary source-available: read it, run it, check it. See [LICENSE](LICENSE).
You may run the audit and the test suite to satisfy yourself that the claims here
hold, and you may publish what you find — including a refutation. A verification
right that excluded publishing a negative finding would be worthless, and this
project exists to argue that unfalsifiable measurements are not measurements.
