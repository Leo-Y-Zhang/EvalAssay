# Method

This document states, in full, what EvalAssay computes and why. It is written so
that a reader who disagrees with a choice can find the choice, see the reasoning,
and re-run the audit under their own settings.

---

## 1. The quantity being estimated

A reported multiple-choice benchmark score is a mixture. Some of it is the model
answering the question. Some of it is the model exploiting regularities that have
nothing to do with the question: a favoured option position, a remembered string,
distractors too weak to be tempting.

EvalAssay estimates how the reported score divides between those, and reports the
part that survives as the **assayed capability**. **Purity** is the assayed
capability as a fraction of the reported score.

## 2. Why the artifacts cannot simply be added up

The obvious method is to measure each artifact on its own and subtract the list.
It is wrong, and quietly so.

The artifacts overlap. An item answerable by positional heuristics is often the
same item answerable from surface form alone. Measured separately, both effects
claim that item. Summed, the total exceeds the real drop and the implied
capability is too low — which, for a tool whose output is a criticism, is a bias
in the worst possible direction.

## 3. The cooperative game

Let the interventions be players `N = {1, …, n}`. For a subset `S ⊆ N`, write
`acc(S)` for the model's accuracy when the interventions in `S` are applied
together. Define the value of a coalition as the accuracy it destroys:

```
v(S) = acc(∅) − acc(S),        v(∅) = 0
```

The **Shapley value** of player `i` is its average marginal contribution over all
orders in which the coalition could form:

```
φ_i = Σ_{S ⊆ N\{i}}  |S|! (n − |S| − 1)! / n!  ·  [ v(S ∪ {i}) − v(S) ]
```

Shapley is the unique attribution satisfying four axioms, and the one that
matters here is **efficiency**:

```
Σ_i φ_i = v(N)
```

The shares sum to the accuracy actually destroyed when everything is applied at
once — exactly, by construction, not approximately. This is checked on every run
and in `tests/test_stats_shapley.py`.

The other three axioms are worth stating because they are also properties anyone
would want: **symmetry** (two players who contribute identically to every
coalition receive equal shares), **dummy** (a player who changes nothing receives
zero), and **additivity**.

## 4. Why exact Shapley is affordable here

Two facts make the exact computation cheap.

First, the player set is small — three interventions, so eight coalitions.

Second, and more usefully, `φ` is a **linear functional** of the coalition
accuracies. Substituting `v(S) = acc(∅) − acc(S)` into the Shapley formula and
cancelling the `acc(∅)` terms leaves a fixed matrix `M` with

```
φ = M · acc
```

where `acc` is the vector of coalition accuracies indexed by bitmask. `M` depends
only on the number of players, so it is built once.

The consequence for intervals is the important one. A bootstrap replicate
resamples *items*, which changes `acc`, which is one matrix product away from
`φ`. **Ten thousand bootstrap replicates therefore cost zero model calls** — the
per-item correctness matrix is already in memory, and the whole resampling is a
single matrix multiplication, chunked to bound peak memory.

## 5. The interventions

Three players, each removing one way a score can be inflated:

| Player | What it removes |
|---|---|
| `permute_options` | advantage from the key sitting at a favoured position |
| `neutral_reframing` | advantage from recognising an exact string |
| `stronger_distractor` | advantage from the wrong options being weak |

Rules they all obey:

- **The option count never changes.** An intervention that added an option would
  lower accuracy mechanically, even for a uniform guesser, and the audit would
  charge that arithmetic to the model. The distractor intervention therefore
  *replaces* a wrong option rather than adding one.
- **Random choices are stable across coalitions.** Each intervention derives its
  generator from the run seed, its own name, and the item identifier — never from
  a shared stream consumed in coalition order. Without this, the same item would
  get different treatment in different coalitions, and the decomposition would be
  attributing noise.
- **Composition is order-free.** Coalitions are sets, so a fixed canonical order
  is applied whatever order the members are given in.

Option permutation places the key at *every* position and averages, rather than
sampling one placement. Sampling would leave positional noise in the estimate
that the audit would then have to separate from positional preference — the very
thing it is measuring.

### Why hiding the question is *not* a player

Removing the question destroys the accuracy that *needed* the question. That is
capability, not an artifact. Including it as a player would make the report claim
that understanding the question is a form of cheating.

It is measured instead as its own paired comparison and reported as a floor:
**blind accuracy**, the score with no question at all. A model well above chance
there is, on those items, not answering anything.

## 6. Interval estimation

Intervals are **bias-corrected and accelerated (BCa)** bootstrap intervals over
resampled items.

BCa is used rather than a plain percentile interval because it corrects for two
things that apply here: the bootstrap distribution sitting off-centre from the
point estimate, and the variance of the statistic changing with its value — both
common for bounded averages near the edge of their range, which is where a strong
model's accuracy lives.

Where the correction is undefined — every replicate identical, or a jackknife
with no spread — the code falls back to a percentile interval rather than
returning a degenerate BCa interval that looks like a result.

The resample is shared across coalitions within a draw, which preserves pairing:
an item weighted heavily in a draw is weighted heavily in *all* conditions of
that draw. Resampling each coalition independently would break the pairing and
inflate every interval.

## 7. Significance, and the default-deny gate

Each share gets a two-sided bootstrap p-value: the proportion of replicates
falling on the far side of zero from the point estimate, with a plus-one
correction so a finite resample can never license a claim of zero probability.

P-values are then corrected across the family with **Holm's step-down
procedure**, which controls the family-wise error rate under arbitrary dependence
— essential here, because the interventions are deliberately correlated — while
being uniformly more powerful than plain Bonferroni.

A share is reported as a deduction only if **all** of the following hold:

1. the Holm-adjusted p-value clears `alpha` (default **0.01**, not 0.05);
2. the confidence interval excludes zero;
3. the effect is at least `min_effect` (default 0.005) in magnitude;
4. the effect *reduces* accuracy.

Anything else is reported as **not established**, with the reason, and
contributes nothing. Condition 4 makes the audit deliberately asymmetric: an
intervention that helps the model is reported but never charged.

The asymmetry is the point. This tool's output is a criticism, and an instrument
that can only understate the charge is the right one to reach for.

### Minimum detectable effect, not post-hoc power

A null result reports the smallest effect the run could have detected at its
sample size, computed from the bootstrap standard error. Post-hoc power is a
deterministic function of the observed p-value and adds nothing; the MDE answers
the question a reader of a null result actually has.

## 8. The model-free layer

Four detectors measure defects in the *benchmark*, needing no model at all:
answer-position skew, the longest-option heuristic, answer-key leakage through
option surface form, and repeated items.

They sit outside the decomposition and never move the assayed score, because they
describe the ruler rather than the thing being measured.

Two details matter:

- **Detectors that assume independent items are given a deduplicated corpus.**
  An exact repeat is not a second observation; leaving repeats in shrinks the
  effective sample size without shrinking the nominal one, which makes the test
  anti-conservative. This was measured, not assumed: a corpus with a tenth of its
  items duplicated made the position-skew test report significance on a key that
  was in fact uniform.
- **The position-skew statistic is null-bias corrected.** Total variation from
  uniform is positive under sampling noise alone, so its expected value under the
  null is subtracted. The correction is exact under the null and conservative
  away from it, making the reported skew a lower bound.

## 9. Calibration

Every claim above rests on the audit recovering effects that are really there and
staying silent about ones that are not. Neither can be checked against a real
model, because with a real model nobody knows the right answer.

So the instrument is calibrated against **simulated models with artifacts dialled
in at known magnitudes**. `tests/test_audit.py` requires that:

- planted memorisation, distractor reliance and positional preference are each
  attributed to the corresponding intervention, with the closed-form planted
  value falling inside the interval the audit itself reports;
- a model with no artifacts, and an inert guesser, are charged nothing;
- the shares sum to the total drop on every run.

These are marked slow and selected by name in continuous integration, so they run
on every commit. A calibration that only runs when someone remembers to ask is
not a calibration.

**Calibration produced a result worth stating on its own:** positional preference
is only an artifact *in combination with a skewed benchmark*. A model that always
answers position one scores exactly chance whether or not the options are
rotated, when the benchmark's key is uniform. The audit correctly charges nothing
in that case, and the two facts have to be read together.

## 10. Reproducibility

Every run emits a manifest: corpus content hash, item count and order, scorer
identity, configuration hash, seed, and library versions.

Local scoring uses exact per-option log-likelihood rather than generation, so it
involves no sampling, no decoding parameters and no answer parsing. Two runs of
the same audit produce byte-identical JSON.

Hosted-API backends are recorded as `deterministic: false` and the report says so
in as many words. A report claiming reproducibility it cannot deliver would be
worse than one that states the gap.
