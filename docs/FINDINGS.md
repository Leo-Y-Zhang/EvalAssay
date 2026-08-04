# Measured findings

Everything below was produced by the code in this repository, at the stated
settings, and every result carries the command that reproduces it. Where a
measurement is a bound rather than an estimate, it says so.

Thresholds throughout are the defaults: family-wise `alpha = 0.01` with Holm
correction, intervals at the same level, and a minimum reportable effect of
0.005. Nothing that failed those thresholds is reported as a number.

---

## 1. Benchmark defects, measured without any model

The model-free layer needs no compute. These are statements about the
benchmarks themselves.

### MMLU (test split, 14,042 items)

All four detectors establish a defect.

| Detector | Effect | 99% interval | What it means |
|---|---|---|---|
| `longest_answer` | **+0.0326** | [0.0232, 0.0420] | picking the longest option scores 28.3% against 25.0% chance |
| `near_duplicate` | **+0.0160** | [0.0135, 0.0190] | 225 of 14,042 items are repeats or near-repeats |
| `position_skew` | **+0.0180** | [0.0092, 0.0282] | the answer key is not uniform over positions |
| `choices_only` | **+0.0144** | [0.0052, 0.0241] | the key is partly recoverable from the options alone |

Reproduce:

```bash
assay convert <mmlu-test.parquet> mmlu-test.jsonl --format mmlu-parquet
assay pathology mmlu-test.jsonl --seed 7
```

**The longest-option result is the one worth sitting with.** A procedure that
reads nothing — not the question, not the meaning of the options, only their
character length — scores **3.3 points above chance** on MMLU. Any model with a
mild preference for longer completions collects some of that, and on a
leaderboard it is indistinguishable from knowing the answer.

**105 items are exact duplicates** of another item, byte-identical after
normalisation. They were withheld from the other three detectors before those
ran, because a repeat is not a second observation.

**Position skew** puts the key at position 3 for 26.8% of items against 25.0%
expected. Small, but it is free accuracy for a model that prefers that slot —
and, as the calibration in `docs/METHOD.md` shows, positional preference is only
an artifact *because* a skew like this exists to exploit.

**Choices-only leakage is a lower bound.** The probe is a naive Bayes model over
option tokens under grouped cross-validation. That it recovers 1.4 points above
chance proves leakage exists; a stronger probe could only find more.

### ARC-Easy (validation, 570 items) and ARC-Challenge (validation, 299 items)

**No defect established on either, by any detector.** This is a clean bill of
health, and it is worth as much as the MMLU result — a tool that only ever finds
problems is not measuring anything.

A null result is only informative alongside what it could have seen:

| Detector | MDE, ARC-Easy | MDE, ARC-Challenge |
|---|---|---|
| `position_skew` | 0.0559 | 0.0765 |
| `longest_answer` | 0.0540 | 0.0885 |
| `choices_only` | 0.0620 | 0.0917 |
| `near_duplicate` | 0.0076 | 0.0144 |

So ARC-Easy carries no positional skew above about five and a half points, and
no duplicate rate above about three quarters of a point. Effects smaller than
that would not have shown at these sample sizes, and this says nothing about
them.

```bash
assay pathology arc-easy.jsonl --seed 7
assay pathology arc-challenge.jsonl --seed 7
```

### A finding that was wrong, and how it was caught

The first version of the duplicate detector reported **53 pairs in MMLU that
pose the same question while disagreeing about the answer** — a serious charge,
since no model can score full marks on both halves of such a pair.

Inspecting them showed it was wrong. One pair:

```
Statement 1 | The function f must necessarily be injective ...
Statement 1 | The function g must necessarily be injective ...
```

Different questions about different functions, with correctly different keys.
They scored a Jaccard above 0.9 because similarity was measured between token
*sets*, which discard word order and multiplicity — and both questions name both
`f` and `g` in their shared setup.

The measure was replaced with token shingles, which keep local order, and the
contradiction claim was moved off similarity entirely: two items now contradict
only when a digest of their question and options matches exactly and their keys
differ. **The corrected count is zero.** MMLU repeats itself 105 times but never
contradicts itself.

The episode is recorded here rather than quietly fixed because it is the clearest
illustration of why this project's output is gated: a plausible, well-formed,
serious-sounding finding was false, and only looking at the underlying items
revealed it.

---

## 2. Instrument calibration

Before any statement about a model is worth reading, the instrument has to be
shown to recover effects that are there and stay silent about ones that are not.
Neither can be checked against a real model, so both are checked against
simulated models with artifacts dialled in at known magnitudes.

Run it yourself in seconds, with no model and no dataset:

```bash
assay demo                      # two artifacts planted, one absent
assay demo --memorisation 0 --distractor 0   # nothing planted
python verify.py                # the whole battery
```

| Planted | Charged to | Recovered | Planted value in interval |
|---|---|---|---|
| memorisation 0.30 | `neutral_reframing` | +0.1375 | yes |
| distractor reliance 0.30 | `stronger_distractor` | +0.1302 | yes |
| positional preference 0.30, skewed key | `permute_options` | +0.0790 | yes |
| nothing (pure skill) | — | nothing charged | — |
| nothing (inert guesser) | — | nothing charged | — |

Each planted value is derived in closed form from the simulated model's own
mechanism and asserted to fall inside the interval the audit reports. The
Shapley shares sum to the observed drop exactly on every run.

### The measured false-positive rate

Silence is only worth anything if it is quantified. Across **150 clean synthetic
corpora** of 500 items each, at a nominal family-wise alpha of 0.01:

| Detector | Fired on | Rate | 99% interval |
|---|---|---|---|
| `position_skew` | 1 of 150 | 0.67% | [0.08%, 5.43%] |
| `longest_answer` | 0 of 150 | 0.00% | [0.00%, 4.24%] |
| `choices_only` | 3 of 150 | 2.00% | [0.51%, 7.56%] |
| `near_duplicate` | 0 of 150 | 0.00% | [0.00%, 4.24%] |
| **any detector** | 4 of 150 | **2.67%** | [0.80%, 8.55%] |

And across 60 clean corpora paired with a simulated model carrying **no
artifacts at all**, the model-side decomposition charged something on **0 of 60**
runs.

The family-wise interval contains the nominal 1%, so the deviation is not
statistically significant. But the point estimate is above nominal and
`choices_only` accounts for most of it, which has a mechanical explanation worth
stating: that detector's randomisation null treats per-item outcomes as
independent draws, while cross-validated predictions are correlated because the
folds share training data. The test is therefore mildly anti-conservative, and
its findings should be read as slightly weaker evidence than its p-value
suggests.

This is published rather than tuned away. An instrument's error characteristic
is a property to measure and report, not a number to make look good.

**Calibration produced a methodological result of its own.** A model with a
30% positional preference is charged *nothing* when the benchmark's key is
uniform — correctly, because always answering position one scores exactly chance
whether or not the options are rotated. Positional preference is only an
artifact in combination with a skewed benchmark. The model-side and
benchmark-side findings have to be read together, which is why this tool
produces both.

---

## 3. A model audited on a real benchmark

**Qwen2.5-0.5B-Instruct on ARC-Easy**, 250 items stratified from the validation
split, seed 7, thresholds at their defaults. The same model and the same items
were audited twice, changing only how the question is put.

| | `cloze` | `labelled` |
|---|---|---|
| Reported score | 0.5520 | **0.7440** |
| Assayed capability | 0.5520 | 0.7440 |
| Purity | 100% | 100% |
| Artifacts charged | none | none |
| Blind accuracy | 0.3160 [0.2440, 0.3920] | 0.3240 [0.2480, 0.4000] |

```bash
assay audit arc-easy.jsonl --model Qwen/Qwen2.5-0.5B-Instruct     --style cloze    --items 250 --seed 7 --json cloze.json
assay audit arc-easy.jsonl --model Qwen/Qwen2.5-0.5B-Instruct     --style labelled --items 250 --seed 7 --json labelled.json
assay compare cloze.json labelled.json
```

### Nineteen points of the score is presentation

`compare` confirms the two runs consumed an identical corpus under identical
thresholds - the manifests agree on both hashes - so the whole difference is
attributable to the one thing that changed:

```
                                baseline     variant    difference
reported score                    0.5520      0.7440       +0.1920
```

Under `cloze` each option is scored as a continuation and the model never sees
the options as a list. Under `labelled` they are presented as A to D and the
model scores the label. Nothing else differs. **The reported score moves 19.2
points**, which is wider than the gap between many models on a public
leaderboard.

Neither number is the right one. The point is that a benchmark score is not a
property of a model alone; it is a property of a model and a presentation, and
the presentation is rarely stated alongside the score.

### The decomposition charged nothing, and that is a limitation worth naming

Both runs came out at 100% purity. Read carelessly, "100% pure, twice" alongside
"the two scores differ by 19 points" looks like a contradiction. It is not, and
the reason matters:

**purity is a statement about the artifacts the audit looks for.** Presentation
is not one of the three players, so a difference that large can sit entirely
outside the decomposition. The comparison is what catches it. A purity figure
should never be read as "this score is 100% real" - only as "none of the three
artifacts tested for was established here".

### The prediction about position, tested in the right direction

The calibration said a positional preference only becomes an artifact when the
benchmark's key is also skewed. ARC-Easy is the case where it should therefore
find nothing, and it behaves exactly as predicted at each step:

- The model-free layer establishes **no position skew** on this corpus.
- Under `cloze`, `permute_options` is reported as **inert against this backend**
  rather than as not established: scoring options as continuations means
  position never reaches the model, so nothing could be measured.
- Under `labelled`, permutation is **not** inert - rotating options does change
  individual answers, measured separately at 3 of 12 items on a sample - yet its
  net share is not established, with an adjusted p of 1.00.

That is the predicted pattern: the model is position-sensitive, the benchmark is
not position-skewed, and so rotation gains and loses in equal measure and no
artifact exists to charge. The confirming half of the prediction requires a
benchmark whose key *is* skewed, which is why MMLU is the next corpus.

### An intervention that helped, and was still not charged

Under `cloze`, neutral reframing came out at **-0.0560**: prefixing the question
with a semantically empty frame made the model *more* accurate, by 5.6 points.
The gate refused it with "effect does not reduce accuracy; not charged".

This is the deliberate asymmetry doing its job. An intervention that helps is
reported, so a reader can see it, but it is never allowed to move the assayed
score - because the audit's output is a criticism, and an instrument that can
only understate the charge is the right one to reach for.

It is also a finding in its own right about small instruction-tuned models: this
one is sensitive to framing in a direction that flatters it.

---

## 4. What is not claimed

- No statement here is about training-data contamination. Sensitivity to exact
  wording is one symptom of it, but establishing contamination needs access to
  the training corpus.
- The MMLU defects are properties of the published test split as distributed.
  They say nothing about the intentions of anyone who built it; benchmarks
  acquire defects through ordinary drift and ordinary deadlines.
- Three of the four numbers can only understate: choices-only leakage, position
  skew, and the fuzzy half of the duplicate count. See `docs/LIMITS.md`.
