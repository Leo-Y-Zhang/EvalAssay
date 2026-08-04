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

### The same effect on three benchmarks

All three corpora were audited identically, changing only the presentation:

| Corpus | `cloze` | `labelled` | difference |
|---|---|---|---|
| ARC-Easy | 0.5520 | 0.7440 | **+0.1920** |
| ARC-Challenge | 0.3720 | 0.5600 | **+0.1880** |
| MMLU | 0.3120 | 0.4200 | **+0.1080** |

Three benchmarks, three positive gaps, between eleven and nineteen points. A
single such result would be a curiosity; the same sign and comparable magnitude
on three independent corpora make it a property of how this model is asked
rather than an accident of one item set.

The magnitude is not constant, and it should not be reported as though it were.
MMLU's gap is roughly half ARC's. What can be said is that the direction held
everywhere and the smallest gap was still eleven points.

### Every question-independent result belongs to a labelled run

Blind accuracy - the score with the question removed entirely - across all six
runs:

| Run | Blind accuracy | 99% interval | Chance | Verdict |
|---|---|---|---|---|
| ARC-Easy, `cloze` | 0.3160 | [0.2440, 0.3920] | 0.2498 | not established |
| ARC-Easy, `labelled` | 0.3240 | [0.2480, 0.4000] | 0.2498 | not established |
| ARC-Challenge, `cloze` | 0.2240 | [0.1560, 0.2920] | 0.2508 | not established |
| ARC-Challenge, `labelled` | **0.3480** | [0.2680, 0.4240] | 0.2508 | **established** |
| MMLU, `cloze` | 0.2600 | [0.1880, 0.3320] | 0.2500 | not established |
| MMLU, `labelled` | **0.3720** | [0.2920, 0.4480] | 0.2500 | **established** |

**Both runs in which the model demonstrably beats chance without a question are
labelled runs. No continuation-scored run clears chance on any corpus** - the
three sit at 0.3160, 0.2240 and 0.2600 against floors near 0.25, one of them
below chance.

There is a mechanism rather than a coincidence. Under `cloze` each option is
scored alone against a prompt containing no question and no other option, so
there is nothing to compare and nothing to exploit. Under `labelled` the options
appear together and a model can pick whichever most resembles an answer.
**Presentation does not merely inflate a score; it manufactures the part of the
score that does not need the question.**

The size of that effect is not uniform either. On ARC-Challenge and MMLU the
blind score is 12.4 and 11.2 points higher under labelled presentation; on
ARC-Easy the two are within a point of each other. So the clean claim is the one
about the verdicts - every established question-independent result is a labelled
one - not a claim of a constant gap.

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

### MMLU: the question is worth 4.8 points, and that is inside the noise

**Qwen2.5-0.5B-Instruct on MMLU**, 250 items stratified across subjects, seed 7.

| | |
|---|---|
| Reported score | 0.4200 |
| Blind accuracy (question removed) | **0.3720** [0.2920, 0.4480] |
| Chance | 0.2500 |
| What the question adds | **+0.0480**, paired p 0.2007 |

Two statements hold at once, and both are established at the stated thresholds:

- The model scores **well above chance with no question at all** - the blind
  interval excludes chance, by 12.2 points.
- Removing the question **does not significantly change its accuracy** - the
  paired test gives p = 0.20, so at this sample size the 4.8 point difference
  cannot be told from zero.

On MMLU, for this model, almost the whole score is available without reading the
question. That is a stronger statement than anything the artifact decomposition
found, and it comes from the diagnostic that deliberately sits *outside* the
decomposition, which is the argument for measuring it separately rather than
folding it in.

It is not a claim that the question is irrelevant in general: p = 0.20 means
*not distinguished from zero here*, not *shown to be zero*. A larger sample could
separate 4.8 points comfortably.

### The position prediction could not be tested at this sample size

MMLU was chosen because its answer key is skewed, which is the condition under
which a positional preference becomes a chargeable artifact. The audit charged
nothing, and the reason is instructive rather than disappointing.

On the **full** 14,042-item test split the key skew is established at **0.0180**.
On the **250-item sample** actually audited, the position-skew detector reports a
minimum detectable effect of **0.0677** - nearly four times the true effect - so
the sample cannot see the skew that the full set establishes. The model-side
permutation share has an MDE of **0.0555**, and no positional preference can
convert a 1.8 point key skew into more than 1.8 points of accuracy.

**The artifact is therefore below the detection threshold by construction, not
by measurement.** Scaling the minimum detectable effect as one over the square
root of the sample size, testing this would need roughly **2,400 items** rather
than 250.

That is the honest outcome: a null with a quantitative reason and a stated
requirement for settling it, rather than a shrug. It is also a live demonstration
of why a null result is only worth reading next to its MDE - the same detector
establishes the defect on the full corpus and cannot see it on a sample.

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

## 4. Two models, 1.6 points apart or 46, depending only on how you ask

Auditing a second model turned the presentation effect from a quantity into a
ranking problem.

| Model | `cloze` | `labelled` | gap |
|---|---|---|---|
| SmolLM2-135M-Instruct | 0.5360 | 0.2840 | **-0.2520** |
| Qwen2.5-0.5B-Instruct | 0.5520 | 0.7440 | **+0.1920** |
| **distance between them** | **0.0160** | **0.4600** | |

Same 250 ARC-Easy items, same seed, same thresholds, identical corpus hash on
every run.

**Under continuation scoring the two models are 1.6 points apart. Under labelled
multiple choice they are 46 points apart.** A leaderboard using the second
format would report one model as roughly two and a half times the other; a
leaderboard using the first would call them tied.

### The gap does not shrink with size, it changes sign

The smaller model is *hurt* by labelled presentation - 25.2 points worse - while
the larger one gains 19.2. That rules out the tidy story the sweep was designed
to test, in which presentation dependence is a small-model failing that washes
out with scale.

The likelier explanation is that labelled multiple choice tests two things at
once: knowing the answer, and being able to follow "reply with the letter". The
135M model scores **0.2840 against a chance floor of 0.2498** in that format -
it is at chance, not because it lacks the knowledge, but because it cannot work
the format. Score it by continuation, where no format compliance is required,
and it recovers to 0.5360, within two points of a model nearly four times its
size.

So the format under-reports models that cannot follow it and rewards models that
can exploit it, and those two errors point in opposite directions. This is the
cloze-versus-multiple-choice distinction known in the evaluation literature; what
is added here is a measured, reproducible instance of it changing a *ranking*
rather than a score.

### The first artifact ever charged in a real run

Every earlier audit came back at 100% purity. This one did not:

```
  stronger_distractor         -0.0740  [0.0120, 0.1380]  charged
  Assayed capability                            0.4620
  Purity                                        86.2%
```

**7.4 points of SmolLM2-135M's continuation-scored 0.5360 came from the wrong
options being weak**, established at the pre-registered threshold. Replacing one
distractor with a plausible statement from another item takes it away. That is
the decomposition doing the job it was built for, on a real model, after five
audits in which it correctly charged nothing.

### A caveat on the 0.5B numbers in that table

They come from the previous night's runs, made before tie-breaking was changed
to be order-invariant. The change cannot have affected the continuation run,
where permutation was reported fully inert with a minimum detectable effect of
exactly zero, meaning no item's outcome moved at all. The labelled run is not
guaranteed untouched, so both are being re-run under current code and this
section will be corrected if they move.

---

## 5. What is not claimed

- No statement here is about training-data contamination. Sensitivity to exact
  wording is one symptom of it, but establishing contamination needs access to
  the training corpus.
- The MMLU defects are properties of the published test split as distributed.
  They say nothing about the intentions of anyone who built it; benchmarks
  acquire defects through ordinary drift and ordinary deadlines.
- Three of the four numbers can only understate: choices-only leakage, position
  skew, and the fuzzy half of the duplicate count. See `docs/LIMITS.md`.
