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

**Calibration produced a methodological result of its own.** A model with a
30% positional preference is charged *nothing* when the benchmark's key is
uniform — correctly, because always answering position one scores exactly chance
whether or not the options are rotated. Positional preference is only an
artifact in combination with a skewed benchmark. The model-side and
benchmark-side findings have to be read together, which is why this tool
produces both.

---

## 3. What is not claimed

- No statement here is about training-data contamination. Sensitivity to exact
  wording is one symptom of it, but establishing contamination needs access to
  the training corpus.
- The MMLU defects are properties of the published test split as distributed.
  They say nothing about the intentions of anyone who built it; benchmarks
  acquire defects through ordinary drift and ordinary deadlines.
- Three of the four numbers can only understate: choices-only leakage, position
  skew, and the fuzzy half of the duplicate count. See `docs/LIMITS.md`.
