# EvalAssay

**How much of a reported benchmark score is actually capability?**

An *assay* is the test that determines how much of an ore is really the metal.
EvalAssay does that to a leaderboard number: it applies controlled, paired
interventions to a multiple-choice benchmark, attributes the resulting accuracy
loss among named artifacts, and refuses to report any attribution it cannot
establish.

The headline metric is **purity** — the fraction of a reported score that
survives the audit.

---

## A measured result, to begin with

Run against the **MMLU test split** (14,042 items), the model-free layer — which
loads no model at all — establishes all four defects it looks for:

| Detector | Effect | 99% interval | What it means |
|---|---|---|---|
| `longest_answer` | **+0.0326** | [0.0232, 0.0420] | picking the longest option scores **28.3%** against 25.0% chance |
| `near_duplicate` | **+0.0160** | [0.0135, 0.0190] | **105 items are byte-identical duplicates** of another item |
| `position_skew` | **+0.0180** | [0.0092, 0.0282] | the key sits at position 3 for 26.8% of items, not 25.0% |
| `choices_only` | **+0.0144** | [0.0052, 0.0241] | part of the key is recoverable from the options alone |

The first row is the one worth sitting with. A procedure that reads *nothing* —
not the question, not the meaning of the options, only their character length —
collects **3.3 points above chance** on the most widely used benchmark in the
field. Any model with a mild preference for longer completions banks some of
that, and on a leaderboard it looks exactly like knowing the answer.

The same layer establishes **nothing** on ARC-Easy or ARC-Challenge, and reports
what it could have seen if it were there. A tool that only ever finds problems is
not measuring anything.

That whole layer runs on **numpy and scipy alone** — no model, no GPU, no
machine-learning stack. Anyone can check a benchmark they were handed. The
heavier dependencies are optional and imported only inside the functions that
need them, which a test enforces by reading the source rather than trusting
whatever happens to be installed.

Full numbers, reproduce commands and caveats: **[docs/FINDINGS.md](docs/FINDINGS.md)**.

---

## What the report looks like

A real audit output — this exact file is committed at
[docs/example-report.txt](docs/example-report.txt) and a test regenerates it on
every run so it cannot go stale:

```
Reported score                                  0.7200
Chance (uniform guessing)                       0.2500
------------------------------------------------------------------------------
  stronger_distractor         -0.0493  [0.0318, 0.0725]  charged
  neutral_reframing           -0.1166  [0.0873, 0.1525]  charged
  permute_options                   -  not established (MDE 0.0445)
                                       adjusted p 0.9595 exceeds alpha 0.0100
------------------------------------------------------------------------------
  Assayed capability                            0.5541
  Purity (share of the score that survived)      77.0%
```

Note the third row. It does **not** print a number, because the audit could not
establish one. It prints the reason, and the smallest effect it could have
detected.

---

## Why this is not another robustness harness

Perturbation testing is well-trodden ground. What is missing from it is
arithmetic honesty, and that is the contribution here.

**1. Artifacts overlap, so you cannot add them up.**
The usual method measures each artifact alone and subtracts the list. That is
wrong: an item answerable by positional heuristics is often the same item
answerable from surface form alone, so measured separately both effects claim it
and the total exceeds the real drop. EvalAssay treats the interventions as
players in a cooperative game and splits the joint effect by **Shapley value**,
whose *efficiency* axiom makes the shares sum to the observed drop **exactly**.
Checked on every run.

**2. Exact attribution is affordable because it is linear.**
The Shapley value is a linear functional of the coalition accuracies, so the
whole attribution is one fixed matrix applied to a vector. A bootstrap replicate
is then a matrix product over a per-item matrix already in memory: **ten thousand
intervals cost zero model calls.**

**3. Silence is a result.**
Every quantity passes a default-deny gate — Holm-corrected significance at
α = 0.01, an interval excluding zero, and a minimum effect size — before it is
reported. A quantity that fails prints *not established*, with the reason, and is
never deducted. The audit is deliberately generous to the model it audits,
because a false artifact claim is an accusation.

**4. Null results say what they could have seen.**
Not post-hoc power, which is just a restatement of the p-value, but the
**minimum detectable effect** at the achieved sample size.

**5. A run is reproducible, or it says it isn't.**
Every audit emits a manifest: corpus hash, item count and order, scorer identity,
config hash, seed, library versions. Local scoring uses exact per-option
log-likelihood rather than generation, so there is no sampling and no answer
parsing, and two runs produce byte-identical JSON. Hosted-API backends are
recorded as `deterministic: false` and the report says so in as many words.

**6. An intervention that cannot bite says so.**
Scoring each option as a continuation never shows the model the option *list*, so
rotating the options cannot change any score and the permutation intervention is
structurally incapable of measuring anything — measured: 0 of 12 items changed
answer under rotation. The audit detects that the intervention changed no outcome
anywhere and reports it as **inert against this backend**, which is not the same
claim as *not established*. A `--style labelled` backend presents the options as a
list, which is how most leaderboard numbers are produced, and there 3 of the same
12 items did change.

**7. Hiding the question is deliberately *not* an artifact.**
Removing the question destroys the accuracy that *needed* the question — which is
capability. Charging it would invert the meaning of the report. It is measured
separately and reported as a floor: **blind accuracy**, the score with nothing to
answer.

---

## How the instrument is proved to work

Everything above rests on the audit recovering effects that are really there and
staying silent about ones that are not. Neither can be checked against a real
model, because with a real model nobody knows the right answer.

So it is calibrated against **simulated models with artifacts dialled in at known
magnitudes**, and each planted value — derived in closed form from the
simulation's own mechanism — must fall inside the interval the audit reports:

| Planted | Charged to | Recovered |
|---|---|---|
| memorisation 0.30 | `neutral_reframing` | +0.1375 ✓ |
| distractor reliance 0.30 | `stronger_distractor` | +0.1302 ✓ |
| positional preference 0.30 | `permute_options` | +0.0790 ✓ |
| nothing at all | — | nothing charged ✓ |

These run in continuous integration on every commit. A calibration that only runs
when someone remembers to ask is not a calibration.

**Calibration produced a result of its own:** a model with a strong positional
preference is charged *nothing* when the benchmark's key is uniform — correctly,
because always answering position one scores exactly chance whether or not the
options are rotated. Positional preference is only an artifact *in combination
with* a skewed benchmark. That is why this tool measures both sides.

---

## A finding that was wrong

The first version of the duplicate detector reported **53 MMLU pairs that pose
the same question while disagreeing about the answer** — a serious charge.
Inspection showed it was false. One pair:

```
Statement 1 | The function f must necessarily be injective ...
Statement 1 | The function g must necessarily be injective ...
```

Different questions about different functions, with correctly different keys.
They matched because similarity was measured between token *sets*, which discard
word order — and both questions name both `f` and `g` in their shared setup.

Similarity now uses token shingles, and the contradiction claim was moved off
similarity entirely: two items contradict only when a digest of their question
and options matches *exactly* and their keys differ. **The corrected count is
zero.**

It is written up in [docs/FINDINGS.md](docs/FINDINGS.md) rather than quietly
fixed, because it is the clearest illustration of why this project gates its own
output. The false finding was plausible, well-formed and serious-sounding, and
only looking at the underlying items revealed it.

---

## Try it in thirty seconds

No model, no dataset, no network:

```bash
python -m pip install -e ".[parquet,dev]"
python -m evalassay.cli demo        # or: assay demo
```

That audits a simulated model with artifacts planted at stated magnitudes, and
you can watch it charge what was planted and refuse what was not.

Then check every claim this README makes:

```bash
python verify.py
```

Each check names the claim it settles and the exit code is the answer. A claim
with no check beside it should be read as decoration.

Against a real benchmark and a real model:

```bash
assay convert data.parquet corpus.jsonl --format mmlu-parquet
assay pathology corpus.jsonl                       # no model needed
assay audit corpus.jsonl --model <hf-id> --items 250 --json report.json
```

No benchmark data ships with this repository. Loaders read datasets you obtained
yourself, under whatever licence they carry.

---

## What is built

| Component | State |
|---|---|
| Core types, content addressing, canonical corpus format | tested |
| Loaders: MMLU, ARC, HellaSwag, TruthfulQA (CSV, JSONL, parquet) | tested |
| Statistics: exact McNemar, BCa bootstrap, Holm, MDE | tested |
| Shapley attribution, all four axioms | tested |
| Default-deny gate | tested |
| Model-free detectors, calibrated against planted defects | tested |
| Interventions and the audit engine | tested |
| Scorers: oracle, local log-likelihood, hosted API | tested |
| Text and JSON reporting, run manifest, `assay` CLI | tested |

Gate on every commit: `ruff check`, `ruff format --check`, `mypy` (strict, over
source *and* tests), `pytest`, and the calibration sweep.

## What it does not do

Read **[docs/LIMITS.md](docs/LIMITS.md)** before quoting anything from here. It
lists what is not measured at all, which numbers can only understate, and the
ways a result could mislead. A tool that measures over-claiming has an obligation
not to over-claim.

Method in full: **[docs/METHOD.md](docs/METHOD.md)**.

## Licence

Proprietary source-available: read it, run it, check it. See [LICENSE](LICENSE).
You may run the audit and the test suite to satisfy yourself that the claims here
hold, and you may publish what you find — **including a refutation**. A
verification right that excluded publishing a negative finding would be
worthless, and this project exists to argue that unfalsifiable measurements are
not measurements.
