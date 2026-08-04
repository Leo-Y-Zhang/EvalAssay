# Limits

A tool that measures over-claiming has an obligation not to over-claim. This
document lists what EvalAssay does not measure, where its numbers are bounds
rather than estimates, and the ways a result here could mislead.

None of this is in an appendix by accident. Anything below that would change how
a reader interprets a number is repeated in the report itself.

---

## What the assayed capability is not

**It is not a corrected benchmark score.** It is the reported score with the
artifact contributions this audit could *establish* removed. Artifacts it did not
look for are still in it, and so are artifacts it looked for but could not
establish at the sample size used.

**It is an upper bound on capability, not an estimate of it.** Everything the
gate refuses is treated as absent. That is deliberate — the audit is generous to
the model it audits, because a false artifact claim is an accusation — but it
means the real capability is at most the assayed figure and could be lower.

**It is specific to the corpus, the model, and the prompt format.** A different
prompt template can move an accuracy by several points on its own. The manifest
records the exact configuration; two assayed scores are comparable only if their
manifests agree.

## Where the numbers are lower bounds

Three quantities can only understate:

- **Neutral reframing** applies a semantically empty prefix and a quote-style
  swap. It detects only memorisation sensitive to the exact string. A genuine
  paraphrase would find far more — but generating one needs a model in the loop,
  which would make the intervention non-deterministic and put an unverified claim
  (that meaning was preserved) underneath every number in the report. A weak,
  provably meaning-preserving rewrite gives a checkable bound. A strong,
  unverifiable one would give a number nobody could check.

- **The choices-only probe** is a naive Bayes model over option tokens. A probe
  that finds leakage proves leakage exists; a probe that finds none proves only
  that *this* probe found none. A stronger probe can raise the number and can
  never lower it.

- **Position skew** subtracts the statistic's expected value under the uniform
  null. The correction is exact under the null and conservative away from it, so
  a large true skew is reported slightly smaller than it is.

## An intervention that cannot bite against the local backend

**Option permutation measures nothing when scoring by log-likelihood.** Each
option is scored independently as a continuation of the same prompt, so the
options are never presented to the model as a list and rotating them cannot
change any option's score. The argmax lands on the same text whichever position
it occupies.

This is not a defect in the scorer — it is a genuine property, and arguably a
desirable one: likelihood scoring is *structurally immune* to option-position
effects. Positional preference is a phenomenon of prompted multiple choice,
where the options appear as a labelled list, which is what the hosted-API
backend does and what most published leaderboard numbers are produced by.

The audit detects this and says so. An intervention whose presence changed no
outcome on any item is reported as **inert against this backend**, not as "not
established", because those mean different things: one is a measurement that
came back small, the other is a measurement that could not be taken. Read a
purity figure from the local backend as covering two artifacts, not three.

## What is not measured at all

- **Training-data contamination.** The audit can detect sensitivity to exact
  wording, which is one symptom, but it cannot tell whether an item appeared in
  training. Establishing that needs access to the training corpus.
- **Reasoning quality.** Only whether the key was selected, never how.
- **Anything but multiple choice.** Free-response, code, and open-ended
  generation are out of scope. The interventions all depend on there being a
  fixed option set.
- **Calibration or confidence.** Only argmax correctness is used.
- **Harms, bias, or safety properties.** Different tools, different questions.

## Threats to validity

**Interventions can interact with the corpus, not just the model.** Positional
preference is the clearest case: it buys a model nothing on a benchmark whose key
is uniform, so a null result there means "no artifact *on this benchmark*", not
"no positional preference". The model-free layer is what tells you which
situation you are in, and the two must be read together.

**Stronger distractors change difficulty, not only weakness.** The replacement is
another item's correct answer, which is well-formed, on-topic and wrong here.
That is intended. But an item can become genuinely harder rather than merely
less exploitable, and the audit cannot separate those. The share is best read as
"accuracy that did not survive a harder alternative", which is slightly broader
than "accuracy that came from weak distractors".

**Some items are unaffected by an intervention.** The distractor swap is inert
where every candidate replacement already appears among the options. Those items
contribute zero to that share, which is correct, but it means the effective
sample size for that player is smaller than the item count.

**The choices-only probe is mildly anti-conservative.** Its randomisation null
treats per-item outcomes as independent draws at chance, but cross-validated
predictions are correlated because the folds share training data. Measured over
150 clean corpora it fired on 2.0% of them against a nominal 1%, which is not a
statistically significant excess but is the largest contributor to the
family-wise rate of 2.7%. Treat a marginal finding from this detector as weaker
evidence than its p-value implies.

**Byte-identical reruns are a same-machine promise.** Log-likelihood scoring is
deterministic and the seeds fix every draw, so a rerun on the same machine with
the same library versions reproduces the JSON exactly. Across machines, a
different linear-algebra backend sums the same matrix product in a different
order, and quantities that are *counts* - the bootstrap p-values - can move
because a replicate sitting algebraically on zero lands either side of it. The
boundary comparison uses a tolerance for exactly this reason, which removes the
common case, but the general guarantee is same-machine. The manifest records
library versions so two results can be checked for comparability at that level.

**Bootstrap p-values are approximations.** They are coherent with the intervals
they accompany and corrected for multiplicity, but they are not exact tests. The
gate's requirement that the interval also exclude zero exists partly to stop a
single approximate quantity carrying a finding on its own.

**Chance accuracy is not always a quarter.** Corpora with ragged option counts
have a per-item chance baseline, which the audit uses; a reader comparing against
a remembered "25%" may misread a result on such a corpus.

## Where a result could mislead

**"Not established" does not mean "refuted".** It means the audit looked and
could not establish the effect at this sample size and threshold. The reported
minimum detectable effect says how large the effect would have needed to be. The
JSON keeps verdicts as words rather than booleans specifically so this
distinction survives being machine-read.

**A skipped detector is not a passed detector.** Detectors decline on corpora too
small to measure, and the report lists them as not measured. A question never
asked must not read as a question asked and answered.

**Purity is not "the share of the score that is real".** It is the share that
survived *the artifacts this audit tests for*. Both ARC-Easy runs came out at
100% purity while differing from each other by 19 points, because presentation
is not one of the three players and so sits entirely outside the decomposition.
Read a purity figure as "none of the three artifacts was established here", never
as "this score is entirely genuine".

**Purity is not a quality score for a model.** It is the fraction of one number
on one benchmark that survived one audit. A model with low purity on a defective
benchmark may be perfectly good; the benchmark is what the finding is about.

**These are statements about items and outputs, not about people.** An
established artifact says a measurement was inflated. It says nothing about the
competence, honesty or conduct of anyone who built the benchmark or the model.
Benchmarks acquire defects through ordinary drift and ordinary deadlines.

## Known gaps, stated plainly

- The reframing intervention is the weakest of the three and would benefit most
  from a verified paraphrase method.
- Only one hosted-API backend is implemented, and it can only produce a choice,
  not a ranking, so the audit cannot see how close a call was.
- The near-duplicate pass caps candidates per item; when the cap is reached the
  report says so and the near-repeat count becomes a lower bound. The exact-
  repeat count is always complete.
- There is no cross-benchmark or cross-model aggregation. Each audit stands
  alone, deliberately, because pooling would require assumptions about
  comparability that the manifests exist to make explicit rather than to hide.
