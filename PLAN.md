# Plan

Status: draft. This file is completed decision by decision during data freezing
(step 2), then frozen in a commit before the first measurement call. After that,
any change is recorded in `DEVIATIONS.md` with its reason.

## Data

### Feature selection

**Filter (set in the protocol).** A feature can be sampled only if its activation
frequency `frac_nonzero`, the share of corpus tokens on which it fires as reported
by Neuronpedia, is strictly between 1e-5 and 1e-2, that is between 0.001% and 1% of
tokens.

**Why the upper bound.** A text is a window of 17 tokens. If a feature fires on a
share f of tokens, a random window contains at least one activation with
probability about 1 − (1 − f)^17, if tokens are taken as independent (activations
cluster, so the true rate is somewhat lower). This is 1.7% at f = 0.1% and 15.7% at
f = 1%. Above 1%, negatives become hard to find and many borrowed passages would be
rejected. Very frequent features also tend to fire on generic patterns that are
hard to state in one sentence. This tendency is not measured on this SAE.

**Why the lower bound.** Below 0.001%, a feature fires on fewer than 1 token in
100,000. Few distinct passages are then available as positives, while 20 distinct
positives at varied activation levels are needed, and the explanation itself rests
on few examples. This is a reason of principle, not a result: in the trial, the
rarest feature, 10468 (f = 0.014%), still had 54 distinct positive passages.

**Status of the bounds.** They are round orders of magnitude, not tuned values and
not taken from a specific paper. They are fixed before sampling.

**Sampling (decided 2026-11-20).** Simple random sampling among eligible features.

1. Seed: **20261120**, written and committed here before any download for the draw.
2. The 24,576 feature indices are shuffled once with `random.Random(20261120).shuffle`
   (Mersenne Twister, Fisher-Yates). The resulting order is saved in
   `data/draw/order.json` with the Python version: Python guarantees the same
   `random()` stream for a seed, not the same `shuffle` algorithm across versions.
3. Features are examined in that order. A feature is eligible if (a) it passes the
   filter above, (b) it has both original explanations, A by gpt-3.5-turbo and B by
   gpt-4o-mini, (c) it has at least 20 distinct stored positive passages
   (`neuronpedia.source_groups`), counted without any call, and (d) it passes the
   gate (Labels, rule 6).
4. The first 100 eligible features form the sample, in that order. The first 5 form
   the pilot. No eligible feature is skipped.
5. A feature dropped later is replaced by the next eligible feature in the same
   order. The position reached in the order is saved, so replacements continue
   from there.
6. Features already seen (the six of the trial, the 120 of the survey) are neither
   excluded nor favoured: only their frequency and explanations were read, nothing
   that the study measures.

**Why simple sampling.** The question is about scorers on the features of this SAE.
A simple random sample estimates that without bias and keeps the planned analysis
(bootstrap over features) unchanged. Frequency is not a variable of the planned
analysis, so quotas per frequency band would only add weights and a stratified
bootstrap. Both options cost a few minutes of downloads.

**Expected, from the survey** (`scripts/survey_features.py`, 120 random features,
exploratory seed 1): 113 of 120 pass conditions (a) to (c), so about 106 features
are examined. The sample should hold about 1 feature between 0.001% and 0.01%, 63
between 0.01% and 0.1%, and 36 between 0.1% and 1%.

**Result of the draw (2026-11-20, `scripts/draw_features.py`, `data/draw/`).** 113
features were examined in the seeded order: 5 failed the frequency filter (one of
them never fires), 8 failed the gate, 100 were kept. The pilot is 1133, 10461,
14889, 47 and 15387. Replacements continue from position 113. The sample holds 1
feature between 0.001% and 0.01%, 71 between 0.01% and 0.1%, and 28 between 0.1%
and 1%.

**Limits, to be reported.**

- *Rare features.* Features below 0.01% are about 1% of eligible features: the
  study says nothing about them.
- *Features that fire on line breaks are excluded by construction.* The scorer's
  text replaces line breaks with spaces (decided 2026-11-19), so a feature whose
  activation sits on the line-break token cannot fire on what the scorer reads.
  Two of the 108 features that reached the gate are of this kind (11574 and 1715,
  peak on the line break, recomputed at 0).
- *The gate shifts the sample towards less frequent features.* The other 6 gate
  failures collapse when their strongest window is sent alone: 4 fall to 0 and 2
  fall below the threshold (6% and 9% of the maximum). Six of the 8 failures are
  above 0.1%, so the sample holds 28 features above 0.1% where the survey predicted
  about 36. The study describes features whose activation can be read from the
  window alone.

### Negatives (decided 2026-11-19)

**Source.** The 20 negative texts of a target feature are drawn from the stored
windows of the other sampled features. Neuronpedia stores 85 windows of 17 tokens
per feature, but its zero-activation windows are sliding windows cut from a few
passages: once duplicates are removed, about 6 distinct passages remain per feature
(checked on features 100, 6048 and 9396), not 20.

**Rule.**

1. The pool holds every distinct passage among the stored windows of the sampled
   features. Windows that share a run of 6 identical tokens count as one passage,
   within a feature and across features.
2. A passage is eligible as a negative for target feature T if it does not come
   from T, does not come from the feature whose explanation serves as T's placebo,
   and shares no run of 6 tokens with any stored window of T.
3. Each passage is used for at most one target feature.
4. Passages are shuffled once with a published seed. Targets then take turns, each
   taking its next eligible unused passage, until each has 20 negatives plus a
   reserve.
5. Each candidate is checked with Neuronpedia's activation API for T, following
   the label rule below. A candidate whose activation for T is above 0 is
   rejected and replaced by the next one in T's list.

**Why this source.** Positives and negatives then come from the same corpus, with
the same tokenisation, the same length and the same decoding, so their form gives
no cue about the label. No download is needed. The number of verification calls is
the same as with any other source.

**Limits and trade-offs, to be reported.**

- *Not a random sample of the corpus.* Each negative is a passage where another
  feature fires strongly, cut around that feature's peak. Negatives may be easier
  or harder to reject than random text. Specificity is therefore measured on this
  population only, and is not comparable with scores computed on random negatives
  (delphi's default).
- *Positives and negatives share a structure.* Both are windows centred on a token
  where some feature fires (position 8 of 17 for most windows). This is intended:
  a scorer cannot tell positives from negatives by where the text is cut.
- *Placebo interaction.* Rule 2 prevents a target's negatives from matching its
  placebo explanation by construction. A negative can still match the placebo, or
  the original explanation, by chance. This holds for any source of negatives.
- *Semantic neighbours.* When two sampled features have close explanations, one
  supplies negatives that fit the other's explanation without activating it. In
  the trial, feature 100 ("phrases related to causes and effects") and feature
  10468 ("transition words indicating cause and effect") were both drawn: a
  "Thus …" passage of 10468 can be a verified negative for 100, which a scorer
  reading explanation A will likely call positive. The label is correct (feature
  100 does not fire), so this measures how precise the explanation is, but how
  often it happens depends on which features were sampled. Close features are not
  filtered out. Each negative keeps its source feature, so these cases can be
  counted afterwards (exploratory).
- *Double use.* A passage can be a positive of its own feature and a negative of
  another one. Rule 3 only forbids using it twice as a negative.
- *Dependence between features.* All targets draw from one shared pool. Rule 3
  removes shared texts, but a target's negatives still depend on which other
  features were sampled. The bootstrap over features treats features as
  independent, which is a small approximation.
- *Labels describe the window alone.* A negative is checked on the text the scorer
  sees, without the text that preceded it in the corpus. This is the label
  that matches the scorer's task, not the activation in the original context
  (on the positives recomputed so far within the filter, recomputed values range
  from 26% below to 115% above the stored ones).
- *Size of the pool.* In the trial (`scripts/check_negatives.py`), six features gave
  472 windows and 346 distinct passages, 38 to 65 per feature. One hundred features
  should give about 5,800 passages, for about 3,000 needed (20 negatives and 10
  reserves per feature). The 12 candidates checked were all at 0 for their target.
- *Order of decisions.* The pool depends on the sampled features, the exclusion
  on the placebo rule, and the rejection on the label rule (decided below). The
  sampling and placebo rules are decided in this file.

### Labels (decided 2026-11-20)

**Rule.**

1. Every label is measured with Neuronpedia's activation API on exactly the text
   shown to the scorer (`neuronpedia.scorer_text`, line breaks replaced by spaces),
   never on stored values. The activation of a text is its maximum over all
   tokens, at any position.
2. *Negative*: activation exactly 0. The SAE returns true zeros, so there is no
   rounding issue.
3. *Positive*: activation at least 10% of the feature's stored maximum
   (`maxActApprox`). The stored maximum stays the reference because no recomputed
   maximum over the corpus exists.
4. *Grey zone*: anything in between is dropped and replaced. Rejections are
   counted per feature and reported.
5. *Choice of positives*: every distinct positive passage of a feature is
   recomputed. Among those labelled positive, sorted by recomputed activation, the
   20 at evenly spaced ranks are kept, so that levels are spread over what the
   scorer reads.
6. *Gate*: before anything else, the window holding a feature's stored maximum is
   recomputed. If it is not a positive by rule 3, the feature is dropped and
   replaced by the next feature of the seeded draw.
7. *Replacement*: a feature that lacks 20 positives or 20 negatives is replaced by
   the next feature of the seeded draw. Replaced features are listed and reported.
8. *Order*: the features of the pilot are verified first.

**Why.** Recomputing changes values both ways. Within the filter, recomputed
values range from 26% below to 115% above the stored ones, and two of the 16
positives checked cross the 10% line. Removing line breaks also changes the
value (feature 9396, window 50: 4.30 with them, 5.40 without). A label must
describe the text the scorer reads. In the trial, feature 16466 (outside the
filter) returned 0 on all four positives recomputed, including its maximum: the
gate catches such features with one call. No positive within the filter fell to 0.
Two independent reviews (2026-11-20) checked this rule against the trial data.

**Cost.** About 7,500 to 8,500 free calls, about 8 hours at 3.6 s per call.

**Limits, to be reported.**

- *Clear cases only.* The grey zone is removed, and Neuronpedia stores almost no
  window below 10% of a feature's maximum. Nothing is said about weak activations.
- *Differences from delphi.* delphi uses activations in context, positives
  spread over quantiles and random negatives, with no grey zone. Here labels are
  recomputed without context, with a threshold relative to the stored maximum.
- *Rejected negatives are semantic neighbours* (in the trial, a "Thus far" passage
  at 0.013 for feature 100). Removing them leaves easier negatives, which slightly
  flatters specificity under the original explanations.
- *Features that need long context* fail the gate or lose positives and are
  replaced. The sample then leans towards features readable from 17 tokens. The
  replaced features are listed so this can be judged.
- *Length.* Removing line breaks shortens about 20% of positive windows to 13 to
  15 tokens. Whether positives and negatives differ in this respect is checked
  before freezing.

### Placebos (decided 2026-11-20)

Each feature receives one placebo: the explanation of another feature (the donor),
with the same scorer prompt. Labels stay those of the target feature.

**Rule.**

1. *Permutation.* The 100 sampled features are shuffled with seed 20261120 until no
   feature is its own donor (a derangement). Each explanation then serves as a
   placebo exactly once.
2. *A or B.* The same seed splits the targets in two halves: 50 receive their
   donor's explanation A (gpt-3.5-turbo), 50 its explanation B (gpt-4o-mini), so
   the placebo does not share the style of only one explainer model.
3. *Filters.* A pair (target, donor) fails if (a) the placebo shares a content word
   with the target's explanations A or B (first five letters, after removing
   generic words such as "phrases", "related", "words"), (b) the donor is among the
   target's 25 nearest features by decoder cosine on Neuronpedia
   (`topkCosSimIndices`), or the reverse, or (c) the placebo's length in words is
   outside 2/3 to 3/2 of the mean length of the target's A and B.
4. *Repair.* While a pair fails, one failing target is drawn and its donor is
   swapped with that of another target drawn at random, unless the swap creates a
   fixed point. The draws use the same seeded generator.
5. *Triage.* A fresh agent running Claude Fable 5.2, with no other context, reads
   the 100 pairs and classes each as *confident* (the placebo cannot plausibly fit
   the target's texts) or *doubt*, with a one-line reason. Its output is kept.
6. *Human review.* Salah reads every *doubt* pair, plus 20 *confident* pairs drawn
   with seed 20261120. If any of these 20 is rejected, he reads all confident pairs.
   Each rejected pair is reassigned by rules 3 and 4 (donors are swapped, so other
   pairs may change), and every changed pair is reviewed again.
7. *Rounds.* After a review, rejected (target, donor) pairs and rejected placebo
   texts are forbidden, and the repair of rule 4 runs again from the previous
   assignment with seed 20261120 + round number. Each round is kept in
   `data/placebos/roundN/` (assignment, triage, review list, decisions). From round
   2 on, the triage reads all 100 pairs again and Salah reviews every doubt pair:
   the check on 20 confident pairs was passed in round 1.
8. *Generic placebos* (decided at the round 1 review). A placebo that describes a
   generic pattern (punctuation, a function word, pronouns, grammar) fits most texts,
   positive or negative. It does not bias the gain, but it makes the placebo
   condition uneven across features, so it is rejected and counts as a doubt in the
   triage.
8b. *Donor pool* (decided after the round 2 triage). Rules 1 and 8 cannot hold
   together: some sampled features have two generic explanations (for example
   "proper nouns, specifically names" and "names of individuals"), so they cannot
   donate, yet rule 1 makes every sampled feature donate once. From round 3 on:
   - a fresh Fable 5.2 agent classes every explanation of the sample, and of the
     candidate donors below, as *specific* or *generic*, and its output is kept,
   - a fourth filter (d) rejects a placebo classed generic,
   - a sampled feature whose A and B are both generic is not a donor (it stays a
     target),
   - the next features of the draw order from position 113 that pass conditions (a)
     to (c) of the sampling are candidate donors (30 fetched,
     `scripts/donor_candidates.py`). Every candidate with at least one specific
     explanation joins the pool. These outside donors are not targets: they supply
     no texts,
   - the pool then holds more donors than targets, and each donor serves at most
     once. With exactly one donor per target no valid assignment existed: a donor
     whose only specific explanation is "percentage values" (2 words) fails the
     length filter for every target, and a maximum matching covered 99 targets,
   - round 3 starts from a new derangement of the targets onto the pool (seed
     20261120 + 3), keeps the pairs rejected in round 1 forbidden, and repairs as
     in rule 4, a failing target also being able to take an unused donor. Round 2
     was not reviewed.
9. *Automatic rounds* (decided at the round 3 review, where Salah rejected all 21
   doubt pairs and found the triage judgments right). Rounds 4 to 6 run without
   human review: the assignment of the previous round is repaired as in rule 4
   (unused donors as spares, every pair rejected so far forbidden, seed 20261120 +
   round), except that a swap is made only if the other target's new pair passes,
   so that valid pairs move only to fix a rejected one (the plain repair changed 99
   placebos to fix 21), a fresh Fable 5.2 agent triages only the pairs whose placebo changed
   (the others keep their verdict), and every doubt is rejected. The number of
   doubts per round is reported. After round 6 (doubts 21, 11, 4, 3 from round 3
   to 6), Salah decided to go on until a round has no doubt, at most up to round 9.
   A target still in doubt after round 9 goes back to him. The first round with no
   doubt gives the final assignment. At the round 9 review, Salah rejected the last
   doubt (5479) and the pair 23318, whose placebo "references to direct
   communication or messages" had been classed generic in rounds 7 and 8, forbade
   that text, and asked for a round 10. From round 10 on, texts rejected as generic
   are forbidden in the automatic rounds too. At the round 10 review, he rejected
   both doubts, forbade the placebo "references to government bodies and
   officials" (flagged on two targets in a row, too broad), and asked for a round
   11. At the round 11 review, he forbade the placebo "names of a person,
   specifically the name "Corey"" (light doubt on two targets in a row) and asked for
   a round 12. At the round 12 review, he rejected the strong doubt on 19647 and
   decided that, from round 13 on, the triage marks each doubt strong or light,
   strong doubts are rejected automatically, and the rounds stop as soon as a round
   leaves only light doubts or none (at most up to round 16). The light doubts left
   then go to him at once, and the assignment is frozen.
10. *Freeze.* The final assignment, the triage and the review are committed before
   any scorer call. Rule 2 of the negatives then excludes each donor's passages
   from its target's negatives.

**Why.** A placebo close in meaning to the target pushes placebo accuracy above
50% and understates the gain. A placebo of very different length mixes meaning and
form. The filters remove the obvious cases (in a simulation on the sample, a
random derangement had 42 failing pairs out of 100, 38 of them for length), the
triage points the review at the subtle ones, and the 20 confident pairs check the
triage instead of trusting it.

**Limits, to be reported.**

- *Triage by a related model.* The triage model is not one of the five scorers, but
  it comes from the same family as three of them (Anthropic). Its notion of
  "unrelated" may suit Claude scorers better than the others. It only directs the
  review: every rejection is a human decision.
- *Not every pair is read by a human* unless the check on 20 confident pairs fails.
  This departs from the protocol's "every placebo reviewed", decided before any
  measurement.
- *Lexical filter.* Meaning can be close without a shared word, and a shared word
  does not always mean close meaning. The filter is a screen, not a judgment.

### Open points

- Windows that contain `<|endoftext|>` join two documents (12 of 85 windows for
  feature 100).
- Passages that hold both positive and negative windows.

## Analyses

To be written before freezing: which pass feeds each analysis, the third-pass rule
and its threshold, kappa conditions, how invalid answers are counted.
