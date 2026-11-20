# Plan

Status: draft. This file is completed decision by decision during data freezing
(step 2), then frozen in a commit before the first measurement call. After that,
any change is recorded in `DEVIATIONS.md` with its reason.

How to read it: section 1 is a one-page summary, section 2 holds the rules in the
order the study runs, section 3 gathers every limit to be reported, and the
appendix keeps the dated decisions, the history and the supporting figures.

## 1. Summary

- **Question.** With texts and explanations held fixed, do scorer models differ in
  their judgments (analysis 1), and in the gain an explanation brings them over an
  unrelated explanation, the placebo (analysis 2)? The question is about scorers on
  the features of this SAE.
- **Sample.** 100 features drawn by simple random sampling (seed 20261120) among the
  24,576 feature indices. A feature is eligible if its activation frequency is
  strictly between 1e-5 and 1e-2, it has both original explanations (A by
  gpt-3.5-turbo, B by gpt-4o-mini), it has at least 20 distinct stored positive
  passages, and its strongest window is still positive when recomputed alone (the
  gate). The first 5 features of the draw form the pilot.
- **The 40 texts per feature.** Each text is a window of 17 tokens stored by
  Neuronpedia. 20 positives are taken at evenly spaced ranks of recomputed
  activation. 20 negatives are borrowed from the stored windows of the other sampled
  features, never from the target or from its placebo donor, each passage serving
  one target only.
- **Labels.** Every label is recomputed with Neuronpedia's activation API on exactly
  the text shown to the scorer. A negative has activation exactly 0, a positive at
  least 10% of the feature's stored maximum, and anything in between is dropped and
  replaced.
- **The three conditions.** The scorer reads explanation A, explanation B, or a
  placebo, which is the explanation of another feature (the donor). Placebos pass
  filters on shared words, decoder similarity, length and generic wording, then a
  triage by a fresh agent, and a human review of the doubts in some rounds. The assignment is frozen in
  `data/placebos/final.json`.
- **Scorers.** Claude Haiku 5.5, Sonnet 5.5 and Opus 5.5, Gemini 2.5 Flash and
  Mistral Small 3.2, one frozen detection prompt (git tag `prompt-frozen`), one text
  per call.
- **Passes.** Every call is made twice, in separate requests. A scorer that changes
  its answer between the two passes on more than 5% of its pilot calls gets a third
  pass on the whole grid. Volume with two passes: 120,000 calls.
- **The three analyses.** (1) Agreement between the 10 pairs of scorers: raw
  agreement, disagreement rate, Cohen's kappa (Figure 1). (2) Gain over the placebo,
  the central measure: mean accuracy with A and B minus accuracy with the placebo,
  per scorer, then the 10 pairwise differences of gain (Figure 2). (3) Profile per
  scorer: sensitivity, specificity, rate of "1" answers, invalid rate, stability
  between passes (Table). Comparisons are always paired, and confidence intervals
  come from a bootstrap over features.
- **The pilot gate.** The whole chain runs on the 5 pilot features. The grid starts
  only if negatives and positives are verified, placebo accuracy is near 50% for
  every scorer, invalid answers stay below 2% per scorer, the real cost per call is
  measured, and both central measures are computed end to end.

## 2. Rules

### 2.1 Feature selection

Simple random sampling among eligible features.

1. *Filter (set in the protocol).* A feature can be sampled only if its activation
   frequency `frac_nonzero`, the share of corpus tokens on which it fires as
   reported by Neuronpedia, is strictly between 1e-5 and 1e-2, that is between
   0.001% and 1% of tokens. The bounds are round orders of magnitude, not tuned
   values and not taken from a specific paper. They are fixed before sampling.
2. *Seed.* The seed is **20261120**, written and committed here before any download
   for the draw.
3. *Order.* The 24,576 feature indices are shuffled once with
   `random.Random(20261120).shuffle` (Mersenne Twister, Fisher-Yates). The
   resulting order is saved in `data/draw/order.json` with the Python version:
   Python guarantees the same `random()` stream for a seed, not the same `shuffle`
   algorithm across versions.
4. *Eligibility.* Features are examined in that order. A feature is eligible if
   (a) it passes the filter of rule 1, (b) it has both original explanations, A by
   gpt-3.5-turbo and B by gpt-4o-mini, (c) it has at least 20 distinct stored
   positive passages (`neuronpedia.source_groups`), counted without any call, and
   (d) it passes the gate (Labels, rule 5).
5. *Sample and pilot.* The first 100 eligible features form the sample, in that
   order. The first 5 form the pilot. No eligible feature is skipped.
6. *Replacement.* A feature dropped later is replaced by the next eligible feature
   in the same order. The position reached in the order is saved, so replacements
   continue from there.
7. *Features already seen.* Features already seen (the six of the trial, the 120 of
   the survey) are neither excluded nor favoured: only their frequency and
   explanations were read, nothing that the study measures.

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
on few examples. This is a reason of principle, not a result (trial figure in A.2).

**Why simple sampling.** The question is about scorers on the features of this SAE.
A simple random sample estimates that without bias and keeps the planned analysis
(bootstrap over features) unchanged. Frequency is not a variable of the planned
analysis, so quotas per frequency band would only add weights and a stratified
bootstrap. Both options cost a few minutes of downloads.

The expected composition (survey) and the result of the draw are in A.2.

### 2.2 Texts

Each feature gets 40 texts: 20 positives and 20 negatives. A text is a window of 17
tokens. The text shown to the scorer is `neuronpedia.scorer_text`, where line breaks
are replaced by spaces.

#### Positives

1. *Mixed passages.* A passage counts as positive for its feature when any of its
   windows is positive. Negatives already exclude any passage close to a window of
   the target, so no conflict arises.
2. *One window per passage.* When a positive passage holds several sliding windows,
   the window kept, recomputed and shown is the one with the highest stored
   activation, centred on the peak.
3. *Choice of positives.* Every distinct positive passage of a feature is
   recomputed. Among those labelled positive (Labels, rule 3), sorted by recomputed
   activation, the 20 at evenly spaced ranks are kept, so that levels are spread
   over what the scorer reads.

The counts of mixed passages and of passages with several windows are in A.6.

#### Negatives

**Source.** The 20 negative texts of a target feature are drawn from the stored
windows of the other sampled features. Neuronpedia stores up to 85 windows of 17
tokens per feature (85 for feature 100), but its zero-activation windows are sliding windows cut from a few
passages: once duplicates are removed, about 6 distinct passages remain per feature
(checked on features 100, 6048 and 9396), not 20.

1. *Pool.* The pool holds every distinct passage among the stored windows of the
   sampled features. Windows that share a run of 6 identical tokens count as one
   passage, within a feature and across features.
2. *Eligibility.* A passage is eligible as a negative for target feature T if it
   does not come from T, does not come from the feature whose explanation serves as
   T's placebo, and shares no run of 6 tokens with any stored window of T.
3. *Single use.* Each passage is used for at most one target feature.
4. *Draw.* Passages are shuffled once with a published seed. Targets then take
   turns, each taking its next eligible unused passage, until each has 20 negatives
   plus a reserve.
5. *Check.* Each candidate is checked with Neuronpedia's activation API for T,
   following the label rule (section 2.3). A candidate whose activation for T is
   above 0 is rejected and replaced by the next one in T's list.

**Why this source.** Positives and negatives then come from the same corpus, with
the same tokenisation, the same length and the same decoding, so their form gives
no cue about the label. No download is needed. The number of verification calls is
the same as with any other source.

#### Document separators

A stored window can hold `<|endoftext|>`, the token that separates two documents in
GPT-2's corpus. Neuronpedia's activation API reads it as one special token, so the
recomputed label matches the text shown.

1. *Kept.* Windows with a separator are kept, for positives and negatives alike,
   and shown to the scorer as they are.
2. *Balance.* After the 40 texts of every feature are built, the share of texts
   holding a separator is compared between positives and negatives. If it differs
   by more than 5 points, negatives with a separator are replaced by the next
   eligible passages without one, from the same seeded list, until the gap is at
   most 5 points, so that the separator says nothing about the label.

**Why.** Dropping these windows would leave two sampled features (14889 and 21484)
with fewer than 20 distinct positive passages, so the sample and the frozen
placebos would change. Dropping them on one side only would make the separator a
cue. The share of windows concerned is in A.6.

### 2.3 Labels

1. *Measure.* Every label is measured with Neuronpedia's activation API on exactly
   the text shown to the scorer (`neuronpedia.scorer_text`, line breaks replaced by
   spaces), never on stored values. The activation of a text is its maximum over
   all tokens, at any position.
2. *Negative.* Activation exactly 0. The SAE returns true zeros, so there is no
   rounding issue.
3. *Positive.* Activation at least 10% of the feature's stored maximum
   (`maxActApprox`). The stored maximum stays the reference because no recomputed
   maximum over the corpus exists.
4. *Grey zone.* Anything in between is dropped and replaced. Rejections are counted
   per feature and reported.
5. *Gate.* Before anything else, the window holding a feature's stored maximum is
   recomputed. If it is not a positive by rule 3, the feature is dropped and
   replaced by the next feature of the seeded draw.
6. *Replacement.* A feature that lacks 20 positives or 20 negatives is replaced by
   the next feature of the seeded draw. Replaced features are listed and reported. A
   replacement feature becomes a target: it receives a placebo by a new round of
   the placebo rules (repair, triage of the changed pairs, review of the doubts),
   and the replaced feature's placebo donor is freed.
7. *Order.* The features of the pilot are verified first.

The choice of the 20 positives among the passages labelled positive is rule 3 of
Texts, Positives.

**Why.** Recomputing changes values both ways, and removing line breaks also
changes the value. A label must describe the text the scorer reads. The gate
catches with one call a feature whose stored positives return 0 when recomputed.
The trial figures behind these statements are in A.4.

**Cost.** About 7,500 to 8,500 free calls, about 8 hours at 3.6 s per call.

### 2.4 Placebos

Each feature (the target) receives one placebo: the explanation of another feature
(the donor), with the same scorer prompt. Labels stay those of the target feature.
The assignment is built in rounds, and several rules took effect at a given round:
each rule states the rounds it applies to. What happened in each round is in A.5.

1. *Starting assignment.* In rounds 1 and 2, the 100 sampled features are shuffled
   with seed 20261120 until no feature is its own donor (a derangement). Each
   explanation then serves as a placebo exactly once. Round 3 starts from a new
   derangement of the targets onto the donor pool of rule 5 (seed 20261120 + 3) and
   keeps the pairs rejected in round 1 forbidden.
2. *A or B.* The same seed splits the targets in two halves: 50 receive their
   donor's explanation A (gpt-3.5-turbo), 50 its explanation B (gpt-4o-mini), so
   the placebo does not share the style of only one explainer model. The letter is
   fixed per target, and filter (d) of rule 3 applies to the donor's explanation of
   that letter: a donor can serve a target only if that explanation is specific.
3. *Filters.* A pair (target, donor) fails if (a) the placebo shares a content word
   with the target's explanations A or B (first five letters, after removing
   generic words such as "phrases", "related", "words"), (b) the donor is among the
   target's 25 nearest features by decoder cosine on Neuronpedia
   (`topkCosSimIndices`), or the reverse, (c) the placebo's length in words is
   outside 2/3 to 3/2 of the mean length of the target's A and B, or, from round 3
   on, (d) the placebo is classed generic (rule 4).
4. *Generic placebos.* A placebo that describes a generic pattern (punctuation, a
   function word, pronouns, grammar) fits most texts, positive or negative. It does
   not bias the gain, but it makes the placebo condition uneven across features, so
   it is rejected and counts as a doubt in the triage. From round 3 on, a fresh
   Fable 5.2 agent classes every explanation of the sample, and of the candidate
   donors of rule 5, as *specific* or *generic*, and its output is kept.
5. *Donor pool (from round 3 on).* A sampled feature whose A and B are both generic
   is not a donor (it stays a target). The next features of the draw order from
   position 113 that pass conditions (a) to (c) of the sampling (Feature selection,
   rule 4) are candidate donors (30 fetched, `scripts/donor_candidates.py`). Every
   candidate with at least one specific explanation joins the pool. These outside
   donors are not targets: they supply no texts. The pool then holds more donors
   than targets, and each donor serves at most once. The reason for the pool is in
   A.5.
6. *Repair.* While a pair fails, one failing target is drawn and its donor is
   swapped with that of another target drawn at random, unless the swap creates a
   fixed point. The draws use the same seeded generator. From round 3 on, a
   failing target can also take an unused donor of the pool.
7. *Triage.* A fresh agent running Claude Fable 5.2, with no other context, reads
   the 100 pairs and classes each as *confident* (the placebo cannot plausibly fit
   the target's texts) or *doubt*, with a one-line reason. Its output is kept.
8. *Human review (round 1).* Salah reads every *doubt* pair, plus 20 *confident* pairs drawn
   with seed 20261120. If any of these 20 is rejected, he reads all confident
   pairs. Each rejected pair is reassigned by rules 3 and 6 (donors are swapped, so
   other pairs may change), and every changed pair is reviewed again.
9. *Rounds.* After a review, rejected (target, donor) pairs and rejected placebo
   texts are forbidden, and the repair of rule 6 runs again from the previous
   assignment with seed 20261120 + round number. Each round is kept in
   `data/placebos/roundN/` (assignment, triage, review list, decisions). In rounds
   2 and 3, the triage reads all 100 pairs again and Salah reviews every doubt
   pair: the check on 20 confident pairs was passed in round 1 (round 2 was not
   reviewed, see rule 5 and A.5).
10. *Automatic rounds (from round 4 on).* Rounds 4 to 8 run without human review.
    The assignment of the previous round is repaired as in rule 6 (unused donors as
    spares, every pair rejected so far forbidden, seed 20261120 + round), except
    that a swap is made only if the other target's new pair passes, so that valid
    pairs move only to fix a rejected one (the plain repair changed 99 placebos to
    fix 21). A fresh Fable 5.2 agent triages only the pairs whose placebo changed
    (the others keep their verdict), and every doubt is rejected. The number of
    doubts per round is reported. This rule first covered rounds 4 to 6, and after
    round 6 it was extended until a round has no doubt, at most up to round 9. From
    round 9 to round 12, the triage still reads only the changed pairs, but the
    doubts left go to Salah, who decides each rejection and may forbid a placebo
    text. From round 10 on, texts rejected as generic are forbidden in the
    automatic repair too. The first round with no doubt gives the final
    assignment.
11. *Strong and light doubts (from round 13 on).* The triage marks each doubt
    strong or light. Strong doubts are rejected automatically, and the rounds stop
    as soon as a round leaves only light doubts or none (at most up to round 16).
    The light doubts left then go to Salah at once, and the assignment is frozen.
12. *Freeze.* The final assignment, the triage and the review are committed before
    any scorer call. Rule 2 of the negatives then excludes each donor's passages
    from its target's negatives.

**Why.** A placebo close in meaning to the target pushes placebo accuracy above
50% and understates the gain. A placebo of very different length mixes meaning and
form. The filters remove the obvious cases (simulation figure in A.5), the triage
points the review at the subtle ones, and the 20 confident pairs check the triage
instead of trusting it.

The frozen result is in `data/placebos/final.json` and described in A.5.

### 2.5 Scoring

1. *Scorers and settings.* Five scorers, with the model ids and settings of
   `src/autointerp/models.py`: Claude Haiku 5.5 (thinking off), Sonnet 5.5 (initial
   thinking off), Opus 5.5 (effort low, thinking cannot be disabled), Gemini 2.5
   Flash (temperature 0, reasoning off, Google AI Studio pinned) and Mistral Small
   3.2 (temperature 0, Parasail bf16 pinned). `max_tokens` is 2048, since thinking
   tokens count towards it. The settings of each call are written into its CSV row.
2. *Prompt.* The frozen detection prompt (git tag `prompt-frozen`), one text per
   call, the same template for the three conditions (explanation A, explanation B,
   placebo).
3. *Reading the answer.* The answer is read with delphi's rule, retried once if
   unreadable, then recorded as `invalid`.
4. *Passes.* Every call is made twice (passes 1 and 2), in separate requests.
5. *Third pass.* If, in the pilot, a scorer gives a different answer in the two
   passes on more than 5% of its calls (X = 5%), it gets a third pass on the whole
   grid. A call is one text under one condition.
6. *Transport.* The three Claude scorers go through Anthropic's Message Batches API
   (half price, delayed answers), the two others through OpenRouter.
7. *Volume.* 100 features × 40 texts × 3 conditions × 5 scorers × 2 passes =
   120,000 calls.
8. *Output.* One CSV row per call: feature, text id, label, condition, explanation
   shown, scorer, model id, settings, pass, raw answer, parsed answer, retry flag,
   tokens (input, output, thinking), cost, latency, timestamp. Raw API responses
   are kept. The CSV is frozen when the grid ends.

### 2.6 Analyses

All analyses are fixed here before any scorer call.

1. *Analysis 1: agreement between scorers (Figure 1).* For each of the 10 pairs of
   scorers: raw agreement, disagreement rate and Cohen's kappa, on conditions A and
   B pooled (the use the explanations are meant for). Agreement under the placebo
   is reported apart, since nearly every answer is 0 there and agreement would be
   inflated.
2. *Analysis 2: gain over the placebo (central measure, Figure 2).* For each
   scorer, gain = mean of its accuracies with A and with B, minus its accuracy with
   the placebo. The gains with A and with B are also reported apart. Then the 10
   pairwise differences of gain between scorers. With 10 comparisons at 95%, about
   one difference in two series of comparisons would exclude 0 by chance:
   differences are read with this in mind, and none is singled out after seeing the
   results.
3. *Analysis 3: profile per scorer (Table).* Sensitivity, specificity, rate of "1"
   answers, invalid rate and stability between passes (share of texts answered
   differently by pass 1 and pass 2, and with a third pass, share of texts whose
   three answers are not all the same), per condition. Stability is never presented
   as a ceiling on agreement between scorers.
4. *Pairing.* Comparisons are always paired: same features, texts and conditions.
5. *Answer per call for accuracy, gain, sensitivity and specificity.* These use the
   mean over passes: a text counts as the share of passes that answered right (1,
   1/2 or 0 with two passes, thirds with three). This estimates the accuracy of a
   single call and uses all passes without breaking ties by chance.
6. *Answer per call for agreement and kappa.* These need one binary answer per
   scorer and text: pass 1 is used. The same figures are recomputed on pass 2 as a
   check, and reported if they differ noticeably.
7. *Invalid answers.* They count as errors in accuracy, are left out of agreement
   (a pair of scorers is compared on the texts both answered), and their rate is
   reported per scorer.
8. *Confidence intervals.* Percentile intervals at 95% from a bootstrap over
   features (the 100 features are resampled with replacement, with their 40 texts
   and all conditions), 10,000 resamples, seed 20261120. Texts are not resampled
   one by one, because the texts of a feature share its explanation and their
   errors move together.
9. *Check on placebo accuracy.* Placebo accuracy is expected near 50%. If it is
   well above (more than 60% for a scorer), a form cue is looked for (length,
   document separators, source feature of the negatives) before any conclusion.
10. *Check on built texts.* The share of texts with a document separator is
    balanced between positives and negatives (Texts, Document separators, rule 2),
    and the mean length in tokens of the scorer's text is compared between
    positives and negatives and reported.
11. *A against B.* The difference between A and B for a given feature is
    exploratory only.

### 2.7 Pilot and stop rules

1. *Pilot.* The pilot runs the whole chain on the 5 pilot features (1133, 10461,
   14889, 47, 15387) with all five scorers, both passes and the three conditions
   (5 × 40 texts × 3 conditions × 2 passes = 1,200 calls per scorer).
2. *Gate.* The grid starts only if the pilot shows:
   - every negative verified inactive, every positive verified,
   - placebo accuracy between 40% and 60% for every scorer,
   - less than 2% invalid answers per scorer,
   - the real cost per call, thinking included, per scorer,
   - both central measures (agreement and gain) computed end to end.
3. *Spending caps.* A spending cap for the pilot is fixed by Salah before the first
   paid call, and the total cap after the pilot, from the measured cost.
4. *Pilot fails.* If the pilot fails, the data are fixed and the grid does not
   start.
5. *Grid too slow or too costly.* A scorer is removed (Opus first) or the number of
   features is reduced (taking the first ones of the draw), never the number of
   texts per feature.
6. *Extensions.* No extension starts before the two figures and the table exist.
   The extensions, in that order if time allows: an arm with no explanation, a
   second prompt, explanations too broad or too narrow (Oikarinen et al.), A
   against B.

The pilot also decides the third pass (Scoring, rule 5).

## 3. Limits to be reported

### Sample

1. *Rare features.* Features below 0.01% are about 1% of eligible features: the
   study says nothing about them.
2. *Features that fire on line breaks are excluded by construction.* The scorer's
   text replaces line breaks with spaces (decided 2026-11-19), so a feature whose
   activation sits on the line-break token cannot fire on what the scorer reads.
   Two of the 108 features that reached the gate are of this kind (11574 and 1715,
   peak on the line break, recomputed at 0).
3. *The gate shifts the sample towards less frequent features.* The other 6 gate
   failures collapse when their strongest window is sent alone: 4 fall to 0 and 2
   fall below the threshold (6% and 9% of the maximum). Six of the 8 failures are
   above 0.1%, so the sample holds 28 features above 0.1% where the survey
   predicted about 36. The study describes features whose activation can be read
   from the window alone.

### Negatives (limits and trade-offs)

4. *Not a random sample of the corpus.* Each negative is a passage where another
   feature fires strongly, cut around that feature's peak. Negatives may be easier
   or harder to reject than random text. Specificity is therefore measured on this
   population only, and is not comparable with scores computed on random negatives
   (delphi's default).
5. *Positives and negatives share a structure.* Both are windows centred on a token
   where some feature fires (position 8 of 17 for most windows). This is intended:
   a scorer cannot tell positives from negatives by where the text is cut.
6. *Placebo interaction.* Rule 2 of the negatives prevents a target's negatives
   from matching its placebo explanation by construction. A negative can still
   match the placebo, or the original explanation, by chance. This holds for any
   source of negatives.
7. *Semantic neighbours.* When two sampled features have close explanations, one
   supplies negatives that fit the other's explanation without activating it. In
   the trial, feature 100 ("phrases related to causes and effects") and feature
   10468 ("transition words indicating cause and effect") were both drawn: a
   "Thus …" passage of 10468 can be a verified negative for 100, which a scorer
   reading explanation A will likely call positive. The label is correct (feature
   100 does not fire), so this measures how precise the explanation is, but how
   often it happens depends on which features were sampled. Close features are not
   filtered out. Each negative keeps its source feature, so these cases can be
   counted afterwards (exploratory).
8. *Double use.* A passage can be a positive of its own feature and a negative of
   another one. Rule 3 of the negatives only forbids using it twice as a negative.
9. *Dependence between features.* All targets draw from one shared pool. Rule 3 of
   the negatives removes shared texts, but a target's negatives still depend on
   which other features were sampled. The bootstrap over features treats features
   as independent, which is a small approximation.
10. *Labels describe the window alone.* A negative is checked on the text the
    scorer sees, without the text that preceded it in the corpus. This is the label
    that matches the scorer's task, not the activation in the original context (on
    the positives recomputed so far within the filter, recomputed values range from
    26% below to 115% above the stored ones).
11. *Size of the pool.* In the trial (`scripts/check_negatives.py`), six features
    gave 472 windows and 346 distinct passages, 38 to 65 per feature. One hundred
    features should give about 5,800 passages, for about 3,000 needed (20 negatives
    and 10 reserves per feature). The 12 candidates checked were all at 0 for their
    target.
12. *Order of decisions.* The pool depends on the sampled features, the exclusion
    on the placebo rule, and the rejection on the label rule (section 2.3, decided
    after the negatives). The sampling and placebo rules are decided in this file.

### Labels

13. *Clear cases only.* The grey zone is removed, and Neuronpedia stores almost no
    window below 10% of a feature's maximum. Nothing is said about weak
    activations.
14. *Differences from delphi.* delphi uses activations in context, positives spread
    over quantiles and random negatives, with no grey zone. Here labels are
    recomputed without context, with a threshold relative to the stored maximum.
15. *Rejected negatives are semantic neighbours* (in the trial, a "Thus far"
    passage at 0.013 for feature 100). Removing them leaves easier negatives, which
    slightly flatters specificity under the original explanations.
16. *Features that need long context* fail the gate or lose positives and are
    replaced. The sample then leans towards features readable from 17 tokens. The
    replaced features are listed so this can be judged.
17. *Length.* Removing line breaks shortens about 20% of positive windows to 13 to
    15 tokens. Whether positives and negatives differ in this respect is checked
    before freezing.

### Placebos

18. *Triage by a related model.* The triage model is not one of the five scorers,
    but it comes from the same family as three of them (Anthropic). Its notion of
    "unrelated" may suit Claude scorers better than the others. In rounds 1, 3 and 9
    to 12, each rejection was decided by Salah. In rounds 4 to 8 and 13, doubts
    (strong doubts in round 13) were rejected on the triage alone, after Salah had
    found the triage right at the round 3 review.
19. *Not every pair is read by a human.* Salah read the doubt pairs and 20
    confident pairs in round 1 (none of the 20 was rejected), the 21 doubt pairs
    of round 3, and the doubts left in rounds 9 to 12. The other confident pairs,
    and the doubts of the automatic rounds, were not read. This departs from the
    protocol's "every placebo reviewed", decided before any measurement.
20. *Lexical filter.* Meaning can be close without a shared word, and a shared word
    does not always mean close meaning. The filter is a screen, not a judgment.

### Document separators

21. About one text in ten joins two documents.

## Appendix A. Decision log

### A.1 Dated decisions

| When | Decision | Rule |
|---|---|---|
| Set in the protocol | Frequency filter, 1e-5 < `frac_nonzero` < 1e-2 | 2.1, rule 1 |
| 2026-11-19 | The scorer's text replaces line breaks with spaces | 2.2, limit 2 |
| 2026-11-19 | Source and rule of the negatives | 2.2, Negatives |
| 2026-11-20 | Sampling: simple random sampling among eligible features, seed 20261120 | 2.1 |
| 2026-11-20 | Draw run with `scripts/draw_features.py`, output in `data/draw/` | A.2 |
| 2026-11-20 | Label rule. Two independent reviews (2026-11-20) checked this rule against the trial data | 2.3 |
| 2026-11-20 | Placebo rule | 2.4 |
| Round 1 review | Generic placebos are rejected and count as a doubt in the triage | 2.4, rule 4 |
| After the round 2 triage | Donor pool, explanation classes, filter (d), outside donors | 2.4, rules 3 to 5 |
| Round 3 review | Automatic rounds 4 to 6 | 2.4, rule 10 |
| After round 6 | Rounds go on until a round has no doubt, at most up to round 9 | 2.4, rule 10 |
| Round 9 review | Round 10 asked for. From round 10 on, texts rejected as generic are forbidden in the automatic rounds too | 2.4, rule 10 |
| Round 10 review | Round 11 asked for | A.5 |
| Round 11 review | Round 12 asked for | A.5 |
| Round 12 review | Strong and light doubts from round 13 on, at most up to round 16 | 2.4, rule 11 |
| 2026-11-20 | Placebo assignment frozen in `data/placebos/final.json` | 2.4, rule 12 |
| 2026-11-20 | Windows: document separators, mixed passages, one window per passage | 2.2 |
| 2026-11-20 | Threshold of the third pass, X = 5% | 2.5, rule 5 |

### A.2 Feature selection: survey, draw and trial figures

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

**Trial figure for the lower bound.** In the trial, the rarest feature, 10468
(f = 0.014%), still had 54 distinct positive passages.

### A.3 Negatives: trial figures

The trial figures of the negatives are part of limits to be reported and are kept
there: the semantic neighbours 100 and 10468 (limit 7), the range of recomputed
values (limit 10) and the size of the pool (limit 11). The check of about 6 distinct
zero-activation passages per feature (features 100, 6048 and 9396) is in section
2.2, Negatives, Source.

### A.4 Labels: trial figures and reviews

- Within the filter, recomputed values range from 26% below to 115% above the
  stored ones, and two of the 16 positives checked cross the 10% line.
- Removing line breaks also changes the value (feature 9396, window 50: 4.30 with
  them, 5.40 without).
- In the trial, feature 16466 (outside the filter) returned 0 on all four positives
  recomputed, including its maximum: the gate catches such features with one call.
- No positive within the filter fell to 0.
- Two independent reviews (2026-11-20) checked the label rule against the trial
  data.

### A.5 Placebos: history of the rounds

**Simulation behind the filters.** In a simulation on the sample, a random
derangement had 42 failing pairs out of 100, 38 of them for length.

**Round 1.** The assignment follows the first form of the rules: a derangement of
the 100 sampled features, each explanation serving exactly once. The check on 20
confident pairs was passed. At the round 1 review, the rule on generic placebos was
decided (section 2.4, rule 4). Rejected: 20 doubt pairs and 4 generic ones. The 4
generic placebo texts are forbidden.

**Round 2.** Round 2 was not reviewed, and no pair was rejected in it. After the
round 2 triage, the donor pool was decided (section 2.4, rule 5), for this reason:
the first form of rule 1 (every sampled feature donates once) and the rule on
generic placebos cannot hold together. Some sampled features have two generic
explanations (for example "proper nouns, specifically names" and "names of
individuals"), so they cannot donate, yet the first form of rule 1 makes every
sampled feature donate once. With exactly one donor per target no valid assignment
existed: a donor whose only specific explanation is "percentage values" (2 words)
fails the length filter for every target, and a maximum matching covered 99
targets. Hence a pool with more donors than targets.

**Round 3.** New derangement of the targets onto the pool (seed 20261120 + 3), with
the pairs rejected in round 1 kept forbidden. At the round 3 review, Salah rejected
all 21 doubt pairs and found the triage judgments right. He then decided the
automatic rounds 4 to 6 (section 2.4, rule 10). The plain repair had changed 99
placebos to fix 21, hence the restricted swap of rule 10.

**Rounds 4 to 6.** Automatic. Doubts were 21, 11, 4, 3 from round 3 to 6. After
round 6, Salah decided to go on until a round has no doubt, at most up to round 9.

**Round 9 review.** Salah rejected the last doubt (5479) and the pair 23318, whose
placebo "references to direct communication or messages" had been classed generic
in rounds 7 and 8, forbade that text, and asked for a round 10. From round 10 on,
texts rejected as generic are forbidden in the automatic rounds too.

**Round 10 review.** He rejected both doubts, forbade the placebo "references to
government bodies and officials" (flagged on two targets in a row, too broad), and
asked for a round 11.

**Round 11 review.** He forbade the placebo "names of a person, specifically the
name "Corey"" (light doubt on two targets in a row) and asked for a round 12.

**Round 12 review.** He rejected the strong doubt on 19647 and decided the rule on
strong and light doubts from round 13 on (section 2.4, rule 11).

**Result (frozen 2026-11-20, `data/placebos/final.json`).** Round 13 left no doubt,
so its assignment is final. Rejected pairs per round:

| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Rejected pairs | 24 (20 doubt, 4 generic) | none (not reviewed) | 21 | 11 | 4 | 3 | 2 | 2 | 2 (1 doubt, 1 generic) | 2 (1 doubt, 1 generic) | 1 (generic) | 1 | 0 |

Seven placebo texts are forbidden: the 4 generic ones of round 1, then "references
to direct communication or messages", "references to government bodies and
officials" and "names of a person, specifically the name "Corey"". The 100 placebos
come from 100 distinct donors, 21 of them outside the sample, 50 from explanations
A and 50 from explanations B, and no target is its own donor. Pairs not changed
since an earlier round keep the verdict of that round's triage, so each final pair
was judged once by a fresh Fable 5.2 agent and found confident.

### A.6 Windows: figures on the sample

- *Document separators.* In the sample, 9.2% of positive windows (480 of 5,201) and 9.3% of
  distinct positive passages (460 of 4,948) hold one, and the peak is after it in two cases out of three,
  never on it.
- *Mixed passages.* Only 2 of the 5,532 passage groups of the sample mix positive
  and zero windows of a feature.
- *One window per passage.* 159 positive passages hold several sliding windows.
