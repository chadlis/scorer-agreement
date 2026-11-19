# Plan

Status: draft. This file is completed decision by decision during data freezing
(step 2), then frozen in a commit before the first measurement call. After that,
any change is recorded in `DEVIATIONS.md` with its reason.

## Data

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
5. Each candidate is checked with Neuronpedia's activation API for T, on the text
   alone. A candidate that activates T above the threshold (to be decided) is
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
- *Labels describe the window alone.* A negative is checked on the 17 tokens the
  scorer sees, without the text that preceded it in the corpus. This is the label
  that matches the scorer's task, not the activation in the original context
  (recomputed values differ from stored ones by 9 to 22% on positives,
  `data/activation_test/`).
- *Size of the pool.* In the trial (`scripts/check_negatives.py`), six features gave
  472 windows and 346 distinct passages, 38 to 65 per feature. One hundred features
  should give about 5,800 passages, for about 3,000 needed (20 negatives and 10
  reserves per feature). The 12 candidates checked were all at 0 for their target.
- *Order of decisions.* The pool depends on the sampled features, the exclusion
  on the placebo rule, and the rejection on the threshold. These are decided next.

### Open points

- Activation threshold for labels, and the exact text sent for verification
  (with or without line breaks).
- Sampling rule for the 100 features, and its seed.
- Placebo rule.
- Windows that contain `<|endoftext|>` join two documents (12 of 85 windows for
  feature 100).
- Passages that hold both positive and negative windows.

## Analyses

To be written before freezing: which pass feeds each analysis, the third-pass rule
and its threshold, kappa conditions, how invalid answers are counted.
