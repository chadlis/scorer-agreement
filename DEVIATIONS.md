# Deviations from the frozen plan

`PLAN.md` was frozen on 2026-11-21 (git tag `plan-frozen`), before the first scorer
call. Every change to the plan made after that date is listed here, with its date,
its reason and when it was decided relative to the results.

## 1. Exploratory analysis: agreement on errors (added 2026-11-21)

**Decided by Salah on 2026-11-21**, after the pilot and while the grid was running,
before any grid result was read. It is exploratory and added after the plan was
frozen: it is reported as such, and no firm conclusion is drawn from it.

**Why.** Analysis 1 (Figure 1) measures how often two scorers give the same answer.
Part of that agreement comes from accuracy alone: two scorers right with rates a1
and a2 are both right on at least a1 + a2 - 1 of the texts, so they agree at least
that often. Kappa corrects for the rates of "1" answers, not for this. Agreement
therefore cannot say whether two scorers make the *same mistakes*. Three of the
five scorers come from one family (Claude), which may share errors.

**What is computed**, on the same data as Analysis 1 (conditions A and B pooled,
pass 1, texts that both scorers answered with 1 or 0):

1. For each of the 10 pairs (x, y): the error rate of each scorer, e_x and e_y, the
   share of texts where both are wrong, e_xy, and the ratio e_xy / (e_x * e_y). A
   ratio of 1 means errors as often together as if independent. A ratio above 1
   means the two scorers tend to fail on the same texts.
2. Family: the mean ratio over the 3 pairs within Claude (Haiku, Sonnet, Opus)
   minus the mean ratio over the 7 pairs across families.
3. Percentile intervals at 95% from the bootstrap over features of PLAN.md 2.6,
   rule 8 (10,000 resamples, seed 20261120).

**How to read it.** Every pair is expected to have a ratio above 1, whatever the
family: some texts are hard for every scorer (the weakest positives, PLAN 2.2), so
errors are never independent. The comparison that matters is within Claude against
across families. It rests on 3 pairs against 7, and family is mixed with settings
(Claude scorers run with thinking off or minimal, the others at temperature 0), so a
difference cannot be attributed to the family alone.

**When.** Only after the two figures and the table of PLAN.md 2.6 exist (PLAN.md
2.7, rule 6).

**Where.** `report.exploratory_errors` and `analysis.error_counts` (hand-checked test
in `tests/test_analysis.py`), numbers in `data/analysis/exploratory_errors.json`,
figure in `figures/exploratory_errors.png`, last section of
`notebooks/analysis.ipynb`. A reading aid, not an added test: the ratio cannot
exceed 1 / max(e_x, e_y), about 4.7 to 4.8 for these pairs, since shared errors are
at most the errors of the scorer that errs less.

## 2. Exploratory analysis, to be done later: do scorers rank explanations alike? (noted 2026-11-21)

**Decided by Salah on 2026-11-21, after reading the grid results** (unlike
deviation 1, which was decided before). Not computed yet. Weaker than deviation 1:
the question was raised by the results themselves, so it is reported as an
exploratory follow-up, not as evidence.

**Why.** The planned analyses compare scorers text by text (agreement) and on
average (gain). The practical use of an autointerp score is to compare
explanations: which of two explanations of a feature is better, or which features
are better explained. A gain larger by 3 points on average does not say whether two
scorers would rank explanations differently.

**What will be computed**, on conditions A and B, mean over passes, invalid counted
as wrong (as for accuracy in PLAN.md 2.6):

1. For each scorer, the accuracy of each of the 200 explanations (100 features times
   A and B). For each pair of scorers, the Spearman rank correlation of these 200
   scores.
2. For each pair of scorers, the share of features where both scorers find the same
   explanation better (A or B), leaving out features where either scorer gives A and
   B the same accuracy.
3. Intervals by the bootstrap over features of PLAN.md 2.6, rule 8.

The difference between A and B for a given feature stays exploratory (PLAN.md 2.6,
rule 11): only the agreement between scorers on that difference is studied, not
which explanation is better.

**Done on 2026-11-21**, while the write-up was being drafted, after an external
review of the repository had computed the same quantities with the definition
above. The numbers were recomputed here and match (`report.exploratory_ranking`,
`data/analysis/exploratory_ranking.json`, notebook). One reference was added that
the definition did not foresee: each scorer against itself, pass 1 against pass 2,
which gives the share of opposite choices due to the scorer's own noise. The share
of opposite choices is counted over the 100 features, ties included.

## 3. Robustness check: simultaneous intervals for the gain differences (2026-11-21)

**Suggested by the same external review, after the results.** PLAN.md 2.6, rule 2,
reads the 10 differences of gain with ordinary 95% intervals and warns about
multiplicity in words. As a check, the 10 differences are also given with
Bonferroni intervals (each at 99.5%, bootstrap over features as in rule 8), so that
the 10 hold together at 95% (`report.gain_differences_simultaneous`,
`data/analysis/gain_differences_simultaneous.json`). The planned intervals stay the
reported ones. The review also proposed a control on 508 identical excerpts that are
positive for one feature and negative for another. It is not adopted: the rule that
selects those excerpts was not fixed beforehand, and it was not reproduced here.
