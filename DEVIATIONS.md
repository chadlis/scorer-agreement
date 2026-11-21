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
