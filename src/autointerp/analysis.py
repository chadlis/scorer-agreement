"""Analyses fixed in PLAN.md (section 2.6): accuracy, gain over the placebo, agreement,
Cohen's kappa, profile per scorer, and a bootstrap over features.

Input: one row per call, with columns feature, text, label (0 or 1), condition
("A", "B" or "P"), scorer, pass (1, 2, 3) and answer ("1", "0" or "invalid").
Every statistic is built from counts summed over features, so the bootstrap only
resamples a small table of per-feature counts.
"""

from collections.abc import Callable
from itertools import combinations

import numpy as np
import pandas as pd

SEED = 20261120
CONDITIONS = ("A", "B", "P")


def text_scores(calls: pd.DataFrame) -> pd.DataFrame:
    """One row per (feature, text, condition, scorer): mean over passes of being right
    (an invalid answer counts as wrong), and of answering "1"."""
    c = calls.assign(
        right=(calls["answer"] == calls["label"].astype(str)).astype(float),
        yes=(calls["answer"] == "1").astype(float),
        invalid=(calls["answer"] == "invalid").astype(float),
    )
    keys = ["feature", "text", "condition", "scorer", "label"]
    return c.groupby(keys, as_index=False)[["right", "yes", "invalid"]].mean()


def accuracy_counts(scores: pd.DataFrame) -> pd.DataFrame:
    """Per feature: sum of scores and number of texts, for each scorer and condition."""
    g = scores.groupby(["feature", "scorer", "condition"])["right"].agg(["sum", "count"])
    return g.unstack(["scorer", "condition"]).fillna(0)


def accuracy(totals: pd.Series, scorer: str, condition: str) -> float:
    return totals[("sum", scorer, condition)] / totals[("count", scorer, condition)]


def gain(totals: pd.Series, scorer: str) -> float:
    """Mean of the accuracies with A and with B, minus the accuracy with the placebo."""
    a, b = accuracy(totals, scorer, "A"), accuracy(totals, scorer, "B")
    return (a + b) / 2 - accuracy(totals, scorer, "P")


def kappa_from_counts(n11: float, n10: float, n01: float, n00: float) -> float:
    """Cohen's kappa of two raters from their 2 x 2 table (1/1, 1/0, 0/1, 0/0)."""
    n = n11 + n10 + n01 + n00
    observed = (n11 + n00) / n
    yes_a, yes_b = (n11 + n10) / n, (n11 + n01) / n
    expected = yes_a * yes_b + (1 - yes_a) * (1 - yes_b)
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def pair_counts(
    calls: pd.DataFrame, conditions=("A", "B"), pass_: int = 1
) -> dict[tuple[str, str], pd.DataFrame]:
    """Per pair of scorers, per feature: the 2 x 2 table of their answers on one pass,
    leaving out texts where either answer is invalid."""
    c = calls[(calls["pass"] == pass_) & calls["condition"].isin(conditions)]
    wide = c.pivot_table(
        index=["feature", "text", "condition"], columns="scorer", values="answer", aggfunc="first"
    )
    out = {}
    for x, y in combinations(sorted(wide.columns), 2):
        both = wide[[x, y]].dropna()
        both = both[(both[x] != "invalid") & (both[y] != "invalid")]
        cells = {
            "n11": (both[x] == "1") & (both[y] == "1"),
            "n10": (both[x] == "1") & (both[y] == "0"),
            "n01": (both[x] == "0") & (both[y] == "1"),
            "n00": (both[x] == "0") & (both[y] == "0"),
        }
        table = pd.DataFrame({k: v.astype(int) for k, v in cells.items()})
        out[(x, y)] = table.groupby(level="feature").sum()
    return out


def agreement(t: pd.Series) -> dict[str, float]:
    n = t.sum()
    raw = (t["n11"] + t["n00"]) / n
    return {
        "agreement": raw,
        "disagreement": 1 - raw,
        "kappa": kappa_from_counts(t["n11"], t["n10"], t["n01"], t["n00"]),
        "n": n,
    }


def stability(calls: pd.DataFrame) -> pd.Series:
    """Per scorer: share of (text, condition) answered differently by passes 1 and 2."""
    keys = ["feature", "text", "condition", "scorer"]
    wide = calls[calls["pass"].isin([1, 2])].pivot_table(
        index=keys, columns="pass", values="answer", aggfunc="first"
    )
    return (wide[1] != wide[2]).groupby(level="scorer").mean()


def cluster_bootstrap(
    per_feature: pd.DataFrame,
    stat: Callable[[pd.Series], float],
    n: int = 10_000,
    seed: int = SEED,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Point estimate and percentile interval of `stat` over features resampled with
    replacement. `per_feature` holds additive counts, one row per feature."""
    values = per_feature.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    k = len(values)
    draws = np.empty(n)
    for b in range(n):
        totals = values[rng.integers(0, k, size=k)].sum(axis=0)
        draws[b] = stat(pd.Series(totals, index=per_feature.columns))
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
    return stat(per_feature.sum()), float(lo), float(hi)
