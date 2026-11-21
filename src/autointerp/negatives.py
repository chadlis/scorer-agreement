"""Candidate negative texts for each target feature (PLAN.md, "Negatives").

Negatives come from the stored windows of the other sampled features. This module
only builds and assigns candidates. Each candidate is then checked with the
activation API (autointerp.activation), and rejected above the threshold.
"""

import random
from dataclasses import dataclass

from autointerp.neuronpedia import kgrams, source_groups


@dataclass(frozen=True)
class Passage:
    feature: int  # feature whose stored windows hold this passage
    window: int  # index of the window in that feature's "activations" list
    tokens: tuple[str, ...]


def build_pool(windows: dict[int, list[list[str]]], k: int = 6) -> list[Passage]:
    """One passage per source text across all features.

    Windows that share a run of k tokens, within a feature or across features,
    count as one passage, represented by its first window in feature order.
    """
    flat = [(f, i, w) for f in sorted(windows) for i, w in enumerate(windows[f])]
    groups = source_groups([w for _, _, w in flat], k)
    seen: set[int] = set()
    pool = []
    for (f, i, w), g in zip(flat, groups, strict=True):
        if g not in seen:
            seen.add(g)
            pool.append(Passage(f, i, tuple(w)))
    return pool


def assign_candidates(
    pool: list[Passage],
    own_windows: dict[int, list[list[str]]],
    excluded: dict[int, set[int]],
    n: int,
    seed: int,
    k: int = 6,
    taken: frozenset[tuple[int, int]] = frozenset(),
) -> dict[int, list[Passage]]:
    """Give each target feature an ordered list of n candidate negatives.

    own_windows: every stored window of each target (its keys are the targets).
    excluded: for each target, the features its negatives must not come from
    (the feature whose explanation is its placebo).
    A candidate never comes from the target itself or an excluded feature, shares
    no run of k tokens with a stored window of the target, and is given to one
    target only. The pool is shuffled once with the seed, then targets take turns
    so that none is left with only the end of the pool.
    taken: (feature, window) of passages already given to other targets. A
    replacement feature (PLAN.md 2.3, rule 6) is assigned alone, after the others,
    with every passage of their lists marked as taken.
    """
    order = list(pool)
    random.Random(seed).shuffle(order)
    targets = sorted(own_windows)
    own_grams = {t: set().union(*(kgrams(w, k) for w in own_windows[t])) for t in targets}
    used: set[int] = set()
    pointer = dict.fromkeys(targets, 0)
    out: dict[int, list[Passage]] = {t: [] for t in targets}
    for _ in range(n):
        for t in targets:
            banned = {t} | excluded.get(t, set())
            while True:
                if pointer[t] >= len(order):
                    raise ValueError(f"pool too small: feature {t} has {len(out[t])} of {n}")
                j = pointer[t]
                pointer[t] += 1
                p = order[j]
                if (
                    j in used
                    or (p.feature, p.window) in taken
                    or p.feature in banned
                    or kgrams(list(p.tokens), k) & own_grams[t]
                ):
                    continue
                used.add(j)
                out[t].append(p)
                break
    return out
