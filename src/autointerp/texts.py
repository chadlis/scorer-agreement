"""Build and label the 40 texts of each feature (PLAN.md, Labels, Negatives, Windows).

Positives: one window per distinct positive passage of the feature (the one with the
highest stored activation), recomputed on the scorer's text. Those at or above 10% of
the stored maximum are positives, those strictly between 0 and 10% are the grey
zone. The 20 kept sit at evenly spaced ranks of the recomputed activation.
Negatives: candidates borrowed from the other sampled features, checked in order
until 20 have a recomputed maximum of exactly 0.
Network access is passed in as a function, so the logic is tested offline.
"""

from collections.abc import Callable

from autointerp.negatives import Passage
from autointerp.neuronpedia import scorer_text, source_groups

N_POS = N_NEG = 20
POSITIVE_SHARE = 0.10


def peak(values: list[float]) -> float:
    return max(values, default=0)


def positive_passages(raw: dict, k: int = 6) -> list[dict]:
    """One window per distinct positive passage: the one with the highest stored peak."""
    pos = [(i, w) for i, w in enumerate(raw["activations"]) if peak(w["values"]) > 0]
    groups = source_groups([w["tokens"] for _, w in pos], k)
    best: dict[int, tuple[int, dict]] = {}
    for (i, w), g in zip(pos, groups, strict=True):
        if g not in best or peak(w["values"]) > peak(best[g][1]["values"]):
            best[g] = (i, w)
    return [
        {"window": i, "tokens": w["tokens"], "stored": peak(w["values"])}
        for i, w in sorted(best.values())
    ]


def even_ranks(n_available: int, n: int = N_POS) -> list[int]:
    """Indices of n items at evenly spaced ranks among n_available sorted items."""
    if n_available < n:
        raise ValueError(f"{n_available} items for {n}")
    return [round(j * (n_available - 1) / (n - 1)) for j in range(n)]


def build_positives(raw: dict, activation: Callable[[str], dict]) -> dict:
    """Recompute every positive passage, label it, keep 20 at evenly spaced ranks."""
    stored_max = raw["maxActApprox"]
    rows, grey, zero = [], [], []
    for p in positive_passages(raw):
        text = scorer_text(p["tokens"])
        value = peak(activation(text)["values"])
        row = {"window": p["window"], "text": text, "stored": p["stored"], "recomputed": value}
        if value >= POSITIVE_SHARE * stored_max:
            rows.append(row)
        elif value > 0:
            grey.append(row)
        else:
            zero.append(row)
    rows.sort(key=lambda r: (r["recomputed"], r["window"]))
    kept = [rows[i] for i in even_ranks(len(rows))] if len(rows) >= N_POS else []
    return {"kept": kept, "positives": len(rows), "grey": grey, "zero": zero}


def build_negatives(candidates: list[Passage], activation: Callable[[str], dict]) -> dict:
    """Check candidates in order until 20 have a recomputed maximum of exactly 0."""
    kept, rejected = [], []
    for c in candidates:
        if len(kept) == N_NEG:
            break
        text = scorer_text(list(c.tokens))
        value = peak(activation(text)["values"])
        row = {"from_feature": c.feature, "window": c.window, "text": text, "recomputed": value}
        (kept if value == 0 else rejected).append(row)
    return {"kept": kept if len(kept) == N_NEG else [], "rejected": rejected}
