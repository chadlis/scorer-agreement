"""Draw the 100 features of the study (PLAN.md, Feature selection).

Simple random sampling among eligible features: the indices are shuffled once with
the published seed, then examined in that order. The first eligible features form
the sample. Network access is passed in as functions, so the logic is tested offline.
"""

import random
from collections.abc import Callable

from autointerp.neuronpedia import scorer_text, source_groups

SEED = 20261120
N_FEATURES = 24576  # gpt2-small/6-res-jb
FRAC_MIN, FRAC_MAX = 1e-5, 1e-2
A_MODEL, B_MODEL = "gpt-3.5-turbo", "gpt-4o-mini"
MIN_POS_PASSAGES = 20
POSITIVE_SHARE = 0.10  # Labels, rule 3: at least 10% of the stored maximum


def shuffled_order(seed: int = SEED, n: int = N_FEATURES) -> list[int]:
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order


def positive_windows(raw: dict) -> list[dict]:
    return [w for w in raw.get("activations", []) if max(w["values"], default=0) > 0]


def check_stored(raw: dict) -> str | None:
    """Conditions (a) to (c), read from the stored JSON. Returns the failure, or None."""
    frac = raw.get("frac_nonzero")
    if frac is None or not FRAC_MIN < frac < FRAC_MAX:
        return "frequency"
    models = {e.get("explanationModelName") for e in raw.get("explanations", [])}
    if A_MODEL not in models or B_MODEL not in models:
        return "explanations"
    pos = positive_windows(raw)
    if len(set(source_groups([w["tokens"] for w in pos]))) < MIN_POS_PASSAGES:
        return "positives"
    return None


def gate(raw: dict, activation: Callable[[str], dict]) -> tuple[bool, float]:
    """Condition (d): the window holding the stored maximum, recomputed on the scorer's
    text, must stay a positive. Returns (passed, recomputed max / stored max)."""
    top = max(positive_windows(raw), key=lambda w: max(w["values"]))
    recomputed = max(activation(scorer_text(top["tokens"]))["values"], default=0)
    share = recomputed / raw["maxActApprox"]
    return share >= POSITIVE_SHARE, share


def draw(
    order: list[int],
    fetch: Callable[[int], dict],
    activation: Callable[[int, str], dict],
    n: int = 100,
    start: int = 0,
) -> dict:
    """Examine `order` from `start` until n features are eligible."""
    sample, examined = [], []
    pos = start
    while len(sample) < n and pos < len(order):
        index = order[pos]
        pos += 1
        raw = fetch(index)
        row = {"index": index, "frac_nonzero": raw.get("frac_nonzero"), "failed": check_stored(raw)}
        if row["failed"] is None:
            passed, share = gate(raw, lambda text, i=index: activation(i, text))
            row["gate_share"] = round(share, 4)
            if not passed:
                row["failed"] = "gate"
        examined.append(row)
        if row["failed"] is None:
            sample.append(index)
    return {"sample": sample, "pilot": sample[:5], "examined": examined, "next_position": pos}
