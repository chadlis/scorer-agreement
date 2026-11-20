"""Assign the placebos from the donor pool of rule 8b (PLAN.md, Placebos).

Usage: uv run scripts/assign_donor_pool.py N   (N >= 3)

Reads the specific/generic classes (data/placebos/generic.json), the candidate
outside donors (data/placebos/donor_candidates.json) and the decisions of earlier
rounds. Sampled features with two generic explanations leave the donor pool, and
every candidate with at least one specific explanation joins it, so the pool holds
more donors than targets. A new derangement of the targets onto the pool (seed
20261120 + N) is repaired, with unused donors as spares, until no pair fails a
filter, uses a generic placebo or was rejected before.
Writes data/placebos/roundN/assignment.json.
"""

import json
import random
import sys

from autointerp.neuronpedia import ROOT, fetch_feature
from autointerp.placebo import SEED, build_pairs, derangement, donor_pool, repair, rows

DIR = ROOT / "data" / "placebos"


def main(n: int) -> None:
    sample = json.loads((ROOT / "data" / "draw" / "draw.json").read_text())["sample"]
    classes = json.loads((DIR / "generic.json").read_text())["explanations"]
    specific = {(e["index"], e["model"]) for e in classes if e["class"] == "specific"}
    candidates = json.loads((DIR / "donor_candidates.json").read_text())["candidates"]

    donors, dropped, outside = donor_pool(sample, specific, [c["index"] for c in candidates])

    forbidden = set()
    for k in range(1, n):
        dec = json.loads((DIR / f"round{k}" / "decisions.json").read_text())
        old = json.loads((DIR / f"round{k}" / "assignment.json").read_text())["pairs"]
        donor_of = {r["target"]: r["donor"] for r in old}
        rejected = dec["rejected_doubt"] + dec["confident_rejected"]
        rejected += [g["target"] for g in dec["rejected_generic"]]
        forbidden |= {(t, donor_of[t]) for t in rejected}

    pairs = build_pairs(sample, fetch_feature, extra_donors=outside, specific=specific)
    rng = random.Random(SEED + n)
    m = derangement(pairs.targets, rng, donors=donors)
    spare = sorted(set(donors) - set(m.values()))
    swaps = repair(m, pairs, rng, frozenset(forbidden), spare=spare)
    out = DIR / f"round{n}" / "assignment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SEED + n,
        "swaps": swaps,
        "dropped_donors": dropped,
        "outside_donors": outside,
        "unused_donors": sorted(spare),
        "pairs": rows(pairs, m, sample),
    }
    out.write_text(json.dumps(payload, indent=1))
    print(
        f"round {n}: {len(dropped)} sampled features out of the donor pool, "
        f"{len(outside)} outside candidates, {len(spare)} donors unused, {swaps} swaps"
    )


if __name__ == "__main__":
    main(int(sys.argv[1]))
