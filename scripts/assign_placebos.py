"""Assign one placebo to each of the 100 sampled features (PLAN.md, Placebos, round 1).

Reads the draw (data/draw/draw.json) and the cached Neuronpedia JSON, no network.
Writes data/placebos/round1/assignment.json: for each target, its explanations A and
B, the donor, which of the donor's explanations is the placebo, and the placebo text.
"""

import json

from autointerp.neuronpedia import ROOT, fetch_feature
from autointerp.placebo import SEED, assign, build_pairs, rows


def main() -> None:
    sample = json.loads((ROOT / "data" / "draw" / "draw.json").read_text())["sample"]
    pairs = build_pairs(sample, fetch_feature)
    m, swaps = assign(pairs, seed=SEED)
    out = ROOT / "data" / "placebos" / "round1" / "assignment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"seed": SEED, "swaps": swaps, "pairs": rows(pairs, m, sample)}, indent=1)
    )
    print(f"{len(sample)} placebos, {swaps} swaps, written {out}")


if __name__ == "__main__":
    main()
