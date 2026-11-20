"""Assign one placebo to each of the 100 sampled features (PLAN.md, Placebos).

Reads the draw (data/draw/draw.json) and the cached Neuronpedia JSON, no network.
Writes data/placebos/assignment.json: for each target, its explanations A and B,
the donor, which of the donor's explanations is the placebo, and the placebo text.
"""

import json

from autointerp.neuronpedia import ROOT, fetch_feature
from autointerp.placebo import SEED, Pairs, assign

A_MODEL, B_MODEL = "gpt-3.5-turbo", "gpt-4o-mini"


def explanation(raw: dict, model: str) -> str:
    return next(
        e["description"].strip() for e in raw["explanations"] if e["explanationModelName"] == model
    )


def main() -> None:
    sample = json.loads((ROOT / "data" / "draw" / "draw.json").read_text())["sample"]
    raws = {i: fetch_feature(i) for i in sample}
    a = {i: explanation(raws[i], A_MODEL) for i in sample}
    b = {i: explanation(raws[i], B_MODEL) for i in sample}
    nb = {i: set(raws[i]["topkCosSimIndices"]) - {i} for i in sample}
    pairs = Pairs(a, b, nb, seed=SEED)
    m, swaps = assign(pairs, seed=SEED)
    rows = [
        {
            "target": t,
            "target_A": a[t],
            "target_B": b[t],
            "donor": m[t],
            "donor_model": pairs.model[t],
            "placebo": pairs.placebo(t, m[t]),
        }
        for t in sample
    ]
    out = ROOT / "data" / "placebos" / "assignment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seed": SEED, "swaps": swaps, "pairs": rows}, indent=1))
    print(f"{len(rows)} placebos, {swaps} swaps, written {out}")


if __name__ == "__main__":
    main()
