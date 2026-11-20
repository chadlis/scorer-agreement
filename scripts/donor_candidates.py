"""Collect candidate outside donors for the placebos (PLAN.md, Placebos, rule 8b).

Walks the draw order from the position where the draw stopped and keeps the first
30 features that pass conditions (a) to (c) of the sampling, with their two
explanations. Free Neuronpedia downloads, cached. No gate: donors supply no texts.
Writes data/placebos/donor_candidates.json.
"""

import json

from autointerp.draw import check_stored
from autointerp.neuronpedia import ROOT, fetch_feature
from autointerp.placebo import A_MODEL, B_MODEL

N_CANDIDATES = 30


def main() -> None:
    order = json.loads((ROOT / "data" / "draw" / "order.json").read_text())["order"]
    pos = json.loads((ROOT / "data" / "draw" / "draw.json").read_text())["next_position"]
    found, skipped = [], []
    while len(found) < N_CANDIDATES:
        index = order[pos]
        raw = fetch_feature(index)
        failed = check_stored(raw)
        if failed:
            skipped.append({"position": pos, "index": index, "failed": failed})
        else:
            expl = {
                e["explanationModelName"]: e["description"].strip() for e in raw["explanations"]
            }
            found.append({"position": pos, "index": index, "A": expl[A_MODEL], "B": expl[B_MODEL]})
        pos += 1
    out = ROOT / "data" / "placebos" / "donor_candidates.json"
    out.write_text(json.dumps({"candidates": found, "skipped": skipped}, indent=1))
    print(f"{len(found)} candidates, {len(skipped)} skipped, last position {pos - 1}")


if __name__ == "__main__":
    main()
