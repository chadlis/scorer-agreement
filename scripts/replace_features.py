"""Replace the sampled features whose texts failed (PLAN.md 2.3, rule 6).

Usage: uv run scripts/replace_features.py   (free Neuronpedia calls only)

A feature whose record in data/texts has status "failed" (fewer than 20 positives
or 20 negatives) is replaced by the next eligible feature of the seeded order, from
the saved position. The replacement takes the place of the replaced feature in the
sample. data/draw/draw.json gets the new sample, the examined features appended, the
new position, and a "replacements" list. The record of the replaced feature stays in
data/texts. The replacement then needs a placebo (scripts/auto_round.py) and its
texts (scripts/build_texts.py).
"""

import json

from autointerp.activation import activation
from autointerp.draw import draw
from autointerp.neuronpedia import ROOT, fetch_feature

DRAW = ROOT / "data" / "draw"
TEXTS = ROOT / "data" / "texts"


def main() -> None:
    result = json.loads((DRAW / "draw.json").read_text())
    order = json.loads((DRAW / "order.json").read_text())["order"]
    result.setdefault("replacements", [])
    for f in list(result["sample"]):
        path = TEXTS / f"{f}.json"
        if not path.exists():
            continue
        rec = json.loads(path.read_text())
        if rec["status"] != "failed":
            continue
        step = draw(order, fetch_feature, activation, n=1, start=result["next_position"])
        if not step["sample"]:
            raise SystemExit("the seeded order is exhausted")
        new = step["sample"][0]
        i = result["sample"].index(f)
        result["sample"][i] = new
        result["examined"] += step["examined"]
        result["next_position"] = step["next_position"]
        reason = {
            "positives": rec["positives"]["positives"],
            "negatives": len(rec["negatives"]["kept"]),
        }
        result["replacements"].append({"out": f, "in": new, "labelled": reason})
        print(f"feature {f} {reason} replaced by {new}")
    if result["pilot"] != result["sample"][:5]:
        raise SystemExit("a pilot feature was replaced: the pilot changes, stop and decide")
    (DRAW / "draw.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
