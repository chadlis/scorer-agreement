"""Draw the 100 features of the study with the committed seed (PLAN.md, Feature selection).

Free Neuronpedia calls only: one feature download per examined feature (cached) and
one activation call for the gate of each feature that passes conditions (a) to (c).
Writes data/draw/order.json (seed, Python version, shuffled order) and
data/draw/draw.json (sample, pilot, every examined feature with its outcome, and the
position to resume from for replacements).
"""

import json
import platform

from autointerp.activation import activation
from autointerp.draw import SEED, draw, shuffled_order
from autointerp.neuronpedia import ROOT, fetch_feature

OUT = ROOT / "data" / "draw"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    order = shuffled_order(SEED)
    (OUT / "order.json").write_text(
        json.dumps({"seed": SEED, "python": platform.python_version(), "order": order})
    )

    def fetch(index: int) -> dict:
        raw = fetch_feature(index)
        print(f"feature {index}", flush=True)
        return raw

    result = draw(order, fetch, lambda index, text: activation(index, text))
    result["seed"] = SEED
    (OUT / "draw.json").write_text(json.dumps(result, indent=1))
    failed = [r["failed"] for r in result["examined"] if r["failed"]]
    print(
        f"sample {len(result['sample'])}, examined {len(result['examined'])},",
        {f: failed.count(f) for f in sorted(set(failed))},
    )


if __name__ == "__main__":
    main()
