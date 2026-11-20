"""List the placebo pairs Salah reviews (PLAN.md, Placebos, rule 6).

Reads data/placebos/assignment.json and the triage data/placebos/triage.json,
draws 20 confident pairs with the seed, and writes data/placebos/review.json:
every doubt pair plus the 20 drawn confident pairs, with the triage reasons.
"""

import json

from autointerp.neuronpedia import ROOT
from autointerp.placebo import SEED, review_sample

DIR = ROOT / "data" / "placebos"


def main() -> None:
    pairs = {p["target"]: p for p in json.loads((DIR / "assignment.json").read_text())["pairs"]}
    triage = json.loads((DIR / "triage.json").read_text())
    verdict = {t["target"]: t for t in triage["pairs"]}
    doubts = [t for t in pairs if verdict[t]["verdict"] == "doubt"]
    checked = review_sample([t for t in pairs if verdict[t]["verdict"] == "confident"], 20, SEED)
    rows = [
        {**pairs[t], "triage": verdict[t]["verdict"], "reason": verdict[t]["reason"]}
        for t in doubts + checked
    ]
    out = DIR / "review.json"
    out.write_text(
        json.dumps({"seed": SEED, "triage_model": triage["model"], "pairs": rows}, indent=1)
    )
    print(f"{len(doubts)} doubt + {len(checked)} confident pairs to review, written {out}")


if __name__ == "__main__":
    main()
