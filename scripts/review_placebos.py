"""List the placebo pairs Salah reviews (PLAN.md, section 2.4, rule 8).

Usage: uv run scripts/review_placebos.py N

Reads data/placebos/roundN/assignment.json and its triage (triage.json). Round 1:
every doubt pair plus 20 confident pairs drawn with the seed, to check the triage.
Later rounds: every doubt pair. Writes data/placebos/roundN/review.json.
"""

import json
import sys

from autointerp.neuronpedia import ROOT
from autointerp.placebo import SEED, review_sample


def main(n: int) -> None:
    d = ROOT / "data" / "placebos" / f"round{n}"
    pairs = {p["target"]: p for p in json.loads((d / "assignment.json").read_text())["pairs"]}
    triage = json.loads((d / "triage.json").read_text())
    verdict = {t["target"]: t for t in triage["pairs"]}
    doubts = [t for t in pairs if verdict[t]["verdict"] == "doubt"]
    confident = [t for t in pairs if verdict[t]["verdict"] == "confident"]
    checked = review_sample(confident, 20, SEED) if n == 1 else []
    out_rows = [
        {**pairs[t], "triage": verdict[t]["verdict"], "reason": verdict[t]["reason"]}
        for t in doubts + checked
    ]
    out = d / "review.json"
    out.write_text(
        json.dumps({"seed": SEED, "triage_model": triage["model"], "pairs": out_rows}, indent=1)
    )
    print(
        f"round {n}: {len(doubts)} doubt + {len(checked)} confident pairs to review, written {out}"
    )


if __name__ == "__main__":
    main(int(sys.argv[1]))
