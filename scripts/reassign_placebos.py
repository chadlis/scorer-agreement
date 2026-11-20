"""Reassign the placebos rejected at review (PLAN.md, section 2.4, rules 6 and 9).

Usage: uv run scripts/reassign_placebos.py N   (N >= 2)

Starts from round N-1's assignment, forbids every (target, donor) pair and every
placebo text rejected in rounds 1 to N-1 (data/placebos/roundK/decisions.json), and
repairs with the seeded swaps. Writes data/placebos/roundN/assignment.json, with
the list of targets whose placebo changed.
"""

import json
import random
import sys

from autointerp.neuronpedia import ROOT, fetch_feature
from autointerp.placebo import SEED, build_pairs, repair, rows

DIR = ROOT / "data" / "placebos"


def main(n: int) -> None:
    sample = json.loads((ROOT / "data" / "draw" / "draw.json").read_text())["sample"]
    pairs = build_pairs(sample, fetch_feature)
    prev = json.loads((DIR / f"round{n - 1}" / "assignment.json").read_text())["pairs"]
    m = {r["target"]: r["donor"] for r in prev}
    before = {r["target"]: (r["donor"], r["placebo"]) for r in prev}
    forbidden, texts = set(), set()
    for k in range(1, n):
        dec = json.loads((DIR / f"round{k}" / "decisions.json").read_text())
        old = {
            r["target"]: r
            for r in json.loads((DIR / f"round{k}" / "assignment.json").read_text())["pairs"]
        }
        forbidden |= {
            (t, old[t]["donor"]) for t in dec["rejected_doubt"] + dec["confident_rejected"]
        }
        for g in dec["rejected_generic"]:
            forbidden.add((g["target"], g["donor"]))
            texts.add(g["placebo"])
    swaps = repair(m, pairs, random.Random(SEED + n), frozenset(forbidden), frozenset(texts))
    out_rows = rows(pairs, m, sample)
    changed = [r["target"] for r in out_rows if (r["donor"], r["placebo"]) != before[r["target"]]]
    out = DIR / f"round{n}" / "assignment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"seed": SEED + n, "swaps": swaps, "changed": changed, "pairs": out_rows}, indent=1
        )
    )
    print(f"round {n}: {swaps} swaps, {len(changed)} placebos changed, written {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
