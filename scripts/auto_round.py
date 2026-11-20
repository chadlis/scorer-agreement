"""Automatic placebo rounds (PLAN.md, Placebos, rule 9).

Usage:
  uv run scripts/auto_round.py assign N   repair round N-1 into round N, list the
                                          changed pairs for the triage
  uv run scripts/auto_round.py close N    merge the triage of the changed pairs with
                                          the verdicts carried over, reject every doubt

assign writes data/placebos/roundN/assignment.json and to_triage.json. The triage
agent writes roundN/triage_changed.json. close writes roundN/triage.json and
roundN/decisions.json.
"""

import json
import random
import sys

from autointerp.neuronpedia import ROOT, fetch_feature
from autointerp.placebo import SEED, build_pairs, donor_pool, repair, rows

DIR = ROOT / "data" / "placebos"


def load(path):
    return json.loads(path.read_text())


def rejected_pairs(n: int) -> set[tuple[int, int]]:
    """Every (target, donor) pair rejected in rounds 1 to n-1."""
    out = set()
    for k in range(1, n):
        dec = load(DIR / f"round{k}" / "decisions.json")
        donor_of = {
            r["target"]: r["donor"] for r in load(DIR / f"round{k}" / "assignment.json")["pairs"]
        }
        rejected = dec["rejected_doubt"] + dec["confident_rejected"]
        rejected += [g["target"] for g in dec["rejected_generic"]]
        out |= {(t, donor_of[t]) for t in rejected}
    return out


def assign(n: int) -> None:
    sample = load(ROOT / "data" / "draw" / "draw.json")["sample"]
    classes = load(DIR / "generic.json")["explanations"]
    specific = {(e["index"], e["model"]) for e in classes if e["class"] == "specific"}
    candidates = [c["index"] for c in load(DIR / "donor_candidates.json")["candidates"]]
    donors, dropped, outside = donor_pool(sample, specific, candidates)
    pairs = build_pairs(sample, fetch_feature, extra_donors=outside, specific=specific)
    prev = load(DIR / f"round{n - 1}" / "assignment.json")["pairs"]
    m = {r["target"]: r["donor"] for r in prev}
    spare = sorted(set(donors) - set(m.values()))
    swaps = repair(
        m,
        pairs,
        random.Random(SEED + n),
        frozenset(rejected_pairs(n)),
        spare=spare,
        keep_valid=True,
    )
    out_rows = rows(pairs, m, sample)
    before = {r["target"]: r["placebo"] for r in prev}
    changed = [r for r in out_rows if r["placebo"] != before[r["target"]]]
    d = DIR / f"round{n}"
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SEED + n,
        "swaps": swaps,
        "changed": [r["target"] for r in changed],
        "unused_donors": sorted(spare),
        "pairs": out_rows,
    }
    (d / "assignment.json").write_text(json.dumps(payload, indent=1))
    (d / "to_triage.json").write_text(json.dumps({"pairs": changed}, indent=1))
    print(f"round {n}: {swaps} swaps, {len(changed)} placebos changed")


def close(n: int) -> None:
    d = DIR / f"round{n}"
    pairs = load(d / "assignment.json")["pairs"]
    new = {t["target"]: t for t in load(d / "triage_changed.json")["pairs"]}
    old = {t["target"]: t for t in load(DIR / f"round{n - 1}" / "triage.json")["pairs"]}
    changed = set(load(d / "assignment.json")["changed"])
    if set(new) != changed:
        raise SystemExit("the triage does not cover exactly the changed pairs")
    merged = [
        {
            **(new[r["target"]] if r["target"] in changed else old[r["target"]]),
            "carried_over": r["target"] not in changed,
        }
        for r in pairs
    ]
    doubts = [t["target"] for t in merged if t["verdict"] == "doubt"]
    (d / "triage.json").write_text(
        json.dumps({"model": "claude-fable-5-2", "pairs": merged}, indent=1)
    )
    decisions = {
        "round": n,
        "reviewer": "automatic (rule 9)",
        "rejected_doubt": doubts,
        "confident_checked": [],
        "confident_rejected": [],
        "rejected_generic": [],
    }
    (d / "decisions.json").write_text(json.dumps(decisions, indent=1))
    print(f"round {n}: {len(changed)} pairs triaged, {len(doubts)} doubts rejected")


if __name__ == "__main__":
    {"assign": assign, "close": close}[sys.argv[1]](int(sys.argv[2]))
