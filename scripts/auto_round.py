"""Automatic placebo rounds (PLAN.md, section 2.4, rules 10 and 11).

A feature replaced in the sample (data/draw/draw.json, "replacements") hands its
donor to its replacement, which is then repaired and triaged like a changed pair.

Usage:
  uv run scripts/auto_round.py assign N   repair round N-1 into round N, list the
                                          changed pairs for the triage
  uv run scripts/auto_round.py final N    write final.json from round N (no doubt left)
  uv run scripts/auto_round.py close N    merge the triage of the changed pairs with
                                          the verdicts carried over, reject every strong
                                          doubt (a doubt without strength counts as strong)

assign writes data/placebos/roundN/assignment.json and to_triage.json. The triage
agent writes roundN/triage_changed.json. close writes roundN/triage.json and
roundN/decisions.json.
"""

import datetime
import json
import random
import sys

from autointerp.neuronpedia import ROOT, fetch_feature
from autointerp.placebo import SEED, build_pairs, donor_pool, repair, rows

DIR = ROOT / "data" / "placebos"


def load(path):
    return json.loads(path.read_text())


def rejected_texts(n: int) -> set[str]:
    """Every placebo text rejected as generic in rounds 1 to n-1."""
    out = set()
    for k in range(1, n):
        dec = load(DIR / f"round{k}" / "decisions.json")
        out |= {g["placebo"] for g in dec["rejected_generic"]}
    return out


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
    drawn = load(ROOT / "data" / "draw" / "draw.json")
    sample = drawn["sample"]
    replacements = drawn.get("replacements", [])
    classes = load(DIR / "generic.json")["explanations"]
    specific = {(e["index"], e["model"]) for e in classes if e["class"] == "specific"}
    candidates = [c["index"] for c in load(DIR / "donor_candidates.json")["candidates"]]
    # A replaced feature stays a donor (PLAN.md 2.3, rule 6), like an outside donor.
    # A candidate donor drawn as a replacement is now a sampled feature.
    candidates = [c for c in candidates if c not in sample] + [r["out"] for r in replacements]
    donors, dropped, outside = donor_pool(sample, specific, candidates)
    pairs = build_pairs(sample, fetch_feature, extra_donors=outside, specific=specific)
    prev = load(DIR / f"round{n - 1}" / "assignment.json")["pairs"]
    m = {r["target"]: r["donor"] for r in prev}
    for r in replacements:  # the replacement starts from the freed donor
        if r["out"] in m:
            m[r["in"]] = m.pop(r["out"])
    spare = sorted(set(donors) - set(m.values()))
    swaps = repair(
        m,
        pairs,
        random.Random(SEED + n),
        frozenset(rejected_pairs(n)),
        frozenset(rejected_texts(n)),
        spare=spare,
        keep_valid=True,
    )
    out_rows = rows(pairs, m, sample)
    before = {r["target"]: r["placebo"] for r in prev}
    changed = [r for r in out_rows if r["placebo"] != before.get(r["target"])]
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
    doubts = [t for t in merged if t["verdict"] == "doubt"]
    # From round 13 on (section 2.4, rule 11): strong doubts rejected, light ones kept.
    strong = [t["target"] for t in doubts if t.get("strength", "strong") == "strong"]
    light = [t["target"] for t in doubts if t.get("strength") == "light"]
    (d / "triage.json").write_text(
        json.dumps({"model": "claude-fable-5-2", "pairs": merged}, indent=1)
    )
    decisions = {
        "round": n,
        "reviewer": "automatic (section 2.4, rule 10)",
        "rejected_doubt": strong,
        "light_doubt": light,
        "confident_checked": [],
        "confident_rejected": [],
        "rejected_generic": [],
    }
    (d / "decisions.json").write_text(json.dumps(decisions, indent=1))
    print(
        f"round {n}: {len(changed)} pairs triaged, {len(strong)} strong doubts rejected, "
        f"{len(light)} light doubts kept for review"
    )


def final(n: int) -> None:
    """Write data/placebos/final.json from round N, which must have no doubt left
    except light doubts that Salah kept (decisions.json, light_doubt_review)."""
    d = DIR / f"round{n}"
    dec = load(d / "decisions.json")
    reviewed = dec.get("light_doubt_review", {})
    if dec["rejected_doubt"] or set(dec["light_doubt"]) - set(reviewed.get("kept", [])):
        raise SystemExit(f"round {n} still has doubts not kept by Salah's review (rule 11)")
    reason = {t["target"]: t["reason"] for t in load(d / "triage.json")["pairs"]}
    out = load(DIR / "final.json")
    out["history"].append(
        {
            "round": n,
            "rejected_pairs": 0,
            "rejected_texts": [],
            "changed": load(d / "assignment.json")["changed"],
            "light_doubts_kept": reviewed.get("kept", []),
        }
    )
    out["frozen"] = datetime.date.today().isoformat()
    out["from_round"] = n
    out["pairs"] = [
        {**r, "triage_reason": reason[r["target"]]} for r in load(d / "assignment.json")["pairs"]
    ]
    (DIR / "final.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"final.json written from round {n}")


if __name__ == "__main__":
    {"assign": assign, "close": close, "final": final}[sys.argv[1]](int(sys.argv[2]))
