"""Checks on the built texts before freezing (PLAN.md 2.2 and 2.6, rule 10).

Usage: uv run scripts/finalize_texts.py [--balance]

First, any negative that comes from its target's placebo donor (possible after a
placebo round that follows the build) is replaced by the next spare candidate whose
recomputed activation is exactly 0 (PLAN.md 2.2, Negatives, rule 2).

Reports, over every built feature: features failed, missing or replaced, the share of texts
holding a document separator among positives and among negatives, and the mean
length in GPT-2 tokens of the scorer's text. With --balance, if the separator shares
differ by more than 5 points, negatives holding a separator are replaced, feature by
feature in draw order, by the next unchecked candidates without one whose recomputed
activation is exactly 0 (free activation calls), until the gap is at most 5 points.
Writes data/texts/summary.json.
"""

import json
import sys

import tiktoken

from autointerp.activation import activation
from autointerp.neuronpedia import ROOT, scorer_text
from autointerp.texts import peak

DIR = ROOT / "data" / "texts"
SEP = "<|endoftext|>"
MAX_GAP = 0.05


def shares(records: dict[int, dict]) -> tuple[float, float]:
    pos = [t["text"] for r in records.values() for t in r["positives"]["kept"]]
    neg = [t["text"] for r in records.values() for t in r["negatives"]["kept"]]
    return sum(SEP in t for t in pos) / len(pos), sum(SEP in t for t in neg) / len(neg)


def spare_candidates(f: int, rec: dict, cands: dict) -> list[dict]:
    """Candidates of f not yet checked, in their seeded order."""
    checked = {(x["from_feature"], x["window"]) for x in rec["negatives"]["kept"]}
    checked |= {(x["from_feature"], x["window"]) for x in rec["negatives"]["rejected"]}
    checked |= {(x["out"]["from_feature"], x["out"]["window"]) for x in replaced(rec)}
    return [c for c in cands[str(f)] if (c["feature"], c["window"]) not in checked]


def replaced(rec: dict) -> list[dict]:
    """Negatives swapped out after the build, kept in the feature's record."""
    return rec["negatives"].setdefault("replaced", [])


def swap(f: int, rec: dict, i: int, new: dict, reason: str) -> None:
    replaced(rec).append({"reason": reason, "out": rec["negatives"]["kept"][i], "in": new})
    rec["negatives"]["kept"][i] = new
    (DIR / f"{f}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))


def take(f: int, rec: dict, spare: list[dict], ok) -> dict | None:
    """Next spare candidate accepted by ok(candidate, text) whose recomputed
    activation for f is exactly 0. Candidates found active are added to rejected."""
    while spare:
        c = spare.pop(0)
        text = scorer_text(c["tokens"])
        if not ok(c, text):
            continue
        value = peak(activation(f, text)["values"])
        entry = {"from_feature": c["feature"], "window": c["window"], "text": text}
        entry["recomputed"] = value
        if value == 0:
            return entry
        rec["negatives"]["rejected"].append(entry)
    return None


def exclude_donors(records: dict[int, dict], donor: dict[int, int]) -> int:
    """PLAN.md 2.2, Negatives, rule 2: no negative comes from the feature whose
    explanation is the target's placebo. A placebo round after the build can break
    it, so offending negatives are replaced by the next spare candidates. Returns
    the number replaced in this run."""
    cands = json.loads((DIR / "candidates.json").read_text())
    swaps = 0
    for f, rec in records.items():
        spare = spare_candidates(f, rec, cands)
        for i, neg in enumerate(rec["negatives"]["kept"]):
            if neg["from_feature"] != donor[f]:
                continue
            new = take(f, rec, spare, lambda c, text, f=f: c["feature"] != donor[f])
            if new is None:
                raise SystemExit(f"feature {f}: no spare candidate left")
            swap(f, rec, i, new, "placebo donor")
            swaps += 1
    return swaps


def balance(records: dict[int, dict], order: list[int], donor: dict[int, int]) -> int:
    """Replace separator negatives until the gap is at most MAX_GAP. Returns the
    number replaced in this run."""
    cands = json.loads((DIR / "candidates.json").read_text())
    swaps = 0
    for f in order:
        sp, sn = shares(records)
        if abs(sp - sn) <= MAX_GAP or sn <= sp:
            break
        rec = records[f]
        spare = spare_candidates(f, rec, cands)
        for i, neg in enumerate(rec["negatives"]["kept"]):
            if SEP not in neg["text"]:
                continue
            new = take(
                f,
                rec,
                spare,
                lambda c, text, f=f: SEP not in text and c["feature"] != donor[f],
            )
            if new is None:
                break
            swap(f, rec, i, new, "separator balance")
            swaps += 1
    return swaps


def main(do_balance: bool) -> None:
    drawn = json.loads((ROOT / "data/draw/draw.json").read_text())
    sample = drawn["sample"]
    records, missing, failed = {}, [], []
    for f in sample:
        path = DIR / f"{f}.json"
        if not path.exists():
            missing.append(f)
            continue
        rec = json.loads(path.read_text())
        (records.__setitem__(f, rec) if rec["status"] == "ok" else failed.append(f))
    final = json.loads((ROOT / "data/placebos/final.json").read_text())["pairs"]
    donor = {p["target"]: p["donor"] for p in final}
    donor_swaps = exclude_donors(records, donor)
    order = [f for f in sample if f in records]
    swaps = balance(records, order, donor) if do_balance else 0
    sp, sn = shares(records)
    enc = tiktoken.get_encoding("gpt2")

    def mean_tokens(kind: str) -> float:
        texts = [t["text"] for r in records.values() for t in r[kind]["kept"]]
        return sum(len(enc.encode(t, allowed_special="all")) for t in texts) / len(texts)

    summary = {
        "features_ok": len(records),
        "features_failed": failed,
        "features_missing": missing,
        "features_replaced": drawn.get("replacements", []),
        "negatives_replaced": [
            {"feature": f, **x} for f, r in records.items() for x in replaced(r)
        ],
        "separator_share": {"positives": round(sp, 4), "negatives": round(sn, 4)},
        "separator_gap_ok": abs(sp - sn) <= MAX_GAP,
        "mean_tokens": {
            "positives": round(mean_tokens("positives"), 2),
            "negatives": round(mean_tokens("negatives"), 2),
        },
        "grey_and_zero_positives": sum(
            len(r["positives"]["grey"]) + len(r["positives"]["zero"]) for r in records.values()
        ),
        "negatives_rejected": sum(len(r["negatives"]["rejected"]) for r in records.values()),
    }
    (DIR / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "negatives_replaced"}, indent=1))
    print(
        f"this run: {donor_swaps} negatives from a placebo donor replaced, {swaps} for balance; "
        f"{len(summary['negatives_replaced'])} replaced in all"
    )


if __name__ == "__main__":
    main("--balance" in sys.argv[1:])
