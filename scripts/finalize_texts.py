"""Checks on the built texts before freezing (PLAN.md 2.2 and 2.6, rule 10).

Usage: uv run scripts/finalize_texts.py [--balance]

Reports, over every built feature: features failed or missing, the share of texts
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


def balance(records: dict[int, dict], order: list[int]) -> list[dict]:
    """Replace separator negatives until the gap is at most MAX_GAP. Returns the swaps."""
    cands = json.loads((DIR / "candidates.json").read_text())
    swaps = []
    for f in order:
        sp, sn = shares(records)
        if abs(sp - sn) <= MAX_GAP or sn <= sp:
            break
        rec = records[f]
        checked = {(x["from_feature"], x["window"]) for x in rec["negatives"]["kept"]}
        checked |= {(x["from_feature"], x["window"]) for x in rec["negatives"]["rejected"]}
        spare = [c for c in cands[str(f)] if (c["feature"], c["window"]) not in checked]
        for i, neg in enumerate(rec["negatives"]["kept"]):
            if SEP not in neg["text"]:
                continue
            while spare:
                c = spare.pop(0)
                text = scorer_text(c["tokens"])
                if SEP in text:
                    continue
                value = peak(activation(f, text)["values"])
                if value == 0:
                    new = {
                        "from_feature": c["feature"],
                        "window": c["window"],
                        "text": text,
                        "recomputed": value,
                    }
                    rec["negatives"]["kept"][i] = new
                    swaps.append({"feature": f, "out": neg, "in": new})
                    break
                rec["negatives"]["rejected"].append(
                    {
                        "from_feature": c["feature"],
                        "window": c["window"],
                        "text": text,
                        "recomputed": value,
                    }
                )
        (DIR / f"{f}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return swaps


def main(do_balance: bool) -> None:
    sample = json.loads((ROOT / "data/draw/draw.json").read_text())["sample"]
    records, missing, failed = {}, [], []
    for f in sample:
        path = DIR / f"{f}.json"
        if not path.exists():
            missing.append(f)
            continue
        rec = json.loads(path.read_text())
        (records.__setitem__(f, rec) if rec["status"] == "ok" else failed.append(f))
    swaps = balance(records, [f for f in sample if f in records]) if do_balance else []
    sp, sn = shares(records)
    enc = tiktoken.get_encoding("gpt2")

    def mean_tokens(kind: str) -> float:
        texts = [t["text"] for r in records.values() for t in r[kind]["kept"]]
        return sum(len(enc.encode(t, allowed_special="all")) for t in texts) / len(texts)

    summary = {
        "features_ok": len(records),
        "features_failed": failed,
        "features_missing": missing,
        "separator_share": {"positives": round(sp, 4), "negatives": round(sn, 4)},
        "separator_gap_ok": abs(sp - sn) <= MAX_GAP,
        "separator_swaps": swaps,
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
    print(json.dumps({k: v for k, v in summary.items() if k != "separator_swaps"}, indent=1))


if __name__ == "__main__":
    main("--balance" in sys.argv[1:])
