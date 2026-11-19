"""Format check of the scorer prompt: 19 texts x 5 scorers, explanation A.

This is not a measurement. It only counts answers that the parsing rule cannot
read, before the prompt is frozen. Texts: windows of Neuronpedia feature 100
(gpt2-small/6-res-jb), one per source text: 14 where the feature fires (spread
over activation levels) and every remaining one where it does not (5 on
2026-11-19). Texts keep <|endoftext|> as is. How texts are built for the
measurement is decided at step 2, not here.

Usage: uv run scripts/check_format.py [--yes] [scorer ...]
Without --yes, nothing is sent: the script prints the planned calls and exits.
"""

import json
import sys
from collections import Counter

from autointerp.clients import ROOT, Clients, call_scorer, load_env
from autointerp.models import SCORERS
from autointerp.neuronpedia import fetch_feature, source_groups, window_text
from autointerp.prompt import SYSTEM, build_user

FEATURE = 100
N_POS = 14


def pick_windows(feature: dict) -> list[dict]:
    acts = sorted(feature["activations"], key=lambda a: -a["maxValue"])
    groups = source_groups([a["tokens"] for a in acts])
    seen, pos, neg = set(), [], []
    for a, g in zip(acts, groups, strict=True):
        if g in seen:
            continue
        seen.add(g)
        (pos if a["maxValue"] > 0 else neg).append(a)
    step = (len(pos) - 1) / (N_POS - 1)
    pos = [pos[round(i * step)] for i in range(N_POS)]
    return [
        {
            "max": round(a["maxValue"], 2),
            "truth": int(a["maxValue"] > 0),
            "text": window_text(a["tokens"]),
        }
        for a in pos + neg
    ]


def main() -> None:
    names = [a for a in sys.argv[1:] if a != "--yes"] or list(SCORERS)
    feature = fetch_feature(FEATURE)
    items = pick_windows(feature)
    expl = feature["explanations"][0]["description"].strip()
    n_pos = sum(i["truth"] for i in items)
    n_neg = len(items) - n_pos
    print(f"{len(items)} texts ({n_pos} positive, {n_neg} negative), explanation: {expl!r}")
    if "--yes" not in sys.argv[1:]:
        print(f"Dry run, nothing sent. Planned paid calls: {len(items)} x {', '.join(names)}")
        print("Rerun with --yes to send them.")
        return
    load_env()
    clients = Clients()
    out = ROOT / "data/smoke/format_check.jsonl"
    for name in names:
        answers = Counter()
        for k, item in enumerate(items):
            rec = call_scorer(clients, name, SYSTEM, build_user(expl, item["text"]))
            rec.update(feature=FEATURE, item=k, truth=item["truth"], max_activation=item["max"])
            with out.open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            answers[rec.get("answer", "error")] += 1
            if rec.get("answer") == "invalid" or not rec["ok"]:
                print(f"  {name} item {k}: {rec.get('text', rec.get('error'))!r}")
        print(f"{name:14} {dict(answers)}")


if __name__ == "__main__":
    main()
