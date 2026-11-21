"""Draw the divergent texts that Salah reads (CLAUDE.md: read about thirty texts
where the scorers disagree). No call.

Usage: uv run scripts/sample_divergent.py
Writes data/analysis/divergent_sample.json (the 30 texts to read, with the scorer
names) and data/analysis/divergent_blind.json (the same texts with the scorers
renamed S1 to S5 in a seeded order, for a reader who must not know which model
answered what).

Rule, fixed before reading: an item is a (feature, text, condition) with condition
A or B, read on pass 1 (as Figure 1). It is divergent if all five scorers answered
1 or 0 and they are not unanimous. Items are shuffled with seed 20261120, then
taken in that order, 15 positives and 15 negatives, at most 2 per feature.
"""

import json
import random

import pandas as pd

from autointerp.analysis import SEED
from autointerp.neuronpedia import ROOT, fetch_feature
from autointerp.report import load_grid
from autointerp.run import load_explanations

OUT = ROOT / "data" / "analysis"
PER_LABEL, PER_FEATURE = 15, 2


def main() -> None:
    calls = load_grid()
    c = calls[(calls["pass"] == 1) & calls.condition.isin(["A", "B"])]
    wide = c.pivot_table(
        index=["feature", "text", "condition", "label"],
        columns="scorer",
        values="answer",
        aggfunc="first",
    )
    valid = wide.isin(["0", "1"]).all(axis=1)
    unanimous = wide.nunique(axis=1) == 1
    divergent = wide[valid & ~unanimous]
    counts = {
        "items": int(len(wide)),
        "all_valid": int(valid.sum()),
        "divergent": int(len(divergent)),
        "divergent_positives": int((divergent.index.get_level_values("label") == 1).sum()),
        "divergent_negatives": int((divergent.index.get_level_values("label") == 0).sum()),
    }
    keys = list(divergent.index)
    random.Random(SEED).shuffle(keys)
    taken, per_feature, per_label = [], {}, {0: 0, 1: 0}
    for k in keys:
        f, _, _, label = k
        if per_label[label] >= PER_LABEL or per_feature.get(f, 0) >= PER_FEATURE:
            continue
        taken.append(k)
        per_feature[f] = per_feature.get(f, 0) + 1
        per_label[label] += 1
        if len(taken) == 2 * PER_LABEL:
            break

    expl = load_explanations(sorted({k[0] for k in taken}))
    items = []
    for f, t, cond, label in taken:
        rec = json.loads((ROOT / "data/texts" / f"{f}.json").read_text())
        kind = "positives" if label == 1 else "negatives"
        entry = rec[kind]["kept"][t if label == 1 else t - 20]
        max_act = fetch_feature(f)["maxActApprox"]
        answers = divergent.loc[(f, t, cond, label)].to_dict()
        items.append(
            {
                "feature": int(f),
                "text_index": int(t),
                "condition": cond,
                "explanation": expl[f][cond],
                "other_explanation": expl[f]["B" if cond == "A" else "A"],
                "text": entry["text"],
                "label": int(label),
                "activation": round(entry["recomputed"], 3),
                "activation_share_of_max": round(entry["recomputed"] / max_act, 3),
                "rank_among_positives": int(t) if label == 1 else None,
                "answers": answers,
                "yes": sum(a == "1" for a in answers.values()),
            }
        )
    scorers = sorted(items[0]["answers"])
    alias = scorers[:]
    random.Random(SEED + 1).shuffle(alias)
    names = {s: f"S{i + 1}" for i, s in enumerate(alias)}
    blind = [
        {
            **{k: v for k, v in it.items() if k != "answers"},
            "answers": {
                names[s]: a for s, a in sorted(it["answers"].items(), key=lambda x: names[x[0]])
            },
        }
        for it in items
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {
        "rule": __doc__.split("Rule, fixed before reading: ")[1].strip(),
        "seed": SEED,
        "counts": counts,
    }
    (OUT / "divergent_sample.json").write_text(
        json.dumps({**meta, "items": items}, indent=1, ensure_ascii=False)
    )
    (OUT / "divergent_blind.json").write_text(
        json.dumps({**meta, "items": blind}, indent=1, ensure_ascii=False)
    )
    print(json.dumps(counts, indent=1))
    print(pd.Series([it["yes"] for it in items]).value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
