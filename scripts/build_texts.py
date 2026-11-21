"""Build and label the 40 texts of each sampled feature (free activation API calls).

Usage: uv run scripts/build_texts.py [feature ...]   (default: all 100, pilot first)

Negative candidates are assigned once for all targets (40 each, seed 20261120),
excluding each target's placebo donor, and saved in data/texts/candidates.json.
The pool is made of the windows of the sample as drawn, before any replacement.
A replacement feature (PLAN.md 2.3, rule 6) gets its 40 candidates afterwards, from
the same seeded pool, skipping every passage already in a list.
candidates.json is committed and reused as is. Rebuilding it from scratch would
need the donors of round 13, under which the first 100 lists were drawn.
Each feature then gets data/texts/<feature>.json: its 20 positives and 20
negatives with their recomputed activations, and every passage set aside. A feature
already written is skipped, and activation calls are cached, so a run can resume.
"""

import json
import sys
from dataclasses import asdict

from autointerp.activation import activation
from autointerp.draw import SEED
from autointerp.negatives import Passage, assign_candidates, build_pool
from autointerp.neuronpedia import ROOT, fetch_feature
from autointerp.texts import build_negatives, build_positives

OUT = ROOT / "data" / "texts"
N_CANDIDATES = 40


def main(only: list[int]) -> None:
    drawn = json.loads((ROOT / "data" / "draw" / "draw.json").read_text())
    sample = drawn["sample"]
    back = {r["in"]: r["out"] for r in drawn.get("replacements", [])}
    original = [back.get(f, f) for f in sample]
    final = json.loads((ROOT / "data" / "placebos" / "final.json").read_text())["pairs"]
    donor = {p["target"]: p["donor"] for p in final}
    raws = {i: fetch_feature(i) for i in set(sample) | set(original)}
    windows = {i: [w["tokens"] for w in raws[i]["activations"]] for i in raws}
    pool = build_pool({i: windows[i] for i in original})
    path = OUT / "candidates.json"
    if path.exists():
        cands = {
            int(t): [Passage(**{**c, "tokens": tuple(c["tokens"])}) for c in cs]
            for t, cs in json.loads(path.read_text()).items()
        }
    else:
        cands = assign_candidates(
            pool,
            {t: windows[t] for t in original},
            {t: {donor[t]} for t in original},
            N_CANDIDATES,
            SEED,
        )
    for t in sample:
        if t in cands:
            continue
        taken = frozenset((c.feature, c.window) for cs in cands.values() for c in cs)
        cands |= assign_candidates(
            pool, {t: windows[t]}, {t: {donor[t]}}, N_CANDIDATES, SEED, taken=taken
        )
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(t): [asdict(c) for c in cs] for t, cs in cands.items()}))
    for k, t in enumerate(f for f in sample if not only or f in only):
        path = OUT / f"{t}.json"
        if path.exists():
            continue
        pos = build_positives(raws[t], lambda text, t=t: activation(t, text))
        neg = build_negatives(cands[t], lambda text, t=t: activation(t, text))
        status = "ok" if pos["kept"] and neg["kept"] else "failed"
        record = {"feature": t, "status": status, "positives": pos, "negatives": neg}
        path.write_text(json.dumps(record, indent=1, ensure_ascii=False))
        print(
            f"{k + 1} feature {t}: {status}, {pos['positives']} positives "
            f"({len(pos['grey'])} grey, {len(pos['zero'])} at 0), "
            f"{len(neg['rejected'])} negatives rejected",
            flush=True,
        )


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]])
