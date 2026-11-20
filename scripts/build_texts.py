"""Build and label the 40 texts of each sampled feature (free activation API calls).

Usage: uv run scripts/build_texts.py [feature ...]   (default: all 100, pilot first)

Negative candidates are assigned once for all targets (40 each, seed 20261120),
excluding each target's placebo donor, and saved in data/texts/candidates.json.
Each feature then gets data/texts/<feature>.json: its 20 positives and 20
negatives with their recomputed activations, and every passage set aside. A feature
already written is skipped, and activation calls are cached, so a run can resume.
"""

import json
import sys
from dataclasses import asdict

from autointerp.activation import activation
from autointerp.draw import SEED
from autointerp.negatives import assign_candidates, build_pool
from autointerp.neuronpedia import ROOT, fetch_feature
from autointerp.texts import build_negatives, build_positives

OUT = ROOT / "data" / "texts"
N_CANDIDATES = 40


def main(only: list[int]) -> None:
    sample = json.loads((ROOT / "data" / "draw" / "draw.json").read_text())["sample"]
    final = json.loads((ROOT / "data" / "placebos" / "final.json").read_text())["pairs"]
    donor = {p["target"]: p["donor"] for p in final}
    raws = {i: fetch_feature(i) for i in sample}
    windows = {i: [w["tokens"] for w in raws[i]["activations"]] for i in sample}
    cands = assign_candidates(
        build_pool(windows), windows, {t: {donor[t]} for t in sample}, N_CANDIDATES, SEED
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "candidates.json").write_text(
        json.dumps({str(t): [asdict(c) for c in cands[t]] for t in sample})
    )
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
