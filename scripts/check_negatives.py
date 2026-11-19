"""Trial of the negatives rule (PLAN.md) on six real features (free, no LLM call).

Builds the pool from their stored windows, assigns candidates, and checks the first
four candidates of three targets with the activation API, on two versions of each
text: the decoded window, and the text shown to scorers (line breaks as spaces).
Produced data/activation_test/negatives_trial.json on 2026-11-19.
Usage: uv run scripts/check_negatives.py
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from autointerp.activation import activation  # noqa: E402 (path set above)
from autointerp.negatives import assign_candidates, build_pool  # noqa: E402
from autointerp.neuronpedia import fetch_feature, scorer_text, window_text  # noqa: E402

# three features already studied, three drawn at random (trial only, not the sample)
FEATURES = [100, 6048, 9396, *random.Random(2026).sample(range(24576), 3)]
TARGETS = [100, 6048, 9396]
OUT = ROOT / "data" / "activation_test" / "negatives_trial.json"


def main() -> None:
    data = {f: fetch_feature(f) for f in FEATURES}
    windows = {f: [a["tokens"] for a in d["activations"]] for f, d in data.items()}
    pool = build_pool(windows)
    per_feature = {f: sum(p.feature == f for p in pool) for f in FEATURES}
    own = {t: windows[t] for t in TARGETS}
    cands = assign_candidates(pool, own, excluded={}, n=10, seed=0)
    checks = []
    for t in TARGETS:
        for p in cands[t][:4]:
            raw, shown = window_text(list(p.tokens)), scorer_text(list(p.tokens))
            a_raw = activation(t, raw)
            a_shown = a_raw if shown == raw else activation(t, shown)
            checks.append(
                {
                    "target": t,
                    "from_feature": p.feature,
                    "window": p.window,
                    "text": shown,
                    "has_line_break": shown != raw,
                    "max_raw": a_raw["maxValue"],
                    "max_shown": a_shown["maxValue"],
                }
            )
    report = {
        "features": FEATURES,
        "frac_nonzero": {f: data[f]["frac_nonzero"] for f in FEATURES},
        "windows": sum(len(v) for v in windows.values()),
        "pool": len(pool),
        "pool_per_feature": per_feature,
        "checks": checks,
    }
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=1))
    for c in checks:
        print(
            c["target"],
            "<-",
            c["from_feature"],
            f"max {c['max_raw']:.2f} / {c['max_shown']:.2f}",
            "LB" if c["has_line_break"] else "  ",
            c["text"][:70],
        )


if __name__ == "__main__":
    main()
