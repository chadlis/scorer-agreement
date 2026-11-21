"""The analyses of PLAN.md 2.6 on the frozen grid (no call).

Usage: uv run scripts/analyze_grid.py
Reads data/runs/grid/calls.csv, writes data/analysis/results.json and prints it.

1. Agreement between scorers (Figure 1): raw agreement, disagreement and Cohen's
   kappa per pair, A and B pooled, pass 1, with bootstrap intervals over features.
   The same on pass 2 and under the placebo, reported apart.
2. Gain over the placebo (Figure 2): per scorer, with A and B apart, then the 10
   paired differences of gain.
3. Profile per scorer (Table): sensitivity, specificity, rate of "1", invalid rate
   per condition, and stability between passes.
Every interval resamples the same features in the same order (same seed), so
differences between scorers are paired.
"""

import json
from itertools import combinations

import pandas as pd

from autointerp import analysis as an
from autointerp.models import SCORERS
from autointerp.neuronpedia import ROOT

OUT = ROOT / "data" / "analysis"


def ci(per_feature: pd.DataFrame, stat) -> dict:
    est, lo, hi = an.cluster_bootstrap(per_feature, stat)
    return {"est": round(est, 4), "lo": round(lo, 4), "hi": round(hi, 4)}


def agreement_block(calls: pd.DataFrame, conditions, pass_: int) -> dict:
    out = {}
    for (x, y), t in an.pair_counts(calls, conditions=conditions, pass_=pass_).items():
        out[f"{x} / {y}"] = {
            "n": int(t.to_numpy().sum()),
            "agreement": ci(t, lambda s: an.agreement(s)["agreement"]),
            "kappa": ci(t, lambda s: an.agreement(s)["kappa"]),
        }
    return out


def main() -> None:
    calls = pd.read_csv(ROOT / "data/runs/grid/calls.csv", dtype={"answer": str, "batch_id": str})
    scorers = list(SCORERS)
    res: dict = {"calls": len(calls), "features": int(calls.feature.nunique())}

    # 1. agreement
    res["agreement_AB_pass1"] = agreement_block(calls, ("A", "B"), 1)
    res["agreement_AB_pass2"] = agreement_block(calls, ("A", "B"), 2)
    res["agreement_P_pass1"] = agreement_block(calls, ("P",), 1)

    # 2. gain
    scores = an.text_scores(calls)
    per_feature = an.accuracy_counts(scores)
    res["accuracy"] = {
        s: {c: ci(per_feature, lambda t, s=s, c=c: an.accuracy(t, s, c)) for c in "ABP"}
        for s in scorers
    }
    res["gain"] = {
        s: {
            "AB": ci(per_feature, lambda t, s=s: an.gain(t, s)),
            "A": ci(per_feature, lambda t, s=s: an.accuracy(t, s, "A") - an.accuracy(t, s, "P")),
            "B": ci(per_feature, lambda t, s=s: an.accuracy(t, s, "B") - an.accuracy(t, s, "P")),
        }
        for s in scorers
    }
    res["gain_differences"] = {
        f"{x} - {y}": ci(per_feature, lambda t, x=x, y=y: an.gain(t, x) - an.gain(t, y))
        for x, y in combinations(scorers, 2)
    }

    # 3. profile
    c = calls.assign(yes=calls.answer == "1", no=calls.answer == "0", inv=calls.answer == "invalid")
    stab = an.stability(calls)
    prof = {}
    for s in scorers:
        prof[s] = {"stability_changed": round(float(stab[s]), 4)}
        for k in "ABP":
            x = c[(c.scorer == s) & (c.condition == k)]
            prof[s][k] = {
                "sensitivity": round(float(x[x.label == 1].yes.mean()), 4),
                "specificity": round(float(x[x.label == 0].no.mean()), 4),
                "yes_rate": round(float(x.yes.mean()), 4),
                "invalid_rate": round(float(x.inv.mean()), 4),
            }
    res["profile"] = prof

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
