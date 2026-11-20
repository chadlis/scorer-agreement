"""Gate report of the pilot (PLAN.md 2.7): the two central measures end to end, and
the gate criteria per scorer.

Usage: uv run scripts/pilot_report.py   (reads data/runs/pilot/calls.jsonl, no call)
Writes data/runs/pilot/report.json and prints it.
"""

import json

import pandas as pd

from autointerp import analysis as an
from autointerp.neuronpedia import ROOT
from autointerp.run import Log, cost

OUT = ROOT / "data" / "runs" / "pilot"


def main() -> None:
    log = Log(OUT / "calls.jsonl")
    finals = [r for r in log.records if r.get("final")]
    cols = ["feature", "text", "label", "condition", "scorer", "pass", "answer"]
    calls = pd.DataFrame([{k: r[k] for k in cols} for r in finals])
    scores = an.text_scores(calls)
    totals = an.accuracy_counts(scores).sum()
    stab = an.stability(calls)
    report = {"calls_finished": len(finals), "usd_spent": round(log.spent(), 4), "scorers": {}}
    for s in sorted(calls["scorer"].unique()):
        sc = scores[scores["scorer"] == s]
        cs = calls[calls["scorer"] == s]
        n_calls = len(cs)
        spent = sum(cost(r) for r in log.records if r.get("scorer") == s)
        acc = {c: round(an.accuracy(totals, s, c), 4) for c in an.CONDITIONS}
        invalid = float((cs["answer"] == "invalid").mean())
        report["scorers"][s] = {
            "accuracy": acc,
            "gain": round(an.gain(totals, s), 4),
            "sensitivity_AB": round(
                float(sc[(sc.label == 1) & (sc.condition != "P")].right.mean()), 4
            ),
            "specificity_AB": round(
                float(sc[(sc.label == 0) & (sc.condition != "P")].right.mean()), 4
            ),
            "yes_rate": {
                c: round(float(sc[sc.condition == c].yes.mean()), 4) for c in an.CONDITIONS
            },
            "invalid_rate": round(invalid, 4),
            "stability_changed": round(float(stab.get(s, float("nan"))), 4),
            "usd_per_call": round(spent / n_calls, 6) if n_calls else None,
            "gate": {
                "placebo_40_60": bool(0.40 <= acc["P"] <= 0.60),
                "invalid_below_2pct": bool(invalid < 0.02),
                "third_pass_needed": bool(float(stab.get(s, 0)) > 0.05),
            },
        }
    report["agreement_AB_pass1"] = {
        f"{x} / {y}": {k: round(float(v), 4) for k, v in an.agreement(t.sum()).items()}
        for (x, y), t in an.pair_counts(calls).items()
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
