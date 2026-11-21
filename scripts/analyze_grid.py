"""The analyses of PLAN.md 2.6 on the frozen grid (no call).

Usage: uv run scripts/analyze_grid.py
Reads data/runs/grid/calls.csv and writes data/analysis/results.json, the two
figures (figures/figure1_agreement.png, figures/figure2_gain.png) and the table
(data/analysis/table_profile.csv). The computation is in autointerp.report, shared
with notebooks/analysis.ipynb.
"""

import json

from autointerp import report
from autointerp.neuronpedia import ROOT

OUT = ROOT / "data" / "analysis"
FIG = ROOT / "figures"


def main() -> None:
    res = report.compute(report.load_grid())
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    report.profile_table(res).to_csv(OUT / "table_profile.csv", index=False)
    report.fig_agreement(res).savefig(FIG / "figure1_agreement.png", dpi=200)
    report.fig_gain(res).savefig(FIG / "figure2_gain.png", dpi=200)
    print(f"written {OUT / 'results.json'}, the table and both figures")


if __name__ == "__main__":
    main()
