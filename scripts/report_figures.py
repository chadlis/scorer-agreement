"""Extra figures for the write-up (no call): the study design, the operating points
of the scorers, and the causes of disagreement on the 30 divergent texts.

Usage: uv run scripts/report_figures.py
Reads the frozen grid and the two blind readings, writes
figures/fig0_design.png, figures/fig3_operating_points.png,
figures/fig5_divergent.png and figures/fig_report_main.png (the two central measures
at the width of the two-page report).
"""

import json

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from autointerp.models import SCORERS
from autointerp.neuronpedia import ROOT
from autointerp.report import (
    BLUE,
    GRID_LINE,
    INK,
    INK2,
    NAMES,
    ORANGE,
    _intervals,
    _style,
    load_grid,
)

FIG = ROOT / "figures"
DATA = ROOT / "data" / "analysis"
# reference palette of the dataviz method, light mode: slots 1 to 5 in fixed order
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def design() -> plt.Figure:
    """The pipeline, left to right, with the count at each stage."""
    fig, ax = plt.subplots(figsize=(8.2, 1.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 25)
    ax.axis("off")
    boxes = [
        (1, "100 features", "SAE gpt2-small/6-res-jb,\nseeded draw"),
        (21, "40 texts each", "20 positives, 20 negatives,\nlabels measured"),
        (41, "3 descriptions", "explanation A, B,\nplacebo"),
        (61, "5 scorers, 2 passes", "one text per call,\nanswer 1 or 0"),
        (81, "120,000 answers", "agreement, gain,\nprofile"),
    ]
    for x, head, sub in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, 7),
                17,
                16,
                boxstyle="round,pad=0.4,rounding_size=1.5",
                facecolor="white",
                edgecolor=INK2,
                linewidth=1,
            )
        )
        ax.text(
            x + 8.5, 18.5, head, ha="center", va="center", fontsize=8.5, color=INK, weight="bold"
        )
        ax.text(x + 8.5, 12, sub, ha="center", va="center", fontsize=6.8, color=INK2)
    for x in (18.6, 38.6, 58.6, 78.6):
        ax.add_patch(
            FancyArrowPatch(
                (x, 15),
                (x + 2.2, 15),
                arrowstyle="-|>",
                mutation_scale=10,
                color=BLUE,
                linewidth=1.5,
            )
        )
    ax.text(
        50,
        2,
        "100 × 40 × 3 × 5 × 2 = 120,000 calls. Frozen before the first call: "
        "texts, labels, placebos and the plan.",
        ha="center",
        fontsize=7.5,
        color=INK2,
    )
    fig.tight_layout()
    return fig


def operating_points(calls) -> plt.Figure:
    """Sensitivity (positives answered 1) against false alarms (negatives answered 1),
    per scorer and condition, both passes. Invalid answers count in neither."""
    c = calls.assign(yes=calls.answer == "1")
    rate = c.groupby(["scorer", "condition", "label"]).yes.mean() * 100
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    marks = {"A": "^", "B": "v", "P": "o"}
    for color, s in zip(SLOTS, SCORERS, strict=True):
        xs = [rate[(s, k, 0)] for k in "ABP"]
        ys = [rate[(s, k, 1)] for k in "ABP"]
        for i in (0, 1):
            ax.plot([xs[2], xs[i]], [ys[2], ys[i]], color=color, linewidth=0.8, alpha=0.5)
        for k, x, y in zip("ABP", xs, ys, strict=True):
            ax.plot(x, y, marks[k], color=color, markersize=6, markeredgecolor="white")
        ax.text(xs[1] + 0.25, ys[1], NAMES[s].split(" ")[0], fontsize=7.5, color=INK, va="center")
    for k, label in (("A", "explanation A"), ("B", "explanation B"), ("P", "placebo")):
        ax.plot([], [], marks[k], color=INK2, label=label)
    ax.legend(fontsize=7.5, frameon=False, loc="center right")
    ax.set_xlabel("False alarms: negatives answered 1 (%)", fontsize=8.5, color=INK2)
    ax.set_ylabel("Sensitivity: positives answered 1 (%)", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 12)
    ax.set_ylim(-3, 80)
    _style(ax)
    ax.grid(axis="y", color=GRID_LINE, linewidth=0.8)
    fig.tight_layout()
    return fig


def divergent() -> plt.Figure:
    """Causes of disagreement on the 30 divergent texts, by reading and label."""
    sample = json.loads((DATA / "divergent_sample.json").read_text())["items"]
    readings = [
        json.loads((DATA / f).read_text())["items"]
        for f in ("divergent_agent_reading.json", "divergent_agent_reading_2.json")
    ]
    cats = [
        ("scorer_error", "scorer error"),
        ("literal_vs_sense", "literal vs sense"),
        ("too_broad", "explanation too broad"),
        ("too_narrow", "explanation too narrow"),
        ("weak_positive", "weak positive"),
        ("context", "context"),
        ("other", "other"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    rows = []
    for r, items in enumerate(readings, 1):
        for lab, name in ((1, "positives"), (0, "negatives")):
            rows.append(
                (
                    f"reading {r}, {name}",
                    [
                        sum(
                            1
                            for it, x in zip(sample, items, strict=True)
                            if it["label"] == lab and x["category"] == c
                        )
                        for c, _ in cats
                    ],
                )
            )
    colors = [BLUE, ORANGE, "#1baf7a", "#eda100", "#e87ba4", "#6250d6", INK2]
    for i, (_, counts) in enumerate(rows[::-1]):
        left = 0
        for n, color in zip(counts, colors, strict=True):
            if n:
                ax.barh(i, n, left=left, color=color, edgecolor="white", linewidth=2, height=0.62)
                ax.text(
                    left + n / 2, i, str(n), ha="center", va="center", fontsize=7.5, color="white"
                )
            left += n
    ax.set_yticks(range(len(rows)), [name for name, _ in rows[::-1]])
    handles = [Patch(color=c, label=label) for (_, label), c in zip(cats, colors, strict=True)]
    ax.legend(
        handles=handles,
        fontsize=7,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.42, -0.22),
    )
    ax.set_xlim(0, 15)
    ax.set_xlabel("texts (15 positives, 15 negatives)", fontsize=8, color=INK2)
    _style(ax)
    fig.tight_layout()
    return fig


def compact(res: dict) -> plt.Figure:
    """The two central measures at the width of a two-page report: (a) Cohen's kappa
    per pair of scorers, (b) the paired differences of gain, larger minus smaller."""
    short = {s: NAMES[s].split(" ")[0] for s in SCORERS}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    block = res["agreement_AB_pass1"]
    pairs = sorted(block, key=lambda k: block[k]["kappa"]["est"])
    v = [block[k]["kappa"] for k in pairs]
    _intervals(a, pairs, [x["est"] for x in v], [x["lo"] for x in v], [x["hi"] for x in v])
    a.set_yticks(range(len(pairs)), [" / ".join(short[p] for p in k.split(" / ")) for k in pairs])
    a.set_xlim(0.65, 0.9)
    a.set_title("(a) Cohen's kappa, A and B, pass 1", fontsize=8.5, color=INK, loc="left")
    diffs = {}
    for k, d in res["gain_differences"].items():
        x, y = k.split(" - ")
        if d["est"] < 0:
            x, y, d = y, x, {"est": -d["est"], "lo": -d["hi"], "hi": -d["lo"]}
        diffs[(x, y)] = d
    keys = sorted(diffs, key=lambda k: diffs[k]["est"])
    for i, k in enumerate(keys):
        d = diffs[k]
        color = BLUE if d["lo"] > 0 or d["hi"] < 0 else INK2
        _intervals(b, [None], [d["est"] * 100], [d["lo"] * 100], [d["hi"] * 100], color, i)
    b.axvline(0, color=INK, linewidth=0.8)
    b.set_yticks(range(len(keys)), [" − ".join(short[p] for p in k) for k in keys])
    b.set_title("(b) Paired difference of gain (points)", fontsize=8.5, color=INK, loc="left")
    for ax in (a, b):
        _style(ax)
        ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    return fig


def main() -> None:
    design().savefig(FIG / "fig0_design.png", dpi=200)
    operating_points(load_grid()).savefig(FIG / "fig3_operating_points.png", dpi=200)
    divergent().savefig(FIG / "fig5_divergent.png", dpi=200)
    res = json.loads((DATA / "results.json").read_text())
    compact(res).savefig(FIG / "fig_report_main.png", dpi=250)
    print("written fig0_design, fig3_operating_points, fig5_divergent, fig_report_main")


if __name__ == "__main__":
    main()
