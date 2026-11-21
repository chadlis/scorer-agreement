"""The analyses of PLAN.md 2.6 on the frozen grid, and their two figures and table.

compute() runs the three planned analyses with paired bootstrap intervals over
features. Every interval uses the same seed, so all statistics are computed on the
same resampled features and differences between scorers stay paired.
fig_agreement() is Figure 1, fig_gain() is Figure 2, profile_table() is the Table.
Both scripts/analyze_grid.py and notebooks/analysis.ipynb call these functions.
"""

from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd

from autointerp import analysis as an
from autointerp.models import SCORERS
from autointerp.neuronpedia import ROOT

GRID = ROOT / "data" / "runs" / "grid" / "calls.csv"
NAMES = {
    "haiku-5.5": "Haiku 5.5",
    "sonnet-5.5": "Sonnet 5.5",
    "opus-5.5": "Opus 5.5",
    "gemini-flash": "Gemini 2.5 Flash",
    "mistral-small": "Mistral Small 3.2",
}
# reference palette of the dataviz method, light mode (static figures for print)
INK, INK2, GRID_LINE, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"


def load_grid() -> pd.DataFrame:
    return pd.read_csv(GRID, dtype={"answer": str, "batch_id": str})


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


def compute(calls: pd.DataFrame) -> dict:
    scorers = list(SCORERS)
    res: dict = {"calls": len(calls), "features": int(calls.feature.nunique())}
    # 1. agreement: A and B pooled, pass 1; pass 2 and placebo reported apart
    res["agreement_AB_pass1"] = agreement_block(calls, ("A", "B"), 1)
    res["agreement_AB_pass2"] = agreement_block(calls, ("A", "B"), 2)
    res["agreement_P_pass1"] = agreement_block(calls, ("P",), 1)
    # 2. gain over the placebo, with A and B apart, and the paired differences
    per_feature = an.accuracy_counts(an.text_scores(calls))
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
    # 3. profile per scorer and condition
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
    return res


def profile_table(res: dict) -> pd.DataFrame:
    """The Table: one row per scorer and condition, rates in %."""
    rows = []
    for s, p in res["profile"].items():
        for k in "ABP":
            v = p[k]
            rows.append(
                {
                    "scorer": NAMES[s],
                    "condition": {"A": "A", "B": "B", "P": "placebo"}[k],
                    "sensitivity": round(100 * v["sensitivity"], 1),
                    "specificity": round(100 * v["specificity"], 1),
                    "rate of 1": round(100 * v["yes_rate"], 1),
                    "invalid": round(100 * v["invalid_rate"], 2),
                    "changed between passes": round(100 * p["stability_changed"], 1),
                }
            )
    return pd.DataFrame(rows)


def _style(ax) -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.grid(axis="x", color=GRID_LINE, linewidth=0.8)
    ax.set_axisbelow(True)


def _intervals(ax, labels, est, lo, hi, color=BLUE, y_offset=0.0, marker="o", size=5):
    ys = [i + y_offset for i in range(len(labels))]
    ax.hlines(ys, lo, hi, color=color, linewidth=2)
    ax.plot(est, ys, marker, color=color, markersize=size, markeredgecolor="white")


def fig_agreement(res: dict):
    """Figure 1: raw agreement and Cohen's kappa per pair of scorers, A and B pooled,
    pass 1, with 95% bootstrap intervals over features. Pairs sorted by kappa."""
    block = res["agreement_AB_pass1"]
    pairs = sorted(block, key=lambda k: block[k]["kappa"]["est"])
    labels = [" / ".join(NAMES[p] for p in k.split(" / ")) for k in pairs]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9), sharey=True)
    for ax, key, title, scale in (
        (axes[0], "agreement", "Raw agreement (%)", 100),
        (axes[1], "kappa", "Cohen's kappa", 1),
    ):
        v = [block[k][key] for k in pairs]
        _intervals(
            ax,
            labels,
            [x["est"] * scale for x in v],
            [x["lo"] * scale for x in v],
            [x["hi"] * scale for x in v],
        )
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        _style(ax)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].set_xlim(84, 95)
    axes[1].set_xlim(0.65, 0.9)
    fig.tight_layout()
    return fig


def fig_gain(res: dict):
    """Figure 2: (a) gain over the placebo per scorer, mean of A and B (blue) and A
    and B apart (grey), (b) the 10 paired differences of gain, blue when the interval
    excludes 0. 95% bootstrap intervals over features, in percentage points."""
    scorers = list(SCORERS)[::-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 3.9), gridspec_kw={"width_ratios": [1, 1.15]})
    labels = [NAMES[s] for s in scorers]
    for key, off, color, marker, size in (
        ("A", 0.22, INK2, "^", 4),
        ("B", -0.22, INK2, "v", 4),
        ("AB", 0.0, BLUE, "o", 6),
    ):
        v = [res["gain"][s][key] for s in scorers]
        if key == "AB":
            _intervals(
                a,
                labels,
                [x["est"] * 100 for x in v],
                [x["lo"] * 100 for x in v],
                [x["hi"] * 100 for x in v],
                color,
                off,
                marker,
                size,
            )
        else:
            a.plot(
                [x["est"] * 100 for x in v],
                [i + off for i in range(len(v))],
                marker,
                color=color,
                markersize=size,
                linestyle="none",
            )
    a.set_yticks(range(len(labels)), labels)
    a.set_title("(a) Gain over the placebo (points)", fontsize=9.5, color=INK, loc="left")
    a.plot([], [], "o", color=BLUE, label="mean of A and B, 95% CI")
    a.plot([], [], "^", color=INK2, label="A alone")
    a.plot([], [], "v", color=INK2, label="B alone")
    a.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=3)
    a.set_xlim(20, 37)
    a.set_xticks([20, 25, 30, 35])
    _style(a)

    # each pair oriented so that its estimate is positive: larger gain minus smaller
    diffs = {}
    for k, d in res["gain_differences"].items():
        x, y = k.split(" - ")
        if d["est"] < 0:
            x, y, d = y, x, {"est": -d["est"], "lo": -d["hi"], "hi": -d["lo"]}
        diffs[(x, y)] = d
    keys = sorted(diffs, key=lambda k: diffs[k]["est"])
    dl = [" − ".join(NAMES[p].split(" ")[0] for p in k) for k in keys]
    for i, k in enumerate(keys):
        d = diffs[k]
        color = BLUE if d["lo"] > 0 or d["hi"] < 0 else INK2
        _intervals(b, [None], [d["est"] * 100], [d["lo"] * 100], [d["hi"] * 100], color, i)
    b.axvline(0, color=INK, linewidth=0.8)
    b.set_yticks(range(len(dl)), dl)
    b.set_title(
        "(b) Paired difference of gain (points)\nblue: the 95% interval excludes 0",
        fontsize=9.5,
        color=INK,
        loc="left",
    )
    _style(b)
    fig.tight_layout()
    return fig


CLAUDE = {s for s in SCORERS if SCORERS[s]["api"] == "anthropic"}


def exploratory_errors(calls: pd.DataFrame) -> dict:
    """Exploratory, added after the freeze (DEVIATIONS.md, 1): do two scorers fail on
    the same texts more than independent errors would? Per pair, the ratio of shared
    errors to e_x * e_y, and the mean ratio within Claude minus across families.
    Same texts as Figure 1 (A and B, pass 1, both answered), same bootstrap."""
    tables = an.error_counts(calls)
    wide = pd.concat({f"{x} / {y}": t for (x, y), t in tables.items()}, axis=1).fillna(0)
    pairs = list(dict.fromkeys(wide.columns.get_level_values(0)))
    within = [p for p in pairs if set(p.split(" / ")) <= CLAUDE]
    across = [p for p in pairs if p not in within]

    def ratio(t: pd.Series, p: str) -> float:
        return an.error_ratio(t[p])

    def family(t: pd.Series) -> float:
        w = sum(ratio(t, p) for p in within) / len(within)
        return w - sum(ratio(t, p) for p in across) / len(across)

    out = {"within_claude": within, "pairs": {}}
    for p in pairs:
        tot = wide[p].sum()
        out["pairs"][p] = {
            "n": int(tot["n"]),
            "e_x": round(tot["ex"] / tot["n"], 4),
            "e_y": round(tot["ey"] / tot["n"], 4),
            "e_xy": round(tot["exy"] / tot["n"], 4),
            "ratio": ci(wide, lambda t, p=p: ratio(t, p)),
        }
    out["within_minus_across"] = ci(wide, family)
    return out


def fig_errors(exp: dict):
    """Exploratory figure: shared-error ratio per pair, 95% bootstrap intervals.
    1 = errors as often together as if independent."""
    pairs = sorted(exp["pairs"], key=lambda p: exp["pairs"][p]["ratio"]["est"])
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for i, p in enumerate(pairs):
        r = exp["pairs"][p]["ratio"]
        color = BLUE if p in exp["within_claude"] else ORANGE
        _intervals(ax, [None], [r["est"]], [r["lo"]], [r["hi"]], color, i)
    ax.set_yticks(range(len(pairs)), [" / ".join(NAMES[s] for s in p.split(" / ")) for p in pairs])
    ax.axvline(1, color=INK, linewidth=0.8)
    ax.plot([], [], "o", color=BLUE, label="both Claude")
    ax.plot([], [], "o", color=ORANGE, label="across families")
    ax.legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(0.04, 0.5))
    ax.set_title(
        "Shared errors / errors expected if independent (exploratory)",
        fontsize=9.5,
        color=INK,
        loc="left",
    )
    _style(ax)
    fig.tight_layout()
    return fig
