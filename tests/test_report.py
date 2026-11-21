import matplotlib

matplotlib.use("Agg")

from autointerp import report  # noqa: E402 (the backend is set before pyplot loads)
from autointerp.models import SCORERS  # noqa: E402

S = list(SCORERS)


def iv(x):
    return {"est": x, "lo": x - 0.01, "hi": x + 0.01}


def fake_results():
    pairs = [f"{x} / {y}" for i, x in enumerate(S) for y in S[i + 1 :]]
    block = {p: {"n": 100, "agreement": iv(0.9), "kappa": iv(0.8)} for p in pairs}
    diffs = {
        f"{x} - {y}": iv(0.02 if x == "opus-5.5" else -0.01)
        for x, y in (p.split(" / ") for p in pairs)
    }
    prof = {
        s: {
            "stability_changed": 0.01,
            **{
                k: {"sensitivity": 0.6, "specificity": 0.95, "yes_rate": 0.3, "invalid_rate": 0.01}
                for k in "ABP"
            },
        }
        for s in S
    }
    return {
        "agreement_AB_pass1": block,
        "gain": {s: {k: iv(0.3) for k in ("AB", "A", "B")} for s in S},
        "gain_differences": diffs,
        "profile": prof,
    }


def test_figures_have_one_row_per_pair_and_per_scorer():
    res = fake_results()
    f1 = report.fig_agreement(res)
    assert len(f1.axes[0].get_yticklabels()) == 10
    f2 = report.fig_gain(res)
    assert len(f2.axes[0].get_yticklabels()) == 5 and len(f2.axes[1].get_yticklabels()) == 10
    # every difference is drawn with a positive estimate (larger gain minus smaller)
    xs = [line.get_xdata()[0] for line in f2.axes[1].get_lines() if line.get_marker() == "o"]
    assert all(x >= 0 for x in xs)


def test_profile_table_has_one_row_per_scorer_and_condition():
    t = report.profile_table(fake_results())
    assert len(t) == 15 and set(t.condition) == {"A", "B", "placebo"}
    assert t["sensitivity"].iloc[0] == 60.0
