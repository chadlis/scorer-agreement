import math

import pandas as pd
import pytest

from autointerp import analysis as an


def calls(rows):
    cols = ["feature", "text", "label", "condition", "scorer", "pass", "answer"]
    return pd.DataFrame(rows, columns=cols)


def test_kappa_on_a_case_worked_by_hand():
    # rater a: 1 1 0 0, rater b: 1 0 0 0. Observed agreement 3/4.
    # yes rates 1/2 and 1/4, expected agreement 1/2 * 1/4 + 1/2 * 3/4 = 1/2.
    # kappa = (3/4 - 1/2) / (1 - 1/2) = 1/2
    assert an.kappa_from_counts(n11=1, n10=1, n01=0, n00=2) == pytest.approx(0.5)


def test_kappa_is_zero_when_agreement_is_what_chance_gives():
    # both raters say 1 half the time, independently: observed = expected = 1/2
    assert an.kappa_from_counts(n11=1, n10=1, n01=1, n00=1) == pytest.approx(0.0)


def test_accuracy_is_the_mean_over_passes_and_invalid_is_wrong():
    df = calls(
        [
            (1, 0, 1, "A", "s", 1, "1"),  # right
            (1, 0, 1, "A", "s", 2, "0"),  # wrong: the text scores 1/2
            (1, 1, 0, "A", "s", 1, "invalid"),  # wrong
            (1, 1, 0, "A", "s", 2, "0"),  # right: the text scores 1/2
        ]
    )
    scores = an.text_scores(df)
    assert list(scores["right"]) == [0.5, 0.5]
    totals = an.accuracy_counts(scores).sum()
    assert an.accuracy(totals, "s", "A") == pytest.approx(0.5)


def test_gain_on_a_case_worked_by_hand():
    rows = []
    # 4 texts (2 positives, 2 negatives), one pass. A: 4/4 right, B: 3/4, P: 2/4.
    answers = {"A": ["1", "1", "0", "0"], "B": ["1", "0", "0", "0"], "P": ["0", "0", "0", "0"]}
    for cond, ans in answers.items():
        for t, (label, a) in enumerate(zip([1, 1, 0, 0], ans, strict=True)):
            rows.append((7, t, label, cond, "s", 1, a))
    totals = an.accuracy_counts(an.text_scores(calls(rows))).sum()
    # gain = (1 + 3/4) / 2 - 1/2 = 3/8
    assert an.gain(totals, "s") == pytest.approx(0.375)


def test_pair_counts_skip_invalid_and_the_placebo():
    rows = [
        (1, 0, 1, "A", "x", 1, "1"),
        (1, 0, 1, "A", "y", 1, "1"),
        (1, 1, 0, "A", "x", 1, "0"),
        (1, 1, 0, "A", "y", 1, "invalid"),
        (1, 2, 0, "P", "x", 1, "1"),
        (1, 2, 0, "P", "y", 1, "0"),
        (1, 0, 1, "A", "x", 2, "0"),  # pass 2 is ignored
    ]
    t = an.pair_counts(calls(rows))[("x", "y")].sum()
    assert dict(t) == {"n11": 1, "n10": 0, "n01": 0, "n00": 0}
    assert an.agreement(t)["agreement"] == 1.0


def test_stability_counts_changed_answers():
    rows = [
        (1, 0, 1, "A", "s", 1, "1"),
        (1, 0, 1, "A", "s", 2, "1"),
        (1, 1, 0, "A", "s", 1, "0"),
        (1, 1, 0, "A", "s", 2, "1"),
    ]
    assert an.stability(calls(rows))["s"] == pytest.approx(0.5)


def test_bootstrap_resamples_features_not_texts():
    # feature 1: all right (10 texts), feature 2: all wrong. Point estimate 1/2.
    per_feature = pd.DataFrame({"sum": [10.0, 0.0], "count": [10.0, 10.0]}, index=[1, 2])
    est, lo, hi = an.cluster_bootstrap(per_feature, lambda s: s["sum"] / s["count"], n=2000)
    assert est == pytest.approx(0.5)
    # with two features the resampled means are 0, 1/2 or 1: the interval spans them
    assert lo == 0.0 and hi == 1.0


def test_bootstrap_is_reproducible_and_constant_data_gives_a_point():
    per_feature = pd.DataFrame({"sum": [5.0, 5.0, 5.0], "count": [10.0] * 3})
    first = an.cluster_bootstrap(per_feature, lambda s: s["sum"] / s["count"], n=500, seed=3)
    assert first == an.cluster_bootstrap(
        per_feature, lambda s: s["sum"] / s["count"], n=500, seed=3
    )
    assert first == (0.5, 0.5, 0.5)
    assert not math.isnan(first[0])
