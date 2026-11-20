from autointerp import draw as d


def window(words, peak):
    return {"tokens": words, "values": [0] * (len(words) - 1) + [peak]}


def feature(frac=1e-3, models=("gpt-3.5-turbo", "gpt-4o-mini"), passages=20, peak=10.0):
    acts = [
        window([f"p{i}a", f"p{i}b", f"p{i}c", f"p{i}d", f"p{i}e", f"p{i}f"], peak)
        for i in range(passages)
    ]
    return {
        "frac_nonzero": frac,
        "maxActApprox": peak,
        "explanations": [{"explanationModelName": m} for m in models],
        "activations": acts,
    }


def test_order_is_a_reproducible_permutation():
    a, b = d.shuffled_order(7, 50), d.shuffled_order(7, 50)
    assert a == b and sorted(a) == list(range(50))
    assert d.shuffled_order(8, 50) != a


def test_stored_conditions_name_the_first_failure():
    assert d.check_stored(feature()) is None
    assert d.check_stored(feature(frac=1e-2)) == "frequency"
    assert d.check_stored(feature(frac=1e-5)) == "frequency"
    assert d.check_stored(feature(models=("gpt-4o-mini",))) == "explanations"
    assert d.check_stored(feature(passages=19)) == "positives"


def test_sliding_windows_count_as_one_passage():
    words = [f"s{i}" for i in range(8)]
    slid = [window(words[:7], 5.0), window(words[1:], 5.0)]  # share the 6 tokens s1..s6
    raw = feature(passages=18)
    raw["activations"] += slid
    assert d.check_stored(raw) == "positives"  # 18 + 1 passage
    raw = feature(passages=19)
    raw["activations"] += slid
    assert d.check_stored(raw) is None  # 19 + 1 passage


def test_gate_compares_recomputed_maximum_with_stored_maximum():
    raw = feature(peak=10.0)
    assert d.gate(raw, lambda text: {"values": [0, 1.0]}) == (True, 0.1)
    passed, share = d.gate(raw, lambda text: {"values": [0, 0.99]})
    assert not passed and share == 0.099


def test_draw_keeps_eligible_features_in_order_and_skips_none():
    feats = {
        0: feature(),
        1: feature(frac=0.05),
        2: feature(),
        3: feature(passages=3),
        4: feature(),
        5: feature(),
    }
    fires = {0: 9.0, 2: 0.0, 4: 5.0, 5: 3.0}  # feature 2 fails the gate
    out = d.draw(
        [3, 0, 1, 2, 4, 5],
        fetch=lambda i: feats[i],
        activation=lambda i, text: {"values": [fires[i]]},
        n=2,
    )
    assert out["sample"] == [0, 4]
    assert [r["failed"] for r in out["examined"]] == ["positives", None, "frequency", "gate", None]
    assert out["next_position"] == 5  # feature 5 is the first replacement


def test_draw_resumes_from_a_position():
    feats = {i: feature() for i in range(4)}
    out = d.draw([0, 1, 2, 3], lambda i: feats[i], lambda i, t: {"values": [9.0]}, n=1, start=2)
    assert out["sample"] == [2] and out["next_position"] == 3
