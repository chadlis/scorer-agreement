from autointerp import texts as tx
from autointerp.negatives import Passage


def window(words, peak):
    return {"tokens": words, "values": [0] * (len(words) - 1) + [peak]}


def feature(n_passages, max_act=10.0):
    acts = [
        window([f"p{i}a", f"p{i}b", f"p{i}c", f"p{i}d", f"p{i}e", f"p{i}f"], 1.0 + i % 5)
        for i in range(n_passages)
    ]
    return {"maxActApprox": max_act, "activations": acts}


def test_even_ranks_cover_both_ends():
    assert tx.even_ranks(39, 20)[0] == 0 and tx.even_ranks(39, 20)[-1] == 38
    assert tx.even_ranks(20, 20) == list(range(20))
    assert len(set(tx.even_ranks(55, 20))) == 20


def test_one_window_per_passage_the_strongest():
    raw = feature(0)
    words = [f"s{i}" for i in range(8)]
    raw["activations"] = [window(words[:7], 2.0), window(words[1:], 5.0), window(["z"] * 6, 0)]
    got = tx.positive_passages(raw)
    assert [(g["window"], g["stored"]) for g in got] == [(1, 5.0)]


def test_positives_are_labelled_on_the_recomputed_value():
    raw = feature(30)
    values = iter([0.0, 0.5] + [3.0 + i for i in range(28)])  # one zero, one grey
    out = tx.build_positives(raw, lambda text: {"values": [next(values)]})
    assert out["positives"] == 28 and len(out["grey"]) == 1 and len(out["zero"]) == 1
    kept = [r["recomputed"] for r in out["kept"]]
    assert len(kept) == 20 and kept[0] == 3.0 and kept[-1] == 30.0 and kept == sorted(kept)


def test_too_few_positives_keep_nothing():
    out = tx.build_positives(feature(19), lambda text: {"values": [5.0]})
    assert out["positives"] == 19 and out["kept"] == []


def test_negatives_need_exactly_zero_and_stop_at_twenty():
    cands = [Passage(9, i, (f"w{i}",)) for i in range(30)]
    values = iter([0.013] + [0.0] * 29)
    calls = []

    def act(text):
        calls.append(text)
        return {"values": [next(values)]}

    out = tx.build_negatives(cands, act)
    assert len(out["kept"]) == 20 and len(out["rejected"]) == 1
    assert len(calls) == 21  # no call after the twentieth negative


def test_too_few_negatives_keep_nothing():
    cands = [Passage(9, i, (f"w{i}",)) for i in range(5)]
    out = tx.build_negatives(cands, lambda text: {"values": [0]})
    assert out["kept"] == []
