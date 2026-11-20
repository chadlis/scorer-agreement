import random

import pytest

from autointerp import placebo as pl

TOPICS = ["cats", "rivers", "taxes", "guitars", "planets", "bread", "chess", "volcanoes"]


def toy_pairs(neighbours=None):
    a = {i: f"phrases related to {t}" for i, t in enumerate(TOPICS)}
    b = {i: f"mentions of {t} in text" for i, t in enumerate(TOPICS)}
    nb = neighbours or {i: set() for i in a}
    return pl.Pairs(a, b, nb, seed=3)


def test_stems_drop_generic_words_and_cut_to_five_letters():
    assert pl.stems('phrases related to "political" parties') == {"polit", "parti"}


def test_half_the_targets_get_a_and_half_b():
    p = toy_pairs()
    assert sorted(p.model.values()) == ["A"] * 4 + ["B"] * 4


def test_filters_catch_shared_words_neighbours_and_length():
    p = toy_pairs()
    p.a[1] = "phrases about cats near water"  # shares "cats" with target 0
    p.b[1] = p.a[1]
    assert "words" in p.problems(0, 1)
    p.nb[0] = {2}
    assert "neighbour" in p.problems(0, 2)
    p.a[3] = p.b[3] = "guitars"  # 1 word against a mean of 4.5
    assert p.problems(0, 3) == ["length"]
    assert p.problems(0, 4) == []


def test_assignment_is_a_derangement_without_failing_pairs():
    p = toy_pairs(neighbours={0: {1}, **{i: set() for i in range(1, 8)}})
    m, _ = pl.assign(p, seed=11)
    assert sorted(m.values()) == sorted(m)  # each explanation used once
    assert all(t != d for t, d in m.items())
    assert all(p.problems(t, d) == [] for t, d in m.items())
    assert pl.assign(p, seed=11)[0] == m  # reproducible


def test_repair_moves_a_forbidden_pair():
    p = toy_pairs()
    m, _ = pl.assign(p, seed=11)
    rejected = m[0]
    pl.repair(m, p, random.Random(1), forbidden=frozenset({(0, rejected)}))
    assert m[0] != rejected
    assert all(t != d for t, d in m.items()) and sorted(m.values()) == sorted(m)


def test_repair_gives_up_when_impossible():
    p = toy_pairs()
    for i in p.a:  # every placebo now shares a word with every target
        p.a[i] = p.b[i] = "shared words about stuff here"
    m = pl.derangement(p.targets, random.Random(0))
    with pytest.raises(RuntimeError):
        pl.repair(m, p, random.Random(0), max_swaps=50)


def test_review_sample_is_reproducible_and_ignores_input_order():
    confident = list(range(80))
    first = pl.review_sample(confident, k=20, seed=5)
    assert first == pl.review_sample(confident[::-1], k=20, seed=5)
    assert len(set(first)) == 20 and set(first) <= set(confident)
