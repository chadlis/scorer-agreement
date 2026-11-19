import pytest

from autointerp.negatives import assign_candidates, build_pool


def w(s: str) -> list[str]:
    return list(s)


# toy windows, k=3: one letter per token
WINDOWS = {
    1: [w("abcdefgh"), w("cdefghij"), w("qrstuvwx")],  # first two: one passage
    2: [w("ABCDEFGH"), w("defghijk")],  # second one: same passage as feature 1's
    3: [w("12345678"), w("IJKLMNOP")],
}


def test_pool_counts_each_source_text_once_across_features():
    pool = build_pool(WINDOWS, k=3)
    assert [(p.feature, p.window) for p in pool] == [(1, 0), (1, 2), (2, 0), (3, 0), (3, 1)]


def big_pool():
    """5 features x 10 windows, all distinct: a pool of 50 passages."""
    windows = {f: [[f"t{f}.{i}.{j}" for j in range(8)] for i in range(10)] for f in range(5)}
    return windows, build_pool(windows, k=8)


def test_candidates_respect_exclusions_and_are_never_shared():
    windows, pool = big_pool()
    own = {0: windows[0], 1: windows[1]}
    out = assign_candidates(pool, own, excluded={0: {2}}, n=8, seed=1, k=8)
    assert all(len(v) == 8 for v in out.values())
    assert all(p.feature not in (0, 2) for p in out[0])
    assert all(p.feature != 1 for p in out[1])
    ids = [(p.feature, p.window) for v in out.values() for p in v]
    assert len(ids) == len(set(ids))


def test_candidates_never_overlap_the_target_windows():
    pool = build_pool(WINDOWS, k=3)
    # feature 9 is a target whose stored window shares "ABC" with feature 2's passage
    out = assign_candidates(pool, {9: [w("zzABCzz")]}, excluded={}, n=4, seed=0, k=3)
    assert (2, 0) not in [(p.feature, p.window) for p in out[9]]


def test_assignment_is_reproducible_and_depends_on_the_seed():
    windows, pool = big_pool()
    own = {0: windows[0], 1: windows[1]}
    a = assign_candidates(pool, own, {}, n=5, seed=7, k=8)
    assert a == assign_candidates(pool, own, {}, n=5, seed=7, k=8)
    assert a != assign_candidates(pool, own, {}, n=5, seed=8, k=8)


def test_too_small_pool_fails_loudly():
    windows, pool = big_pool()
    with pytest.raises(ValueError, match="pool too small"):
        assign_candidates(pool, {0: windows[0]}, {}, n=41, seed=0, k=8)
