"""Assign one placebo explanation to each sampled feature (PLAN.md, Placebos).

A seeded derangement of the sample gives each target a donor. Half of the targets
receive the donor's explanation A, half its explanation B. Pairs that fail one of
three filters are repaired by seeded swaps of donors. Pairs rejected at review are
passed back as `forbidden`, and rejected placebo texts as `forbidden_texts`, then
repaired the same way.
"""

import random
import re

SEED = 20261120
LEN_MIN, LEN_MAX = 2 / 3, 3 / 2
STOP = set(
    """a an the of or and to in on for with by as at from that which this these those is
are be being been it its their there related relating relation phrases phrase words word
terms term references reference referring mentions mention mentioning instances instance
specific specifically various particular particularly such text texts context contexts
including include includes presence concept concepts associated indicating indicate
indicates describing description descriptions something often like other others variations
variation occurrences occurrence use usage used involving involve different certain type
types form forms usually commonly common within especially""".split()
)


def stems(text: str) -> set[str]:
    """Content words of an explanation, cut to their first five letters."""
    return {w[:5] for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP and len(w) > 2}


def n_words(text: str) -> int:
    return len(text.split())


class Pairs:
    """Explanations and neighbours of the sample, and the filters on one pair."""

    def __init__(
        self,
        a: dict,
        b: dict,
        neighbours: dict,
        seed: int = SEED,
        targets: list[int] | None = None,
        specific: set[tuple[int, str]] | None = None,
    ):
        """`a`, `b`, `neighbours` cover targets and donors. `specific` holds the
        (feature, "A" or "B") explanations classed specific, or None for no filter."""
        self.a, self.b, self.nb, self.specific = a, b, neighbours, specific
        self.targets = sorted(targets if targets is not None else a)
        order = self.targets[:]
        random.Random(seed).shuffle(order)
        half = len(order) // 2
        self.model = {t: ("A" if k < half else "B") for k, t in enumerate(order)}

    def placebo(self, target: int, donor: int) -> str:
        return self.a[donor] if self.model[target] == "A" else self.b[donor]

    def problems(self, target: int, donor: int) -> list[str]:
        p = self.placebo(target, donor)
        out = []
        if stems(p) & (stems(self.a[target]) | stems(self.b[target])):
            out.append("words")
        if donor in self.nb[target] or target in self.nb[donor]:
            out.append("neighbour")
        mean = (n_words(self.a[target]) + n_words(self.b[target])) / 2
        if not LEN_MIN <= n_words(p) / mean <= LEN_MAX:
            out.append("length")
        if self.specific is not None and (donor, self.model[target]) not in self.specific:
            out.append("generic")
        return out


A_MODEL, B_MODEL = "gpt-3.5-turbo", "gpt-4o-mini"


def build_pairs(
    sample: list[int],
    fetch,
    seed: int = SEED,
    extra_donors: list[int] = (),
    specific: set[tuple[int, str]] | None = None,
) -> Pairs:
    """Pairs for the sampled features (and outside donors), from their Neuronpedia
    JSON (`fetch(index)`)."""
    raws = {i: fetch(i) for i in [*sample, *extra_donors]}

    def explanation(raw: dict, model: str) -> str:
        return next(
            e["description"].strip()
            for e in raw["explanations"]
            if e["explanationModelName"] == model
        )

    a = {i: explanation(r, A_MODEL) for i, r in raws.items()}
    b = {i: explanation(r, B_MODEL) for i, r in raws.items()}
    nb = {i: set(r["topkCosSimIndices"]) - {i} for i, r in raws.items()}
    return Pairs(a, b, nb, seed=seed, targets=sample, specific=specific)


def rows(pairs: Pairs, m: dict[int, int], order: list[int]) -> list[dict]:
    """One record per target, in the order of the sample."""
    return [
        {
            "target": t,
            "target_A": pairs.a[t],
            "target_B": pairs.b[t],
            "donor": m[t],
            "donor_model": pairs.model[t],
            "placebo": pairs.placebo(t, m[t]),
        }
        for t in order
    ]


def donor_pool(
    sample: list[int], specific: set[tuple[int, str]], candidates: list[int]
) -> tuple[list[int], list[int], list[int]]:
    """Rule 8b: (donors, sampled features left out, outside donors). A feature is a
    donor if at least one of its explanations is specific."""

    def has_specific(i: int) -> bool:
        return (i, "A") in specific or (i, "B") in specific

    dropped = [i for i in sample if not has_specific(i)]
    outside = [i for i in candidates if has_specific(i)]
    return [i for i in sample if has_specific(i)] + outside, dropped, outside


def derangement(
    targets: list[int], rng: random.Random, donors: list[int] | None = None
) -> dict[int, int]:
    """Map each target to a distinct donor, none to itself. Donors default to targets.
    With more donors than targets, the donors left over are not used."""
    pool = sorted(donors) if donors is not None else targets
    while True:
        donors = pool[:]
        rng.shuffle(donors)
        donors = donors[: len(targets)]
        if all(t != d for t, d in zip(targets, donors, strict=True)):
            return dict(zip(targets, donors, strict=True))


def repair(
    m: dict[int, int],
    pairs: Pairs,
    rng: random.Random,
    forbidden: frozenset = frozenset(),
    forbidden_texts: frozenset = frozenset(),
    max_swaps: int = 100_000,
    spare: list[int] | None = None,
    keep_valid: bool = False,
) -> int:
    """Swap donors in place until no pair fails the filters, is a forbidden (target,
    donor) pair, or uses a forbidden placebo text. A failing target swaps its donor
    with another target's, or with an unused donor from `spare` (updated in place).
    With `keep_valid`, a swap is made only if the other target's new pair passes, so
    valid pairs are only touched to fix a failing one. Returns the swap count."""

    def fails_with(t: int, d: int) -> bool:
        return (
            (t, d) in forbidden
            or pairs.placebo(t, d) in forbidden_texts
            or bool(pairs.problems(t, d))
        )

    def fails(t: int) -> bool:
        return fails_with(t, m[t])

    swaps = 0
    targets = sorted(m)
    while bad := [t for t in targets if fails(t)]:
        if swaps >= max_swaps:
            raise RuntimeError("no valid assignment found")
        t = rng.choice(bad)
        others = [x for x in targets if x != t]
        k = rng.randrange(len(others) + len(spare or []))
        if k < len(others):
            u = others[k]
            if m[u] != t and m[t] != u and not (keep_valid and fails_with(u, m[t])):
                m[t], m[u] = m[u], m[t]
        elif (d := spare[k - len(others)]) != t:
            spare[k - len(others)], m[t] = m[t], d
        swaps += 1
    return swaps


def review_sample(confident: list[int], k: int = 20, seed: int = SEED) -> list[int]:
    """The confident pairs the reviewer reads to check the triage (PLAN.md, section 2.4, rule 8)."""
    return sorted(random.Random(seed).sample(sorted(confident), k))


def assign(pairs: Pairs, seed: int = SEED) -> tuple[dict[int, int], int]:
    rng = random.Random(seed)
    m = derangement(pairs.targets, rng)
    return m, repair(m, pairs, rng)
