"""Progress of a run (no paid call): finished calls per scorer, spend, unreadable
answers, and the state of the recent Claude batches.

Usage: uv run scripts/grid_status.py [grid|pilot]
"""

import sys
from collections import Counter

from autointerp.clients import Clients, load_env
from autointerp.models import SCORERS
from autointerp.neuronpedia import ROOT
from autointerp.run import Log

PER_SCORER = {"grid": 24_000, "pilot": 1_200}
WORKERS = 8  # OpenRouter calls at a time (run.run_openrouter)


def main(run: str) -> None:
    log = Log(ROOT / "data" / "runs" / run / "calls.jsonl")
    finals = [r for r in log.records if r.get("final")]
    done = Counter(r["scorer"] for r in finals)
    invalid = Counter(r["scorer"] for r in finals if r["answer"] == "invalid")
    n = PER_SCORER[run]
    total = n * len(SCORERS)
    print(
        f"{len(finals)} of {total} calls finished, {total - len(finals)} left, "
        f"{log.spent():.2f} USD spent"
    )
    for s in SCORERS:
        bar = "#" * round(20 * done[s] / n)
        print(f"  {s:14} {done[s]:6} / {n}  [{bar:20}]  {n - done[s]:6} left  invalid {invalid[s]}")
    # rough time left for OpenRouter: recent latency per call, 8 calls at a time
    recent = [r["latency_s"] for r in log.records[-500:] if r.get("latency_s")]
    left = sum(n - done[s] for s in SCORERS if SCORERS[s]["api"] != "anthropic")
    if recent and left:
        minutes = left * sum(recent) / len(recent) / WORKERS / 60
        print(f"  OpenRouter: about {minutes:.0f} min left (rough, from recent latencies)")
    load_env()
    for b in Clients().anthropic.messages.batches.list(limit=3):
        c = b.request_counts
        print(
            f"  batch {b.id[-6:]}: {b.processing_status}, {c.succeeded} succeeded, "
            f"{c.processing} processing, {c.errored} errored, {c.expired} expired"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "grid")
