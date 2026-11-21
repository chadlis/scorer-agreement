"""Run the pilot: the 5 pilot features, 5 scorers, 3 conditions, 2 passes (PLAN.md 2.7).

Usage:
  uv run scripts/run_pilot.py          print what would be sent and the estimate
  uv run scripts/run_pilot.py --yes    send it (paid), capped at CAP_USD

A paid run refuses to start unless the tag plan-frozen exists and PLAN.md, the texts
and the placebos are unchanged since. Raw records go to data/runs/pilot/calls.jsonl
(append-only, the run resumes), finished calls to data/runs/pilot/calls.csv.
"""

import json
import sys

from autointerp.clients import Clients, call_scorer, load_env
from autointerp.models import SCORERS
from autointerp.neuronpedia import ROOT
from autointerp.run import (
    BATCH_DISCOUNT,
    PRICES,
    Log,
    export_csv,
    load_explanations,
    plan_calls,
    require_frozen,
    run_batch,
    run_openrouter,
)

CAP_USD = 10.0  # decided by Salah on 2026-11-20 (PLAN.md 2.7, rule 3)
OUT = ROOT / "data" / "runs" / "pilot"
# deliberately high per-call guess for the batch check: 300 input, 400 output tokens
EST_IN, EST_OUT = 300, 400


def estimate(c: dict) -> float:
    price_in, price_out = PRICES[SCORERS[c["scorer"]]["model"]]
    return (EST_IN * price_in + EST_OUT * price_out) / 1e6 * BATCH_DISCOUNT


def main(paid: bool) -> None:
    pilot = json.loads((ROOT / "data/draw/draw.json").read_text())["pilot"]
    texts = {}
    for f in pilot:
        rec = json.loads((ROOT / "data/texts" / f"{f}.json").read_text())
        if rec["status"] != "ok":
            raise SystemExit(f"feature {f}: texts not ready ({rec['status']})")
        texts[f] = rec
    calls = plan_calls(texts, load_explanations(pilot), list(SCORERS))
    claude = [c for c in calls if SCORERS[c["scorer"]]["api"] == "anthropic"]
    others = [c for c in calls if SCORERS[c["scorer"]]["api"] != "anthropic"]
    log = Log(OUT / "calls.jsonl")
    print(
        f"{len(calls)} calls ({len(claude)} by batch, {len(others)} by OpenRouter), "
        f"{len(log.done())} already done, {log.spent():.2f} USD spent, cap {CAP_USD} USD"
    )
    if not paid:
        print("dry run: add --yes to send")
        return
    require_frozen(ROOT)
    load_env()
    clients = Clients()
    run_batch(log, claude, clients.anthropic.messages.batches, CAP_USD, estimate)
    run_openrouter(log, others, lambda s, sys_, user: call_scorer(clients, s, sys_, user), CAP_USD)
    export_csv(log, OUT / "calls.csv")
    print(f"done: {len(log.done())} of {len(calls)} calls, {log.spent():.2f} USD spent")


if __name__ == "__main__":
    main("--yes" in sys.argv[1:])
