"""Run the grid: 100 features, 5 scorers, 3 conditions, 2 passes (PLAN.md 2.5).

Usage:
  uv run scripts/run_grid.py          print what would be sent and the estimate
  uv run scripts/run_grid.py --yes    send it (paid), capped at CAP_USD minus the pilot

The total cap covers the whole project, so the grid may spend CAP_USD minus what
the pilot spent. The cost per call of each scorer is the one measured in the pilot
(data/runs/pilot/report.json), with a margin. The Claude scorers go out as three
Message Batches, one per scorer, while the OpenRouter scorers run at the same time.
Their estimated cost is reserved: OpenRouter stops before it would eat into it, so
the total stays under the cap as long as Claude costs no more than estimated. A
paid run refuses to start unless the frozen files are unchanged since plan-frozen.
Raw records go to data/runs/grid/calls.jsonl (append-only, the run resumes),
finished calls to data/runs/grid/calls.csv.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

from autointerp.clients import Clients, call_scorer, load_env
from autointerp.models import SCORERS
from autointerp.neuronpedia import ROOT
from autointerp.run import (
    Log,
    export_csv,
    load_explanations,
    plan_calls,
    require_frozen,
    run_batch,
    run_openrouter,
)

CAP_USD = 40.0  # total for the project, decided by Salah on 2026-11-21 after the pilot
MARGIN = 1.3  # on the cost per call measured in the pilot
PILOT = ROOT / "data" / "runs" / "pilot"
OUT = ROOT / "data" / "runs" / "grid"


def main(paid: bool) -> None:
    sample = json.loads((ROOT / "data/draw/draw.json").read_text())["sample"]
    texts = {}
    for f in sample:
        rec = json.loads((ROOT / "data/texts" / f"{f}.json").read_text())
        if rec["status"] != "ok":
            raise SystemExit(f"feature {f}: texts not ready ({rec['status']})")
        texts[f] = rec
    calls = plan_calls(texts, load_explanations(sample), list(SCORERS))
    per_call = {
        s: v["usd_per_call"] * MARGIN
        for s, v in json.loads((PILOT / "report.json").read_text())["scorers"].items()
    }
    pilot_spent = Log(PILOT / "calls.jsonl").spent()
    cap = CAP_USD - pilot_spent
    by_scorer = {s: [c for c in calls if c["scorer"] == s] for s in SCORERS}
    claude = [s for s in SCORERS if SCORERS[s]["api"] == "anthropic"]
    others = [c for s in SCORERS if s not in claude for c in by_scorer[s]]
    reserve = sum(len(by_scorer[s]) * per_call[s] for s in claude)
    est = reserve + len(others) * max(per_call[s] for s in SCORERS if s not in claude)
    log = Log(OUT / "calls.jsonl")
    print(
        f"{len(calls)} calls, {len(log.done())} already done, {log.spent():.2f} USD spent. "
        f"Cap for the grid {cap:.2f} USD (total {CAP_USD} minus pilot {pilot_spent:.2f}). "
        f"Estimate with margin {est:.2f} USD, of which Claude {reserve:.2f} reserved."
    )
    if est > cap:
        raise SystemExit("the estimate exceeds the cap: remove a scorer or features (PLAN 2.7)")
    if not paid:
        print("dry run: add --yes to send")
        return
    require_frozen(ROOT)
    load_env()
    clients = Clients()

    def batch(s: str) -> None:
        run_batch(
            log,
            by_scorer[s],
            clients.anthropic.messages.batches,
            cap,
            lambda c: per_call[c["scorer"]],
        )
        print(f"{s}: batch done", flush=True)

    def openrouter() -> None:
        def send(s: str, sys_: str, user: str) -> dict:
            return call_scorer(clients, s, sys_, user)

        run_openrouter(log, others, send, cap - reserve, own_spend=True)
        print("openrouter: done", flush=True)

    with ThreadPoolExecutor(len(claude) + 1) as pool:
        jobs = [pool.submit(batch, s) for s in claude] + [pool.submit(openrouter)]
        for j in jobs:
            j.result()
    export_csv(log, OUT / "calls.csv")
    print(f"done: {len(log.done())} of {len(calls)} calls, {log.spent():.2f} USD spent")


if __name__ == "__main__":
    main("--yes" in sys.argv[1:])
