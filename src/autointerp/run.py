"""Run scorer calls for the pilot or the grid (PLAN.md, sections 2.5 and 2.7).

Every call is identified by (feature, text, condition, scorer, pass). Raw records are
appended to a JSONL file and never rewritten, so a run resumes where it stopped. The
spend is summed from the records: no new call or batch starts if it would cross the
cap. Claude scorers go through the Message Batches API, the others through
OpenRouter, a few calls at a time. A paid run refuses to start unless PLAN.md, the
texts and the placebos are frozen (git tag `plan-frozen`, nothing changed since).
"""

import csv
import json
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from autointerp.models import MAX_TOKENS, SCORERS
from autointerp.prompt import SYSTEM, build_user, parse_answer

# USD per million tokens (input, output), Anthropic's published prices. Batch is half.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
}
BATCH_DISCOUNT = 0.5
FROZEN = ("PLAN.md", "data/texts", "data/placebos/final.json")


def call_id(c: dict) -> str:
    """Batch custom ids allow letters, digits, '-' and '_' only."""
    scorer = c["scorer"].replace(".", "_")
    return f"f{c['feature']}-t{c['text']}-{c['condition']}-{scorer}-p{c['pass']}"


def plan_calls(
    texts: dict[int, dict], explanations: dict[int, dict], scorers: list[str], passes=(1, 2)
) -> list[dict]:
    """All calls for the given features. Texts 0-19 are the positives, 20-39 the
    negatives, in the order stored in data/texts. explanations[f] maps "A", "B", "P"
    to the text shown."""
    out = []
    for f, rec in texts.items():
        items = [(t["text"], 1) for t in rec["positives"]["kept"]]
        items += [(t["text"], 0) for t in rec["negatives"]["kept"]]
        for i, (text, label) in enumerate(items):
            for cond in ("A", "B", "P"):
                for s in scorers:
                    for p in passes:
                        c = {
                            "feature": f,
                            "text": i,
                            "label": label,
                            "condition": cond,
                            "scorer": s,
                            "pass": p,
                            "explanation": explanations[f][cond],
                            "user": build_user(explanations[f][cond], text),
                        }
                        c["id"] = call_id(c)
                        out.append(c)
    return out


def cost(record: dict) -> float:
    """USD spent on one attempt, from its usage."""
    usage = record.get("usage") or {}
    if "cost" in usage:  # OpenRouter reports the cost itself
        return float(usage["cost"] or 0)
    price_in, price_out = PRICES.get(record.get("model"), (0.0, 0.0))
    usd = usage.get("input_tokens", 0) * price_in + usage.get("output_tokens", 0) * price_out
    usd /= 1e6
    return usd * (BATCH_DISCOUNT if record.get("transport") == "batch" else 1)


class Log:
    """Append-only JSONL of attempts, with the spend so far and the finished calls."""

    def __init__(self, path: Path):
        self.path, self.lock = path, threading.Lock()
        self.records = (
            [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        )

    def add(self, record: dict) -> None:
        with self.lock:
            self.records.append(record)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")

    def spent(self) -> float:
        return sum(cost(r) for r in self.records)

    def done(self) -> set[str]:
        """Calls with a final reading ("1", "0" or "invalid" after its retry)."""
        return {r["id"] for r in self.records if r.get("final")}


def require_frozen(root: Path) -> None:
    """Refuse a paid run unless the tag plan-frozen exists and the frozen files are
    unchanged since, in the work tree and in the index."""

    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)

    if git("rev-parse", "--verify", "plan-frozen").returncode != 0:
        raise SystemExit("not frozen: the git tag plan-frozen does not exist")
    if git("diff", "--quiet", "plan-frozen", "--", *FROZEN).returncode != 0:
        raise SystemExit("frozen files changed since the tag plan-frozen")


CALL_KEYS = ("id", "feature", "text", "label", "condition", "scorer", "pass")


def finish(log: Log, call: dict, record: dict, attempt: int) -> bool:
    """Store one attempt. Returns True if the call is finished: a readable answer, or
    an unreadable one already retried once (PLAN.md, Scoring, rule 3).
    The model's reply is stored as "reply". "text" stays the index of the text."""
    answer = parse_answer(record.get("text")) if record.get("ok") else "error"
    final = answer in ("1", "0") or (answer == "invalid" and attempt == 2)
    rec = {("reply" if k == "text" else k): v for k, v in record.items()}
    keep = {k: call[k] for k in CALL_KEYS}
    log.add({**rec, **keep, "answer": answer, "attempt": attempt, "final": final})
    return final


def call_fields(record: dict) -> dict:
    """The call's fields and the reply of one stored attempt. In the pilot log
    (written before 2026-11-21 06:00), the reply overwrote "text", so the text index
    is always read back from the id (f{feature}-t{text}-...)."""
    reply = record["reply"] if "reply" in record else record.get("text")
    text = int(record["id"].split("-")[1][1:])
    return {**{k: record[k] for k in CALL_KEYS}, "text": text, "reply": reply}


def run_openrouter(
    log: Log,
    calls: list[dict],
    send: Callable[[str, str, str], dict],
    cap: float,
    workers: int = 8,
) -> None:
    """Calls one by one per worker, each retried once if unreadable, until the cap."""

    def one(c: dict) -> None:
        for attempt in (1, 2):
            if log.spent() >= cap:
                return
            rec = send(c["scorer"], SYSTEM, c["user"])
            rec["transport"] = "openrouter"
            if finish(log, c, rec, attempt) or rec.get("answer") != "invalid":
                return

    done = log.done()
    todo = [c for c in calls if c["id"] not in done]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, todo))


def batch_params(c: dict) -> dict:
    cfg = SCORERS[c["scorer"]]
    return {
        "model": cfg["model"],
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": c["user"]}],
        **cfg["params"],
    }


def run_batch(
    log: Log,
    calls: list[dict],
    batches,
    cap: float,
    est_per_call: Callable[[dict], float],
    poll_s: float = 30,
) -> None:
    """Send the calls not yet finished as one Message Batch, wait, store the results.
    Unreadable answers go out again in a second batch (their one retry)."""
    for attempt in (1, 2):
        done = log.done()
        # an attempt that got no answer (API error, expired request) is tried again
        answered = {r["id"] for r in log.records if r.get("attempt") == attempt and r.get("ok")}
        todo = [c for c in calls if c["id"] not in done and c["id"] not in answered]
        if attempt == 2:
            invalid = {
                r["id"]
                for r in log.records
                if r.get("attempt") == 1 and r.get("answer") == "invalid"
            }
            todo = [c for c in todo if c["id"] in invalid]
        if not todo:
            continue
        estimate = sum(est_per_call(c) for c in todo)
        if log.spent() + estimate > cap:
            raise SystemExit(f"cap: {log.spent():.2f} spent + {estimate:.2f} estimated > {cap}")
        by_id = {c["id"]: c for c in todo}
        batch = batches.create(
            requests=[{"custom_id": i, "params": batch_params(c)} for i, c in by_id.items()]
        )
        while batches.retrieve(batch.id).processing_status != "ended":
            time.sleep(poll_s)
        for entry in batches.results(batch.id):
            c = by_id[entry.custom_id]
            cfg = SCORERS[c["scorer"]]
            rec = {
                "model": cfg["model"],
                "params": cfg["params"],
                "system": SYSTEM,
                "user": c["user"],
                "transport": "batch",
                "batch_id": batch.id,
                "result_type": entry.result.type,
            }
            if entry.result.type == "succeeded":
                m = entry.result.message
                rec.update(
                    ok=True,
                    raw=m.to_dict(),
                    served_model=m.model,
                    stop=m.stop_reason,
                    usage=m.usage.to_dict(),
                    text="".join(b.text for b in m.content if b.type == "text"),
                )
            else:
                rec.update(ok=False, raw=entry.result.to_dict())
            finish(log, c, rec, attempt)


def export_csv(log: Log, path: Path) -> None:
    """One row per finished call (PLAN.md, Scoring, rule 8)."""
    cols = [
        "feature",
        "text",
        "label",
        "condition",
        "scorer",
        "model",
        "params",
        "pass",
        "attempt",
        "transport",
        "answer",
        "raw_text",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "latency_s",
        "batch_id",
        "id",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in log.records:
            if not r.get("final"):
                continue
            u = r.get("usage") or {}
            w.writerow(
                {
                    **{k: r.get(k) for k in cols},
                    **{k: v for k, v in call_fields(r).items() if k != "reply"},
                    "params": json.dumps(r.get("params")),
                    "raw_text": call_fields(r)["reply"],
                    "input_tokens": u.get("input_tokens", u.get("prompt_tokens")),
                    "output_tokens": u.get("output_tokens", u.get("completion_tokens")),
                    "cost_usd": round(cost(r), 8),
                }
            )
