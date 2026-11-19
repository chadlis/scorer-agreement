"""One synchronous call to one scorer, returned as a full raw record.

Used by the setup scripts. The asynchronous and batch clients for the grid will
build on the same record format.
"""

import os
import time
from pathlib import Path

import anthropic
import openai

from autointerp.models import MAX_TOKENS, SCORERS
from autointerp.prompt import parse_answer
from autointerp.tls import ssl_context

ROOT = Path(__file__).resolve().parents[2]


def load_env(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines into os.environ without overriding existing values."""
    for line in path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


class Clients:
    def __init__(self) -> None:
        self.anthropic = anthropic.Anthropic(
            base_url="https://api.anthropic.com",  # ignore ANTHROPIC_BASE_URL
            max_retries=2,
            http_client=anthropic.DefaultHttpxClient(verify=ssl_context()),
        )
        self.openrouter = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            http_client=openai.DefaultHttpxClient(verify=ssl_context()),
        )


def call_scorer(clients: Clients, name: str, system: str, user: str) -> dict:
    """Send one paid call and return a record with the prompt, settings and raw reply."""
    cfg = SCORERS[name]
    rec = {
        "scorer": name,
        "model": cfg["model"],
        "params": cfg["params"],
        "system": system,
        "user": user,
    }
    t0 = time.time()
    try:
        if cfg["api"] == "anthropic":
            r = clients.anthropic.messages.create(
                model=cfg["model"],
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
                **cfg["params"],
            )
            text = "".join(b.text for b in r.content if b.type == "text")
            rec.update(
                raw=r.to_dict(), served_model=r.model, stop=r.stop_reason, usage=r.usage.to_dict()
            )
        else:
            params = dict(cfg["params"])
            temperature = params.pop("temperature", None)
            r = clients.openrouter.chat.completions.create(
                model=cfg["model"],
                max_tokens=MAX_TOKENS,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                extra_body={"provider": cfg["provider"], "usage": {"include": True}, **params},
            )
            text = r.choices[0].message.content
            d = r.to_dict()
            rec.update(
                raw=d,
                served_model=d.get("model"),
                provider=d.get("provider"),
                provider_requested=cfg["provider"],
                stop=r.choices[0].finish_reason,
                usage=d.get("usage"),
            )
        rec.update(text=text, answer=parse_answer(text), ok=True)
    except Exception as e:  # keep every failure as a record
        rec.update(ok=False, error=f"{type(e).__name__}: {e}")
    rec["latency_s"] = round(time.time() - t0, 2)
    return rec


def score(clients: Clients, name: str, system: str, user: str) -> dict:
    """Protocol rule: an unreadable answer is retried once with the identical call.

    Returns the final record, with every attempt kept in "attempts" and the final
    reading in "final_answer" ("1", "0" or "invalid"). API errors are not retried
    here (the SDKs already retry rate limits and server errors).
    """
    attempts = [call_scorer(clients, name, system, user)]
    if attempts[0]["ok"] and attempts[0]["answer"] == "invalid":
        attempts.append(call_scorer(clients, name, system, user))
    final = attempts[-1]
    return {
        **final,
        "attempts": attempts,
        "n_attempts": len(attempts),
        "final_answer": final.get("answer", "invalid") if final["ok"] else "invalid",
    }
