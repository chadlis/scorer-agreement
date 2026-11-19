"""Step 1: one paid call per scorer, raw responses kept.

Usage: uv run scripts/run_smoke.py [name ...]

Records in data/smoke/smoke.jsonl without a "system" field were made with the
earlier draft prompt (commit 6c294f1). New records store the prompt sent.
"""

import json
import os
import sys
import time
from pathlib import Path

import anthropic
import openai

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from autointerp.models import MAX_TOKENS, SCORERS
from autointerp.prompt import SYSTEM, build_user, parse_answer
from autointerp.tls import ssl_context

ROOT = Path(__file__).resolve().parents[1]
for line in (ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

# Toy item (feature 100, explanation A, a real top-activating window)
EXPL = "phrases related to causes and effects"
TEXT = (
    " cost\n\nResearch is showing the effect of human interaction with world's largest primates\n"
)


def main() -> None:
    ant = anthropic.Anthropic(
        base_url="https://api.anthropic.com",
        max_retries=2,
        http_client=anthropic.DefaultHttpxClient(verify=ssl_context()),
    )
    orr = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        http_client=openai.DefaultHttpxClient(verify=ssl_context()),
    )
    out = ROOT / "data/smoke/smoke.jsonl"

    for name in sys.argv[1:] or SCORERS:
        cfg = SCORERS[name]
        t0 = time.time()
        rec = {
            "scorer": name,
            "model": cfg["model"],
            "params": cfg["params"],
            "system": SYSTEM,
            "user": build_user(EXPL, TEXT),
        }
        try:
            if cfg["api"] == "anthropic":
                r = ant.messages.create(
                    model=cfg["model"],
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM,
                    messages=[{"role": "user", "content": build_user(EXPL, TEXT)}],
                    **cfg["params"],
                )
                raw = "".join(b.text for b in r.content if b.type == "text")
                rec.update(
                    raw=r.to_dict(),
                    served_model=r.model,
                    stop=r.stop_reason,
                    usage=r.usage.to_dict(),
                )
            else:
                p = dict(cfg["params"])
                temp = p.pop("temperature", None)
                r = orr.chat.completions.create(
                    model=cfg["model"],
                    max_tokens=MAX_TOKENS,
                    temperature=temp,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": build_user(EXPL, TEXT)},
                    ],
                    extra_body=dict(provider=cfg["provider"], usage={"include": True}, **p),
                )
                raw = r.choices[0].message.content
                d = r.to_dict()
                rec.update(
                    raw=d,
                    served_model=d.get("model"),
                    provider=d.get("provider"),
                    stop=r.choices[0].finish_reason,
                    usage=d.get("usage"),
                )
            rec.update(text=raw, answer=parse_answer(raw), ok=True)
        except Exception as e:
            rec.update(ok=False, error=f"{type(e).__name__}: {e}")
        rec["latency_s"] = round(time.time() - t0, 2)
        with out.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(
            f"{name:14} ok={rec['ok']} answer={rec.get('answer')} text={rec.get('text')!r} "
            f"served={rec.get('served_model')} prov={rec.get('provider')} stop={rec.get('stop')} "
            f"usage={rec.get('usage')} err={rec.get('error', '')[:300]}"
        )


if __name__ == "__main__":
    main()
