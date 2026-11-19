"""Check which Gemini Flash versions accept disabled reasoning (one paid call each).

Produced data/smoke/gemini_versions.jsonl on 2026-11-19, with the earlier draft
prompt (commit 6c294f1). Rerunning uses the current prompt and stores it.
Usage: uv run scripts/check_gemini_versions.py
"""

import json
import os
import sys
import time
from pathlib import Path

import openai

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from autointerp.prompt import SYSTEM, build_user, parse_answer  # noqa: E402 (path set above)
from autointerp.tls import ssl_context  # noqa: E402

MODELS = [
    "google/gemini-2.5-flash",
    "google/gemini-3.5-flash",
    "google/gemini-3.6-flash",
    "google/gemini-3.7-flash",
]
PROVIDER = {"order": ["google-ai-studio"], "allow_fallbacks": False, "require_parameters": True}
# Same item as run_smoke.py (feature 100, explanation A)
EXPL = "phrases related to causes and effects"
TEXT = (
    " cost\n\nResearch is showing the effect of human interaction with world's largest primates\n"
)


def main() -> None:
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        http_client=openai.DefaultHttpxClient(verify=ssl_context()),
    )
    out = ROOT / "data/smoke/gemini_versions.jsonl"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_user(EXPL, TEXT)},
    ]
    for model in MODELS:
        rec = {
            "model": model,
            "params": {"temperature": 0, "reasoning": {"enabled": False}},
            "provider_order": PROVIDER["order"],
            "system": SYSTEM,
            "user": messages[1]["content"],
        }
        t0 = time.time()
        try:
            r = client.chat.completions.create(
                model=model,
                max_tokens=2048,
                temperature=0,
                messages=messages,
                extra_body={
                    "provider": PROVIDER,
                    "reasoning": {"enabled": False},
                    "usage": {"include": True},
                },
            )
            d = r.to_dict()
            usage = d.get("usage") or {}
            text = r.choices[0].message.content
            rec.update(
                ok=True,
                raw=d,
                text=text,
                answer=parse_answer(text),
                provider=d.get("provider"),
                reasoning_tokens=(usage.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens"
                ),
                cost=usage.get("cost"),
            )
        except Exception as e:
            rec.update(ok=False, error=f"{type(e).__name__}: {e}")
        rec["latency_s"] = round(time.time() - t0, 2)
        with out.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(
            f"{model:26} ok={rec['ok']} answer={rec.get('answer')} "
            f"reasoning_tokens={rec.get('reasoning_tokens')} {rec.get('error', '')[:160]}"
        )


if __name__ == "__main__":
    main()
