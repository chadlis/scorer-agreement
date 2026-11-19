"""Neuronpedia activation API: recompute one feature's activations on a text.

POST https://www.neuronpedia.org/api/activation/new (free, no key). The announced
limit is 1,000 requests per hour and is not reported in response headers, so calls
are spaced by at least MIN_INTERVAL_S. Raw responses are cached under data/raw/
and never modified: a cached text is never sent again.
"""

import hashlib
import json
import time
from pathlib import Path

import httpx

from autointerp.neuronpedia import CACHE
from autointerp.tls import ssl_context

URL = "https://www.neuronpedia.org/api/activation/new"
MIN_INTERVAL_S = 3.6  # 1,000 calls per hour at most
_last_call = 0.0


def cache_path(index: int, text: str, model: str, sae: str) -> Path:
    key = hashlib.sha256(text.encode()).hexdigest()[:24]
    return CACHE / "activations" / model / sae / str(index) / f"{key}.json"


def activation(
    index: int,
    text: str,
    model: str = "gpt2-small",
    sae: str = "6-res-jb",
    client: httpx.Client | None = None,
) -> dict:
    """Return Neuronpedia's raw response (tokens, values, maxValue, ...) for one text."""
    global _last_call
    path = cache_path(index, text, model, sae)
    if path.exists():
        return json.loads(path.read_text())["response"]
    wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    body = {"feature": {"modelId": model, "source": sae, "index": str(index)}, "customText": text}
    own = client is None
    client = client or httpx.Client(verify=ssl_context(), timeout=60)
    try:
        r = client.post(URL, json=body)
        r.raise_for_status()
    finally:
        _last_call = time.monotonic()
        if own:
            client.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"request": body, "response": r.json()}))
    return r.json()
