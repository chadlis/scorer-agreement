"""Neuronpedia feature download, on-disk cache and token decoding.

API: GET https://www.neuronpedia.org/api/feature/{model}/{sae}/{index} (no key).
Raw responses are cached under data/raw/ and never modified.

Token strings in these responses use GPT-2's byte-level alphabet (newline is "Ċ",
the bytes of "’" are "âĢĻ"), except that spaces are already plain " ". A
multi-byte character can be split across two tokens, so decoding works on the
whole token sequence.
"""

import json
import re
import time
from pathlib import Path

import httpx

from autointerp.tls import ssl_context

API = "https://www.neuronpedia.org/api/feature"
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "raw"
SPECIAL = {"<|endoftext|>"}


def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode table."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


_BYTES = _byte_decoder()


def _to_bytes(token: str) -> bytes:
    return b"".join(bytes([_BYTES[ch]]) if ch in _BYTES else ch.encode() for ch in token)


def decode_tokens(tokens: list[str]) -> list[str]:
    """Decode token strings, one output string per token (same length as input).

    When a character spans several tokens, the whole character goes to the first
    of them and the following ones become "". Bytes that never form a valid
    character are replaced with U+FFFD.
    """
    out: list[str] = []
    pending = b""
    start = None
    for tok in tokens:
        if tok in SPECIAL:
            if pending:
                out[start] = pending.decode("utf-8", "replace")
                pending, start = b"", None
            out.append(tok)
            continue
        data = pending + _to_bytes(tok)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            if start is None:
                start = len(out)
                out.append("")
            else:
                out.append("")
            pending = data
            continue
        if start is None:
            out.append(text)
        else:
            out[start] = text
            out.append("")
            pending, start = b"", None
    if pending:
        out[start] = pending.decode("utf-8", "replace")
    return out


def window_text(tokens: list[str]) -> str:
    return "".join(decode_tokens(tokens))


_NEWLINES = re.compile(r"[ \t]*\n[\s]*")


def scorer_text(tokens: list[str]) -> str:
    """Text shown to a scorer: the decoded window with every run of line breaks
    (and the blanks around it) replaced by one space.

    Decided on 2026-11-19 after a format check in which line breaks inside a
    window made scorers read it as several examples (data/smoke/format_check.jsonl).
    Activations are unaffected: they were measured on the original text.
    """
    return _NEWLINES.sub(" ", window_text(tokens))


def kgrams(tokens: list[str], k: int = 6) -> set[tuple[str, ...]]:
    """Every run of k consecutive tokens."""
    return {tuple(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def source_groups(windows: list[list[str]], k: int = 6) -> list[int]:
    """Group sliding windows that come from the same source text.

    Two windows are linked when they share a run of k identical tokens. Links are
    transitive. Returns one group id per window, numbered from 0 by first window.
    """
    parent = list(range(len(windows)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: dict[tuple[str, ...], int] = {}  # k-gram -> first window that holds it
    for i, w in enumerate(windows):
        for g in kgrams(w, k):
            j = first.setdefault(g, i)
            if j != i:
                a, b = root(i), root(j)
                parent[max(a, b)] = min(a, b)
    ids: dict[int, int] = {}
    return [ids.setdefault(root(i), len(ids)) for i in range(len(windows))]


def fetch_feature(
    index: int, model: str = "gpt2-small", sae: str = "6-res-jb", pause: float = 1.0
) -> dict:
    """Return the raw JSON for one feature, from the cache or the API."""
    path = CACHE / model / sae / f"{index}.json"
    if path.exists():
        return json.loads(path.read_text())
    with httpx.Client(verify=ssl_context(), timeout=60) as client:
        r = client.get(f"{API}/{model}/{sae}/{index}")
        r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(r.text)
    time.sleep(pause)
    return r.json()
