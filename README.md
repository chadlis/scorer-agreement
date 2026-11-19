# scorer-agreement

Autointerp scorers: agreement and gain over placebo.

Question: with texts and explanations held fixed, do detection scorers differ
in their judgments, and in how much the explanation helps them compared with an
unrelated explanation (placebo)?

Source: SAE `gpt2-small/6-res-jb` via the Neuronpedia API.

## Origin

This project started as an entry to the Mechanistic Interpretability Hackathon 5:
Autointerp (kick-off 2026-11-20, submission 2026-11-22). Preparation work began
before the kick-off, as allowed by the rules and stated in the report. The commit
history shows what was fixed before any measurement.

The exact state submitted to the hackathon is the git tag `hackathon-submission`.
Work on `main` may continue after that date.

## Setup

```bash
uv sync
uv run pytest
```

API keys are read from environment variables only (never committed).

_Work in progress._

## Network note (TLS)

The author's machine sits behind a TLS-inspecting proxy whose CA certificate lacks a
*critical* Basic Constraints extension, which Python's `VERIFY_X509_STRICT` rejects.
`src/autointerp/tls.py` clears that single flag for the project's own HTTP clients;
chain, signature, hostname and expiry are still verified. On a normal network this
has no effect.

## Scorer settings (tested 2026-11-19, one call each)

| Scorer | Model id | Settings | Thinking tokens on test |
|---|---|---|---|
| Haiku 5.5 | `claude-haiku-5-5` | thinking disabled, effort low | 0 |
| Sonnet 5.5 | `claude-sonnet-5-5` | thinking `between_tools`, effort low | 0 |
| Opus 5.5 | `claude-opus-5-5` | thinking cannot be disabled; effort low | 0 |
| Gemini Flash | `google/gemini-3.8-flash` (OpenRouter, Google AI Studio pinned) | temperature 0, reasoning effort minimal (cannot be disabled) | 54 |
| Mistral Small 3.2 | `mistralai/mistral-small-3.2-24b-instruct` (OpenRouter, Parasail bf16 pinned) | temperature 0 | 0 |

Claude models reject sampling parameters, so no temperature is sent to them.
No server-side refusal fallback is enabled: it would silently swap the scorer.
