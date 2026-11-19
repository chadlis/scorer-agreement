# scorer-agreement

Autointerp scorers: agreement and gain over placebo.

**Question.** With texts and explanations held fixed, do detection scorers differ
in their judgments, and in how much the explanation helps them compared with an
unrelated explanation (placebo)?

**Background.** A sparse autoencoder (SAE) splits a model layer into "features",
each meant to capture one idea. An LLM writes an *explanation* of when a feature
fires. A *scorer* LLM then reads the explanation and one text, and predicts
whether the feature fires on it (detection scoring). Its accuracy is the
explanation's score. This project asks how much that score depends on which
scorer is used.

Source: SAE `gpt2-small/6-res-jb` via the [Neuronpedia](https://www.neuronpedia.org) API.

## Origin

This project started as an entry to the Mechanistic Interpretability Hackathon 5:
Autointerp (kick-off 2026-11-20, submission 2026-11-22). Preparation work began
before the kick-off, as allowed by the rules and stated in the report. The commit
history shows what was fixed before any measurement.

The exact state submitted to the hackathon is the git tag `hackathon-submission`.
Work on `main` may continue after that date.

## Status

| Step | State |
|---|---|
| 1. Setup: one test call per scorer, scorer prompt frozen | test calls done; prompt is a **draft**, not frozen yet |
| 2. Freeze data: texts, verified labels, placebos, `PLAN.md` | not started |
| 3. Pilot on 5 features (go / no-go gate) | not started |
| 4. Full grid | not started |
| 5. Analysis: 2 figures, 1 table | not started |
| 6. Report and submission | not started |

## Planned design

To be frozen in `PLAN.md` before any measurement call.

- 100 features, 40 texts each (20 positive at varied activation levels, 20 negative).
- 3 conditions: explanation A (written by gpt-3.5-turbo), explanation B (written
  by gpt-4o-mini), placebo (the explanation of an unrelated feature).
- 5 scorers (below), one text per call, answer `1` or `0`, every call run twice.
- 100 × 40 × 3 × 5 × 2 = 120,000 calls. Claude models through the Message Batches API.
- Confidence intervals by bootstrap over features.

## Scorers

Settings tested on 2026-11-19 with one call each.

| Scorer | Model id | Settings | Thinking tokens on test |
|---|---|---|---|
| Haiku 5.5 | `claude-haiku-5-5` | thinking disabled, effort low | 0 |
| Sonnet 5.5 | `claude-sonnet-5-5` | thinking `between_tools`, effort low | 0 |
| Opus 5.5 | `claude-opus-5-5` | thinking cannot be disabled; effort low | 0 |
| Gemini Flash | `google/gemini-2.5-flash` (OpenRouter, Google AI Studio pinned) | temperature 0, reasoning disabled | 0 |
| Mistral Small 3.2 | `mistralai/mistral-small-3.2-24b-instruct` (OpenRouter, Parasail bf16 pinned) | temperature 0 | 0 |

- Claude models reject sampling parameters, so no temperature is sent to them.
- No server-side refusal fallback is enabled: it would silently swap the scorer.
  A refusal counts as an unreadable answer.
- OpenRouter providers are pinned with fallbacks disabled; the serving provider
  is logged with every response.
- Gemini version: van der Ben et al. (2026) name only "Gemini Flash". Gemini 3.5
  to 3.8 Flash reject disabling reasoning (HTTP 400; 3.8 at minimal effort used
  54-61 reasoning tokens on the test item). Gemini 2.5 Flash accepts it, so it is
  used to keep settings aligned with the other non-reasoning scorers.
- Mistral Small 3.2 is not served by Mistral itself on OpenRouter; Parasail is
  pinned because it serves the original bf16 weights.

The registry lives in `src/autointerp/models.py`.

## Scorer prompt

Draft in `src/autointerp/prompt.py`: a fixed system message and a user message
with two slots, the explanation and the text. The answer is read from its first
non-blank character (`1` or `0`); otherwise one identical retry, then `invalid`.
It will be compared with the detection prompt of EleutherAI's delphi, and
differences will be listed here, before it is frozen.

## Repository layout

```
src/autointerp/   prompt, scorer registry, TLS context
scripts/          run_smoke.py: one paid call per scorer
                  check_gemini_versions.py: which Gemini Flash accepts no reasoning
tests/            unit tests (no network, no paid calls)
data/raw/         Neuronpedia cache (not committed, re-downloadable)
data/smoke/       raw responses of the setup test calls
```

## Setup

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

API keys are read from environment variables only, never committed. Put them in
a `.env` file at the repository root (ignored by git):

```
ANTHROPIC_API_KEY=...
OPENROUTER_API_KEY=...
```

One paid test call per scorer (a fraction of a cent in total):

```bash
uv run scripts/run_smoke.py            # all scorers
uv run scripts/run_smoke.py mistral-small
```

## Network note (TLS)

The author's machine sits behind a TLS-inspecting proxy whose CA certificate lacks
a *critical* Basic Constraints extension, which Python's `VERIFY_X509_STRICT`
rejects. `src/autointerp/tls.py` clears that single flag for the project's own HTTP
clients; chain, signature, hostname and expiry are still verified. On a normal
network this has no effect.

## License

MIT.
