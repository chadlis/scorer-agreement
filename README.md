# scorer-agreement

Do autointerp scorers agree with each other?

A sparse autoencoder (SAE) splits one layer of a language model into features, and a
language model writes an explanation of when each feature fires. To score an
explanation, a second model, the scorer, reads it together with one text and predicts
whether the feature fires on that text. This project asks whether different scorers
give the same judgments, and how much the explanation helps each of them compared
with an unrelated explanation (a placebo).

Data: SAE `gpt2-small/6-res-jb` from [Neuronpedia](https://www.neuronpedia.org).

## Origin

This project was started for the Mechanistic Interpretability Hackathon 5: Autointerp
(2026-11-20 to 2026-11-22). Preparation began before the kick-off, and the report says
so. The state submitted to the hackathon is the git tag `hackathon-submission`.

## Status

Setup is done: each scorer has answered one test call. The scorer prompt is still a
draft. No measurement has been made yet.

## Planned design

100 features, 40 texts per feature (20 where the feature fires, 20 where it does not),
3 conditions (the two original explanations and a placebo), 5 scorers, and every call
made twice: 120,000 calls. The plan will be frozen in `PLAN.md` before the first
measurement.

## Scorers

| Scorer | Model | Settings |
|---|---|---|
| Claude Haiku 5.5 | `claude-haiku-5-5` | thinking disabled, effort low |
| Claude Sonnet 5.5 | `claude-sonnet-5-5` | thinking `between_tools`, effort low |
| Claude Opus 5.5 | `claude-opus-5-5` | effort low (thinking cannot be disabled) |
| Gemini Flash | `google/gemini-2.5-flash` via OpenRouter, Google AI Studio | temperature 0, reasoning disabled |
| Mistral Small 3.2 | `mistralai/mistral-small-3.2-24b-instruct` via OpenRouter, Parasail bf16 | temperature 0 |

Claude models reject sampling parameters, so none are sent to them. No fallback model
is enabled: a refusal counts as an unreadable answer. Gemini 2.5 Flash is used because
later Flash versions do not allow reasoning to be turned off
(`data/smoke/gemini_versions.jsonl`).

## Scorer prompt

The prompt is the detection prompt of EleutherAI's
[delphi](https://github.com/EleutherAI/delphi) (commit `a11130d`, default setting of
one text per call), kept word for word except for two changes. The three solved
examples placed before the question are removed: three of their 15 labels contradict
the activations recorded on Neuronpedia, and two of them come from the SAE studied
here. The sentence "return 1 if the sentence is correctly labeled or 0 if the tokens
are mislabeled", taken from delphi's fuzzing prompt, becomes "return 1 if the example
possesses the latent or 0 if it does not", because detection shows no labels. Answers
are parsed with delphi's rule. See `src/autointerp/prompt.py`. The prompt is not
frozen yet.

## Usage

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

API keys are read from a `.env` file at the repository root, which git ignores:

```
ANTHROPIC_API_KEY=...
OPENROUTER_API_KEY=...
```

`uv run scripts/run_smoke.py` makes one paid call per scorer (a fraction of a cent).

`src/autointerp/tls.py` relaxes one strict certificate check that fails behind a
TLS-inspecting proxy. Certificate verification stays on.

## License

MIT.
