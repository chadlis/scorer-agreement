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

Setup is in progress. Each scorer has been reached with one test call, and the scorer
prompt is a draft that has not been frozen. No measurement has been made yet.

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

Claude models reject sampling parameters, so none are sent to them: they run with
their default sampling, not at temperature 0. No fallback model
is enabled: a refusal counts as an unreadable answer. Gemini 2.5 Flash is used because
later Flash versions do not allow reasoning to be turned off
(`data/smoke/gemini_versions.jsonl`).

## Scorer prompt

The prompt is the detection prompt of EleutherAI's
[delphi](https://github.com/EleutherAI/delphi) (commit `a11130d`, default setting of
one text per call). Its instructions are kept word for word except for one replaced
sentence and one added sentence, and the three solved examples that delphi places
before the question are removed.
These examples make up most of delphi's prompt, so results are not directly
comparable with delphi's detection scores. They were removed because three of their
15 labels disagree with the activations recorded on Neuronpedia, and two of them
come from the SAE studied here. The sentence "return 1 if the sentence is correctly
labeled or 0 if the tokens are mislabeled", identical to a sentence of delphi's
fuzzing prompt, becomes "return 1 if the example
possesses the latent or 0 if it does not", because detection shows no labels. The
added sentence, "Your list must contain exactly one value, 0 or 1, per example", follows
a test call in which two scorers answered `[0, 1]` to a single example. Line breaks
inside a text are replaced by spaces before it is shown to a scorer, because a format
check showed that they make scorers read one text as several examples
(`data/smoke/format_check.jsonl`). An unreadable answer is retried once with the
identical call, then counted as invalid. Answers
are parsed with delphi's rule. See `src/autointerp/prompt.py`. The prompt is not
frozen yet.

## Usage

```bash
uv sync
uv run pytest
scripts/check.sh   # lint, format, tests and README typography, stops at the first failure
```

API keys are read from a `.env` file at the repository root, which git ignores:

```
ANTHROPIC_API_KEY=...
OPENROUTER_API_KEY=...
```

`uv run scripts/run_smoke.py --yes` makes one paid call per scorer (a fraction of a
cent). `uv run scripts/check_format.py --yes` sends 19 texts to each scorer to count
unreadable answers. Without `--yes`, scripts only print what they would send.

Behind a TLS-inspecting proxy, set `AUTOINTERP_RELAX_X509_STRICT=1` to relax one
strict certificate check (`src/autointerp/tls.py`). Certificate verification stays on.
Without the variable, the default check applies.

## License

MIT.
