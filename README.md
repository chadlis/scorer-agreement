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

Setup is done. The scorer prompt was frozen on 2026-11-19 (git tag `prompt-frozen`)
after three format checks on 19 texts per scorer (`data/smoke/`).

The data are frozen (git tag `plan-frozen`, 2026-11-21): `PLAN.md`, the 40 verified
texts of each of the 100 features (`data/texts/`) and the placebos
(`data/placebos/final.json`). The pilot ran on 2026-11-21 on the 5 pilot features
(6,000 calls, 1.39 USD, raw log and gate report in `data/runs/pilot/`). It passed
the gate (`data/runs/pilot/decision.md`). The grid ran on the same day: 120,000
calls, 27.55 USD, frozen with the git tag `grid-frozen` (`data/runs/grid/`: one CSV
row per call, the raw log compressed as `calls.jsonl.gz`, counts and checksums in
`summary.json`). The planned analyses are in `notebooks/analysis.ipynb` (executed,
replayed from a fresh clone): Figure 1 (`figures/figure1_agreement.png`), Figure 2
(`figures/figure2_gain.png`), the table (`data/analysis/table_profile.csv`) and every
number with its interval (`data/analysis/results.json`). The 100 features are a simple
random sample of the eligible features, drawn with a seed committed beforehand. One
feature that kept too few positives after recomputation was replaced by the next
one of the draw (`PLAN.md`, appendix A.2). Negatives are drawn from the stored windows of the other
sampled features. Every label, positive or negative, is then checked with
Neuronpedia's activation API on the exact text shown to the scorer.

## Planned design

100 features, 40 texts per feature (20 where the feature fires, 20 where it does not),
3 conditions (the two original explanations and a placebo), 5 scorers, and every call
made twice: 120,000 calls. Answers are scored against labels recomputed on the exact
text shown, and the analyses (agreement, gain over the placebo, profile per scorer)
are fixed in `PLAN.md`, which is frozen before the first measurement.
Changes made after the freeze are listed in `DEVIATIONS.md`: so far, one exploratory
analysis (agreement on errors), added before any grid result was read.

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
one text per call), with four changes. Each one follows a check recorded in
`data/smoke/`.

1. The three solved examples that delphi places before the question are removed.
   Three of their 15 labels disagree with the activations recorded on Neuronpedia,
   and two of them come from the SAE studied here. These examples make up most of
   delphi's prompt, so results are not directly comparable with delphi's scores.
2. "return 1 if the sentence is correctly labeled or 0 if the tokens are mislabeled",
   identical to a sentence of delphi's fuzzing prompt, becomes "return 1 if the
   example possesses the latent or 0 if it does not": detection shows no labels.
3. "Your list must contain exactly one value, 0 or 1, per example." is added: without
   it, two scorers answered `[0, 1]` to a single text.
4. "Each example is a short excerpt and may start or end mid-sentence." is added:
   without it, one scorer refused to judge a text cut mid-sentence.

Line breaks inside a text are replaced by spaces, because they made scorers read one
text as several. Answers are free text, parsed with delphi's rule (the first `[...]`
must hold exactly one 0 or 1). Structured outputs are not used: they would add a
provider-specific decoding constraint that differs between scorers, and hide refusals.
An unreadable answer is retried once, then counted as invalid. An API failure that
persists after backoff is recorded as an error, apart from answers. Code:
`src/autointerp/prompt.py`.

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

Data-freezing scripts:

- `uv run scripts/check_activation_api.py` recomputes five stored windows of feature
  100 with Neuronpedia's free activation API (`data/activation_test/`).
- `uv run scripts/check_negatives.py` tries the negatives rule on six features.
- `uv run scripts/survey_features.py` downloads 120 random features (exploratory
  seed 1, distinct from the final draw) to size the sampling rule
  (`data/feature_survey/`).
- `uv run scripts/draw_features.py` draws the 100 features with the committed seed
  (`data/draw/`).
- `uv run scripts/assign_placebos.py` assigns one placebo to each feature (round 1),
  `uv run scripts/review_placebos.py N` lists the pairs to review in round N, and
  `uv run scripts/reassign_placebos.py N` reassigns the pairs rejected before round
  N (`data/placebos/roundN/`). From round 3, `uv run scripts/donor_candidates.py`
  fetches candidate outside donors and `uv run scripts/assign_donor_pool.py N`
  assigns from a pool that leaves out features whose two explanations are generic.
  `uv run scripts/auto_round.py assign N` and `close N` run the automatic rounds
  (`PLAN.md`, section 2.4, rules 10 and 11), and `final N` writes the frozen
  assignment, `data/placebos/final.json`.
- `uv run scripts/build_texts.py [feature ...]` builds and labels the 40 texts of
  each feature, pilot first, with the free activation API (`data/texts/`, about 8 h,
  resumable). `uv run scripts/replace_features.py` replaces a feature whose texts
  failed by the next eligible feature of the draw.
- `uv run scripts/run_pilot.py` prints what the pilot would send. With `--yes` it
  sends it (paid, capped at 10 USD, Claude through the Message Batches API), and only
  once the git tag `plan-frozen` exists (`data/runs/pilot/`).
- `uv run scripts/run_grid.py` prints what the grid would send. With `--yes` it sends
  it (paid). The total cap of 40 USD covers the pilot and the grid. The three Claude
  batches and the OpenRouter calls run at the same time (`data/runs/grid/`).
  `uv run scripts/grid_status.py` shows its progress (free).
- `uv run scripts/analyze_grid.py` runs the analyses of `PLAN.md` 2.6 on the frozen
  grid (no call, about 30 s) and writes the results, both figures and the table.
  `uv run jupyter nbconvert --to notebook --execute notebooks/analysis.ipynb` replays
  the notebook, which checks the grid checksums and the committed results.
- `uv run scripts/finalize_texts.py [--balance]` checks the built texts before the
  freeze (no negative from the placebo donor, separator balance, token lengths), and `uv run scripts/pilot_report.py`
  computes the pilot gate and both central measures from the pilot log.

Calls to the activation API are spaced by 3.6 s, retried on rate limits and server
errors, and cached under `data/raw/`.

Behind a TLS-inspecting proxy, set `AUTOINTERP_RELAX_X509_STRICT=1` to relax one
strict certificate check (`src/autointerp/tls.py`). Certificate verification stays on.
Without the variable, the default check applies.

## License

MIT.
