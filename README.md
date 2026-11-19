# Autointerp scorers: agreement and gain over placebo

Hackathon project (Mechanistic Interpretability Hackathon 5: Autointerp).
Work started before the kick-off; this is stated in the report.

Question: with texts and explanations held fixed, do detection scorers differ
in their judgments, and in how much the explanation helps them compared with an
unrelated explanation (placebo)?

Source: SAE `gpt2-small/6-res-jb` via the Neuronpedia API.

## Setup

```bash
uv sync
uv run pytest
```

API keys are read from environment variables only (never committed).

_Work in progress._
