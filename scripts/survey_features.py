"""Survey a random sample of features to prepare the sampling rule (PLAN.md).

Exploratory only: the seed here is not the seed of the final draw, and the features
seen here get no special treatment later. For each feature, records its activation
frequency, whether both original explanations exist (A = gpt-3.5-turbo,
B = gpt-4o-mini), and how many distinct positive passages Neuronpedia stores.
Free Neuronpedia feature API, one request per feature with a pause, cached.
Writes data/feature_survey/survey.json.
"""

import json
import random
import sys

from autointerp.neuronpedia import ROOT, fetch_feature, source_groups

N_FEATURES = 24576  # gpt2-small/6-res-jb
SURVEY_SEED = 1  # exploratory, distinct from the final draw's seed
A_MODEL, B_MODEL = "gpt-3.5-turbo", "gpt-4o-mini"


def summarise(index: int, raw: dict) -> dict:
    models = [e.get("explanationModelName") for e in raw.get("explanations", [])]
    pos = [w["tokens"] for w in raw.get("activations", []) if max(w["values"], default=0) > 0]
    return {
        "index": index,
        "frac_nonzero": raw.get("frac_nonzero"),
        "max_act": raw.get("maxActApprox"),
        "has_a": A_MODEL in models,
        "has_b": B_MODEL in models,
        "n_explanations": len(models),
        "n_windows": len(raw.get("activations", [])),
        "n_pos_windows": len(pos),
        "n_pos_passages": len(set(source_groups(pos))) if pos else 0,
        "explanations": {
            e.get("explanationModelName"): e.get("description") for e in raw["explanations"]
        },
    }


def main(n: int) -> None:
    indices = random.Random(SURVEY_SEED).sample(range(N_FEATURES), n)
    rows = []
    for k, i in enumerate(indices, 1):
        rows.append(summarise(i, fetch_feature(i)))
        print(f"{k}/{n} feature {i}", flush=True)
    out = ROOT / "data" / "feature_survey" / "survey.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seed": SURVEY_SEED, "features": rows}, indent=1))
    print("written", out)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
