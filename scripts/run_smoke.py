"""Step 1: one paid call per scorer, raw responses kept.

Usage: uv run scripts/run_smoke.py [--yes] [name ...]
Without --yes, nothing is sent: the script prints the planned calls and exits.

Records in data/smoke/smoke.jsonl without a "system" field were made with the
earlier draft prompt (commit 6c294f1). New records store the prompt sent.
"""

import json
import sys

from autointerp.clients import ROOT, Clients, load_env, score
from autointerp.models import SCORERS
from autointerp.prompt import SYSTEM, build_user

# Feature 100, explanation A, its top-activating window on Neuronpedia,
# with line breaks replaced by spaces as in neuronpedia.scorer_text
EXPL = "phrases related to causes and effects"
TEXT = " cost Research is showing the effect of human interaction with world's largest primates "


def main() -> None:
    names = [a for a in sys.argv[1:] if a != "--yes"] or list(SCORERS)
    if "--yes" not in sys.argv[1:]:
        print("Dry run, nothing sent. Planned paid calls:", ", ".join(names))
        print("Rerun with --yes to send them.")
        return
    load_env()
    clients = Clients()
    out = ROOT / "data/smoke/smoke.jsonl"
    for name in names:
        rec = score(clients, name, SYSTEM, build_user(EXPL, TEXT))
        with out.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        answer, tries = rec["final_answer"], rec["n_attempts"]
        print(f"{name:14} ok={rec['ok']} answer={answer} tries={tries} text={rec.get('text')!r}")


if __name__ == "__main__":
    main()
