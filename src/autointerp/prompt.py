"""Scorer prompt: delphi's detection prompt with four documented changes.

Base: EleutherAI/delphi, delphi/scorers/classifier/prompts/detection_prompt.py at
commit a11130d6cf9493ac7feb9205b4fa9edf67cadb66, default configuration
(n_examples_shown=1, one text per call). Everything is kept verbatim except:

1. The three solved few-shot examples are removed (zero-shot). They make up most
   of delphi's prompt, so scores are not directly comparable with delphi's. Three
   of their 15 labels disagree with the activations recorded on Neuronpedia, and
   two of the three examples come from the SAE studied here (gpt2-small/6-res-jb,
   features 6048 and 9396).
2. "return 1 if the sentence is correctly labeled or 0 if the tokens are
   mislabeled" is replaced by "return 1 if the example possesses the latent or 0
   if it does not". The original sentence is identical to one in delphi's fuzzing
   prompt, where marked tokens are labels. Detection shows no labels.
3. One sentence is added after "You must return your response in a valid Python
   list.": "Your list must contain exactly one value, 0 or 1, per example." In a
   smoke test without it (2026-11-19, data/smoke/smoke.jsonl), Haiku 5.5 and
   Mistral Small 3.2 answered [0, 1] to a single example. The solved examples
   removed in change 1 were delphi's only demonstration of the list format.
4. One sentence is added after "You will then be given several text examples.":
   "Each example is a short excerpt and may start or end mid-sentence." In a
   format check without it (data/smoke/format_check_v2.jsonl), Sonnet 5.5 twice
   refused to answer on a window cut mid-sentence and asked for the full text.

The answer is parsed with delphi's rule (Classifier._parse).
FROZEN on 2026-11-19 (git tag prompt-frozen). Do not edit SYSTEM, USER_TEMPLATE or
parse_answer: any later change is a deviation, recorded in DEVIATIONS.md.
"""

import json
import re

SYSTEM = (
    "You are an intelligent and meticulous linguistics researcher.\n\n"
    'You will be given a certain latent of text, such as "male pronouns" or '
    '"text with negative sentiment".\n\n'
    "You will then be given several text examples. Each example is a short excerpt "
    "and may start or end mid-sentence. Your task is to determine which examples "
    "possess the latent.\n\n"
    "For each example in turn, return 1 if the example possesses the latent or 0 if it "
    "does not. You must return your response in a valid Python list. Your list must "
    "contain exactly one value, 0 or 1, per example. Do not return anything else "
    "besides a Python list.\n"
)

# delphi GENERATION_PROMPT, with delphi's example line "Example {i}: {text}" for i = 0
USER_TEMPLATE = "Latent explanation: {explanation}\n\nText examples:\n\nExample 0: {text}\n"

_BRACKETS = re.compile(r"\[.*?\]")


def build_user(explanation: str, text: str) -> str:
    return USER_TEMPLATE.format(explanation=explanation, text=text)


def parse_answer(raw: str | None) -> str:
    """Return '1', '0' or 'invalid', following delphi's Classifier._parse.

    The first [...] in the answer must be a JSON list of exactly one value,
    0, 1, true or false.
    """
    if not raw:
        return "invalid"
    match = _BRACKETS.search(raw)
    if match is None:
        return "invalid"
    try:
        values = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "invalid"
    if not isinstance(values, list) or len(values) != 1:
        return "invalid"
    v = values[0]
    if type(v) not in (bool, int) or v not in (0, 1):
        return "invalid"
    return "1" if v else "0"
