"""Scorer prompt: delphi's detection prompt with two documented changes.

Base: EleutherAI/delphi, delphi/scorers/classifier/prompts/detection_prompt.py at
commit a11130d6cf9493ac7feb9205b4fa9edf67cadb66, default configuration
(n_examples_shown=1, one text per call). Everything is kept verbatim except:

1. The three solved few-shot examples are removed (zero-shot). Three of their
   15 labels contradict the activations recorded on Neuronpedia, and two of the
   three examples come from the SAE studied here (gpt2-small/6-res-jb, features
   6048 and 9396).
2. "return 1 if the sentence is correctly labeled or 0 if the tokens are
   mislabeled" is replaced by "return 1 if the example possesses the latent or 0
   if it does not". The original sentence comes from delphi's fuzzing prompt,
   where marked tokens are labels. Detection shows no labels.

The answer is parsed with delphi's rule (Classifier._parse).
DRAFT until frozen by review.
"""

import json
import re

SYSTEM = (
    "You are an intelligent and meticulous linguistics researcher.\n\n"
    'You will be given a certain latent of text, such as "male pronouns" or '
    '"text with negative sentiment".\n\n'
    "You will then be given several text examples. Your task is to determine which "
    "examples possess the latent.\n\n"
    "For each example in turn, return 1 if the example possesses the latent or 0 if it "
    "does not. You must return your response in a valid Python list. Do not return "
    "anything else besides a Python list.\n"
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
