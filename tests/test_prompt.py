from autointerp.prompt import SYSTEM, build_user, parse_answer


def test_parse_valid_lists():
    assert parse_answer("[1]") == "1"
    assert parse_answer("[0]") == "0"
    assert parse_answer(" [1]\n") == "1"
    assert parse_answer("[true]") == "1"
    assert parse_answer("The answer is [0].") == "0"


def test_parse_rejects_what_delphi_rejects():
    assert parse_answer("1") == "invalid"  # no list
    assert parse_answer("[10]") == "invalid"  # not binary
    assert parse_answer("[1, 0]") == "invalid"  # one text per call, one value
    assert parse_answer('["1"]') == "invalid"  # string, not int
    assert parse_answer("[yes]") == "invalid"  # not JSON
    assert parse_answer("") == "invalid"
    assert parse_answer(None) == "invalid"


def test_build_user_matches_delphi_layout():
    u = build_user("male pronouns", " he said")
    assert u == "Latent explanation: male pronouns\n\nText examples:\n\nExample 0:  he said\n"


def test_system_keeps_delphi_wording_except_the_label_sentence():
    assert SYSTEM.startswith("You are an intelligent and meticulous linguistics researcher.")
    assert "correctly labeled" not in SYSTEM
    assert "return 1 if the example possesses the latent or 0 if it does not" in SYSTEM


# delphi detection_prompt.py, DSCORER_SYSTEM_PROMPT, commit a11130d (verbatim)
DELPHI_SYSTEM = """You are an intelligent and meticulous linguistics researcher.

You will be given a certain latent of text, such as "male pronouns" or "text with negative sentiment".

You will then be given several text examples. Your task is to determine which examples possess the latent.

For each example in turn, return 1 if the sentence is correctly labeled or 0 if the tokens are mislabeled. You must return your response in a valid Python list. Do not return anything else besides a Python list.
"""  # noqa: E501


def test_system_is_delphi_with_only_documented_changes():
    expected = (
        DELPHI_SYSTEM.replace(
            "return 1 if the sentence is correctly labeled or 0 if the tokens are mislabeled.",
            "return 1 if the example possesses the latent or 0 if it does not.",
        )
        .replace(
            "several text examples. Your task",
            "several text examples. Each example is a short excerpt and may start or end "
            "mid-sentence. Your task",
        )
        .replace(
            "valid Python list. Do not",
            "valid Python list. Your list must contain exactly one value, 0 or 1, "
            "per example. Do not",
        )
    )
    assert SYSTEM == expected
