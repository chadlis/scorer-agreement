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
