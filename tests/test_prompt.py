from autointerp.prompt import build_user, parse_answer


def test_parse_clear_answers():
    assert parse_answer("1") == "1"
    assert parse_answer("0") == "0"
    assert parse_answer(" 1\n") == "1"


def test_parse_unreadable_answers():
    assert parse_answer("Yes") == "invalid"
    assert parse_answer("") == "invalid"
    assert parse_answer(None) == "invalid"


def test_build_user_fills_both_slots():
    u = build_user("  male pronouns ", "he said")
    assert "Feature description: male pronouns\n" in u
    assert "<<<he said>>>" in u
