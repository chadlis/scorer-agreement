"""Scorer prompt (detection, one text per call). DRAFT until frozen by review."""

SYSTEM = (
    "You are a meticulous linguistics researcher. You will be given a description "
    'of a feature of text, such as "male pronouns" or "text with negative sentiment". '
    "You will then be given one short text excerpt. Your task is to decide whether "
    "the excerpt contains the feature described. Answer 1 if it does and 0 if it "
    "does not. Reply with a single character, 1 or 0, and nothing else."
)

USER_TEMPLATE = (
    "Feature description: {explanation}\n\n"
    "Text excerpt:\n"
    "<<<{text}>>>\n\n"
    "Answer (1 or 0):"
)


def build_user(explanation: str, text: str) -> str:
    return USER_TEMPLATE.format(explanation=explanation.strip(), text=text)


def parse_answer(raw: str | None) -> str:
    """Return '1', '0' or 'invalid'. Rule: first non-space character."""
    if not raw:
        return "invalid"
    s = raw.strip()
    return s[0] if s[:1] in ("0", "1") else "invalid"
