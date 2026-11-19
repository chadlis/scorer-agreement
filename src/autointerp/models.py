"""Scorer registry: model ids and per-model settings, written into every CSV row."""

SCORERS = {
    "haiku-5.5": dict(
        api="anthropic",
        model="claude-haiku-5-5",
        params=dict(thinking={"type": "disabled"}, output_config={"effort": "low"}),
    ),
    "sonnet-5.5": dict(
        api="anthropic",
        model="claude-sonnet-5-5",
        params=dict(
            thinking={"type": "between_tools"}, output_config={"effort": "low"}
        ),
    ),
    "opus-5.5": dict(
        api="anthropic",
        model="claude-opus-5-5",
        params=dict(output_config={"effort": "low"}),
    ),  # thinking cannot be disabled
    "gemini-flash": dict(
        api="openrouter",
        model="google/gemini-3.8-flash",
        params=dict(temperature=0, reasoning={"effort": "minimal"}),
        provider={
            "order": ["google-ai-studio"],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    ),
    "mistral-small": dict(
        api="openrouter",
        model="mistralai/mistral-small-3.2-24b-instruct",
        params=dict(temperature=0),
        provider={
            "order": ["parasail/bf16"],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    ),
}
MAX_TOKENS = 2048  # margin: thinking tokens count toward max_tokens
