"""Scorer registry: model ids and per-model settings, written into every CSV row."""

SCORERS = {
    "haiku-5.5": {
        "api": "anthropic",
        "model": "claude-haiku-5-5",
        "params": {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}},
    },
    "sonnet-5.5": {
        "api": "anthropic",
        "model": "claude-sonnet-5-5",
        "params": {"thinking": {"type": "between_tools"}, "output_config": {"effort": "low"}},
    },
    "opus-5.5": {
        "api": "anthropic",
        "model": "claude-opus-5-5",
        "params": {"output_config": {"effort": "low"}},
    },  # thinking cannot be disabled
    "gemini-flash": {
        "api": "openrouter",
        "model": "google/gemini-2.5-flash",
        "params": {"temperature": 0, "reasoning": {"enabled": False}},
        "provider": {
            "order": ["google-ai-studio"],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    },
    "mistral-small": {
        "api": "openrouter",
        "model": "mistralai/mistral-small-3.2-24b-instruct",
        "params": {"temperature": 0},
        "provider": {
            "order": ["parasail/bf16"],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    },
}
MAX_TOKENS = 2048  # margin: thinking tokens count toward max_tokens
