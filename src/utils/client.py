"""Shared OpenAI-compatible client factory for multi-provider support."""

import os
from functools import lru_cache
from typing import Any, Dict

from openai import OpenAI

# Model-name prefix → (base_url env var, api_key env var, key required).
#
# Keep more-specific prefixes before generic ones: ``Qwen/Qwen3.5-*`` must not
# be captured by the legacy ``Qwen`` route used by Qwen3-14B.
_PROVIDER_ROUTES = (
    ("Qwen/Qwen3.5", "QWEN35_BASE_URL", "QWEN35_API_KEY", False),
    ("qwen/qwen3.5", "QWEN35_BASE_URL", "QWEN35_API_KEY", False),
    # DashScope-compatible Qwen3.8 (1.sh: QWEN_38_*); keep before generic qwen/.
    ("qwen3.8",      "QWEN_38_BASE_URL", "QWEN_38_API_KEY", True),
    ("Qwen3.8",      "QWEN_38_BASE_URL", "QWEN_38_API_KEY", True),
    ("deepseek",     "DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY", True),
    ("gpt",          "OPENAI_BASE_URL",   "OPENAI_API_KEY", True),
    ("o1",           "OPENAI_BASE_URL",   "OPENAI_API_KEY", True),
    ("o3",           "OPENAI_BASE_URL",   "OPENAI_API_KEY", True),
    ("o4",           "OPENAI_BASE_URL",   "OPENAI_API_KEY", True),
    ("gemini",       "GEMINI_BASE_URL",   "GEMINI_API_KEY", True),
    ("qwen",         "QWEN_BASE_URL",     "QWEN_API_KEY", True),
    ("Qwen",         "QWEN_BASE_URL",     "QWEN_API_KEY", True),
    ("claude",       "OPENAI_BASE_URL",   "OPENAI_API_KEY", True),
)


def _resolve_provider(model: str) -> tuple[str, str]:
    """Return (base_url, api_key) for a given model name, reading from env.

    If the model name matches a known provider prefix but the corresponding
    env vars are missing, raise loudly — silently falling back to OPENAI_*
    would route the call to the wrong provider (and wrong billing account).
    """
    for prefix, url_env, key_env, key_required in _PROVIDER_ROUTES:
        if model.startswith(prefix):
            base_url = os.environ.get(url_env, "")
            api_key = os.environ.get(key_env, "")
            missing = [url_env] if not base_url else []
            if key_required and not api_key:
                missing.append(key_env)
            if missing:
                raise RuntimeError(
                    f"Model '{model}' matches provider prefix '{prefix}' but required "
                    f"env var(s) are not set: {', '.join(missing)}. Refusing to silently "
                    f"fall back to a different provider. Please source your env config "
                    f"(e.g. `source 1.sh`) or export the variables before running."
                )
            # The OpenAI SDK requires a non-empty key even when a compatible
            # local endpoint performs no authentication. Avoid falling back to
            # OPENAI_API_KEY (and leaking it to that endpoint) by using a
            # harmless placeholder for explicitly keyless routes.
            return base_url, api_key or "not-required"
    # No known prefix → generic OpenAI-compatible fallback
    return os.environ.get("OPENAI_BASE_URL", ""), os.environ.get("OPENAI_API_KEY", "")


def make_client(model: str) -> OpenAI:
    """Build a new OpenAI client for ``model`` (no process-wide cache)."""
    base_url, api_key = _resolve_provider(model)
    kwargs: Dict[str, Any] = {}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    return OpenAI(**kwargs)


@lru_cache(maxsize=8)
def get_client(model: str) -> OpenAI:
    """Get or create an OpenAI client for the given model name.

    Clients are cached by model name so the same provider reuses one client
    inside a process. Parallel eval workers should call ``make_client`` instead.
    """
    return make_client(model)
