"""Runtime LLM connection settings (API key, base URL, model) for the web UI.

The web app's Settings panel lets a user point GPT Researcher at an
OpenAI-compatible endpoint without editing files by hand. Values are persisted
to the project's ``.env`` file -- the same store the README documents for API
keys -- and mirrored into ``os.environ`` so the next research run picks them up
without a server restart.

Security notes:
    * The API key is never returned to a client in full; only a masked preview
      is exposed so the UI can show that a key is stored.
    * Secret values are never logged.
    * Values are validated to reject control characters so they cannot be used
      to inject additional lines into ``.env``.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from dotenv import dotenv_values, set_key

logger = logging.getLogger(__name__)

# backend/server/llm_settings.py -> backend/server -> backend -> repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"

API_KEY_VAR = "OPENAI_API_KEY"
BASE_URL_VAR = "OPENAI_BASE_URL"
MODEL_VARS = ("FAST_LLM", "SMART_LLM", "STRATEGIC_LLM")

DEFAULT_PROVIDER = "openai"

MANAGED_KEYS: tuple[str, ...] = (API_KEY_VAR, BASE_URL_VAR, *MODEL_VARS)

MAX_API_KEY_LENGTH = 512
MAX_BASE_URL_LENGTH = 2048
MAX_MODEL_LENGTH = 256

# Values that would let a user break out of their `.env` line. python-dotenv
# escapes quotes but not line breaks, so any control character is rejected.

def _has_control_char(value: str) -> bool:
    """True when `value` contains a character that must not reach `.env`."""
    return any(ord(char) < 32 for char in value)


class SettingsValidationError(ValueError):
    """Raised when a submitted setting fails validation."""


def _clean(value: str, field: str, max_length: int) -> str:
    """Trim and validate a user-supplied value."""
    if not isinstance(value, str):
        raise SettingsValidationError(f"{field} must be a string")
    value = value.strip()
    if _has_control_char(value):
        raise SettingsValidationError(f"{field} contains unsupported characters")
    if len(value) > max_length:
        raise SettingsValidationError(f"{field} is too long (max {max_length} characters)")
    return value


def _validate_base_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise SettingsValidationError(
            "Base URL must be a valid http:// or https:// URL, e.g. http://localhost:11434/v1"
        )
    return value


def _normalize_model(value: str) -> str:
    """Return a `provider:model` string, defaulting to the openai provider."""
    from gpt_researcher.llm_provider.generic.base import _SUPPORTED_PROVIDERS

    provider, separator, model = value.partition(":")
    if separator and provider in _SUPPORTED_PROVIDERS:
        normalized = f"{provider}:{model.strip()}"
    else:
        normalized = f"{DEFAULT_PROVIDER}:{value}"

    _, _, model_name = normalized.partition(":")
    if not model_name:
        raise SettingsValidationError(
            "Model ID must be a model name such as 'gpt-4o' or 'provider:model'"
        )
    return normalized


def mask_secret(secret: Optional[str]) -> str:
    """Return a masked preview of a secret, never the full value."""
    if not secret:
        return ""
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}{'*' * 8}{secret[-4:]}"


def _read_env_file() -> Dict[str, Optional[str]]:
    if not ENV_PATH.exists():
        return {}
    try:
        return dict(dotenv_values(str(ENV_PATH)))
    except Exception as exc:  # pragma: no cover - unreadable/malformed .env
        logger.warning("Could not read %s: %s", ENV_PATH.name, exc)
        return {}


def _harden_env_file() -> None:
    """Restrict `.env` permissions to the owner where the platform supports it."""
    if os.name != "posix":
        return
    try:
        os.chmod(ENV_PATH, stat.S_IRUSR | stat.S_IWUSR)
    except OSError as exc:  # pragma: no cover - best effort only
        logger.debug("Could not restrict permissions on %s: %s", ENV_PATH.name, exc)


def _write_env_value(key: str, value: Optional[str]) -> None:
    """Set ``key`` to ``value`` in .env using safe parsing, or remove it.

    python-dotenv's ``set_key(..., quote_mode='always')`` writes literal quote
    characters into the file.  On some versions those quotes are *not* stripped
    by dotenv itself, causing downstream config parsers (e.g. Config.parse_llm)
    to see e.g. ``'openai`` as a provider name.  We avoid this by removing
    existing keys first and writing raw lines ourselves.
    """
    if value is None:
        if not ENV_PATH.exists():
            return
        _remove_env_key(key)
        return

    if not ENV_PATH.exists():
        ENV_PATH.touch()

    # Remove any prior definition of this key
    _remove_env_key(key)

    try:
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError:
        lines = []

    new_lines = lines + [f"{key}={value}\n"]
    ENV_PATH.write_text("".join(new_lines), encoding="utf-8")
    _harden_env_file()


def _remove_env_key(key: str) -> None:
    """Drop a single key from `.env` while preserving comments and formatting."""
    try:
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError as exc:  # pragma: no cover - best effort only
        logger.warning("Could not update %s: %s", ENV_PATH.name, exc)
        return

    prefixes = (f"{key}=", f"export {key}=")
    kept: List[str] = [line for line in lines if not line.lstrip().startswith(prefixes)]
    if kept != lines:
        ENV_PATH.write_text("".join(kept), encoding="utf-8")


def _apply_to_environ(updates: Dict[str, Optional[str]], override: bool) -> None:
    """Mirror settings into `os.environ` so the next run uses them."""
    for key, value in updates.items():
        if value is None:
            if override:
                os.environ.pop(key, None)
            continue
        if override or not os.environ.get(key):
            os.environ[key] = value


def apply_persisted_settings(override: bool = False) -> None:
    """Load settings saved through the web UI into the process environment.

    Values already present in the real environment win unless `override` is
    set, matching python-dotenv's usual "environment over .env" precedence.
    """
    stored = _read_env_file()
    pending: Dict[str, Optional[str]] = {}
    for key in MANAGED_KEYS:
        value = stored.get(key)
        if value:
            pending[key] = value
    if pending:
        _apply_to_environ(pending, override=override)
        logger.info("Loaded %d saved LLM setting(s) from %s", len(pending), ENV_PATH.name)


def get_llm_settings() -> Dict[str, Any]:
    """Return the effective LLM settings with the API key masked."""
    stored = _read_env_file()
    api_key = os.environ.get(API_KEY_VAR) or stored.get(API_KEY_VAR) or ""

    model = ""
    for key in MODEL_VARS:
        model = os.environ.get(key) or stored.get(key) or ""
        if model:
            break

    provider, separator, model_name = model.partition(":")
    if not separator:
        provider, model_name = (DEFAULT_PROVIDER, model) if model else ("", "")

    persisted = bool(stored.get(API_KEY_VAR) or stored.get(BASE_URL_VAR) or any(stored.get(k) for k in MODEL_VARS))

    return {
        "api_key_set": bool(api_key),
        "api_key_masked": mask_secret(api_key),
        "base_url": os.environ.get(BASE_URL_VAR) or stored.get(BASE_URL_VAR) or "",
        "model_id": model,
        "model_provider": provider,
        "model_name": model_name,
        "persisted": persisted,
    }


def save_llm_settings(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_id: Optional[str] = None,
    clear_api_key: bool = False,
) -> Dict[str, Any]:
    """Validate, persist and immediately apply LLM settings.

    `api_key` is left untouched when blank so the UI can submit a form without
    the user retyping the stored key; pass `clear_api_key=True` to remove it.
    `base_url` and `model_id` are always applied, and an empty string clears
    them.
    """
    updates: Dict[str, Optional[str]] = {}

    if clear_api_key:
        updates[API_KEY_VAR] = None
    elif api_key is not None and api_key.strip():
        updates[API_KEY_VAR] = _clean(api_key, "API key", MAX_API_KEY_LENGTH)

    if base_url is not None:
        cleaned_url = _clean(base_url, "Base URL", MAX_BASE_URL_LENGTH)
        updates[BASE_URL_VAR] = _validate_base_url(cleaned_url) if cleaned_url else None

    if model_id is not None:
        cleaned_model = _clean(model_id, "Model ID", MAX_MODEL_LENGTH)
        normalized_model = _normalize_model(cleaned_model) if cleaned_model else None
        for key in MODEL_VARS:
            updates[key] = normalized_model

    for key, value in updates.items():
        _write_env_value(key, value)
    _apply_to_environ(updates, override=True)

    # Log the shape of the change only -- never the values themselves.
    cleared = sorted(key for key, value in updates.items() if value is None)
    written = sorted(key for key, value in updates.items() if value is not None)
    logger.info(
        "LLM settings updated (saved: %s; cleared: %s)",
        ", ".join(written) or "none",
        ", ".join(cleared) or "none",
    )

    return get_llm_settings()


def test_connection() -> Dict[str, Any]:
    """Attempt a lightweight call to the configured LLM endpoint.

    Reads the current environment (already synced with .env via
    ``_apply_to_environ``), builds an OpenAI-compatible ChatClient, and sends
    a minimal prompt. Errors are caught and returned so the UI can show a clear
    message without crashing.
    """
    try:
        api_key = os.environ.get(API_KEY_VAR) or _read_env_file().get(API_KEY_VAR) or ""
        if not api_key:
            return {
                "success": False,
                "message": "No API key set",
                "details": "Enter an API key in the Settings panel before testing.",
            }

        # Read the effective model from env (or .env as fallback)
        raw_model = ""
        for key in MODEL_VARS:
            raw_model = os.environ.get(key) or _read_env_file().get(key) or ""
            if raw_model:
                break

        if not raw_model:
            return {
                "success": False,
                "message": "No LLM model configured",
                "details": "Set SMART_LLM or FAST_LLM before testing.",
            }

        # Normalize: ensure provider:model format.
        normalized_model = _normalize_model(raw_model)
        provider_name, _, model_name = normalized_model.partition(":")

        # Build kwargs based on provider type.
        provider_kwargs: Dict[str, Any] = {"model": model_name}
        base_url = os.environ.get(BASE_URL_VAR) or _read_env_file().get(BASE_URL_VAR)
        if base_url:
            provider_kwargs["openai_api_base"] = base_url

        # Lazy-load the right client class.
        global _OPENAI_CLIENT_CLS
        if _OPENAI_CLIENT_CLS is None:
            _load_openai_client()

        if provider_name == "openai" or (_OPENAI_CLIENT_CLS and provider_name not in _PROVIDER_SPECIAL):
            if _OPENAI_CLIENT_CLS:
                provider_kwargs["api_key"] = api_key
                client = _OPENAI_CLIENT_CLS(**provider_kwargs)
                return _send_test_message(client)
            else:
                raise ImportError("langchain_openai.ChatOpenAI not available")

        # For non-OpenAI providers, fall back to GenericLLMProvider.
        from gpt_researcher.llm_provider.generic.base import GenericLLMProvider

        llm_instance = GenericLLMProvider.from_provider(provider_name, **provider_kwargs)
        return _send_test_message(llm_instance)

    except Exception as exc:  # pragma: no cover - depends on installed packages
        msg = str(exc).strip().encode("ascii", "replace").decode("ascii")
        return {
            "success": False,
            "message": "Failed to reach the endpoint",
            "details": f"{exc.__class__.__name__}: {msg[:150]}",
        }


# Providers whose models are passed through OpenAI-compatible ChatOpenAI
# (i.e., they speak the OpenAI chat completions API).
_PROVIDERS_THROUGH_OPENAI = frozenset({
    "openai",
    "vllm_openai",
    "litellm",
    "openrouter",
    "together",
    "mistralai",
    "fireworks",
    "groq",
    "huggingface",
    "cohere",
})

# Set once by _load_openai_client.
_OPENAI_CLIENT_CLS: Any = None


def _load_openai_client() -> None:
    """Lazy-load langchain_openai.ChatOpenAI for connection testing."""
    try:
        from langchain_openai import ChatOpenAI as CoC
    except ImportError:
        CoC = None  # type: ignore
    global _OPENAI_CLIENT_CLS
    _OPENAI_CLIENT_CLS = CoC


def _send_test_message(client: Any) -> Dict[str, Any]:
    """Send a minimal prompt and return a result dict."""
    try:
        # Prefer the standard langchain invocation signature.
        if hasattr(client, "invoke"):
            from langchain_core.messages import HumanMessage

            response = client.invoke([HumanMessage(content="Say 'ok'")])
            content = getattr(response, "content", "") or ""
        elif hasattr(client, "create_chat_completion"):
            # Some internal wrappers use this signature.
            resp = client.create_chat_completion(
                messages=[{"role": "user", "content": "Say 'ok'"}],
                config_dict=None,
                stream=False,
            )
            content = (
                getattr(resp, "choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
        else:
            # Last resort: bare openai-python API via httpx.
            import httpx

            base_url = getattr(client, "base_url", None)
            headers = {"Authorization": f"Bearer {getattr(client, 'api_key', '')}", "Content-Type": "application/json"}
            payload = {
                "model": getattr(client, "model", ""),
                "messages": [{"role": "user", "content": "Say 'ok'"}],
            }
            url = f"{base_url}/chat/completions" if base_url else "https://api.openai.com/v1/chat/completions"
            with httpx.Client() as http:
                r = http.post(url.strip("/") + "/chat/completions", json=payload, headers=headers, timeout=30)
                data = r.json()
                content = data["choices"][0]["message"]["content"]
        if content and len(str(content)) > 0:
            return {
                "success": True,
                "message": "Connection successful",
                "details": f"Model responded ({len(str(content))} chars)",
            }
        return {
            "success": False,
            "message": "Endpoint returned an empty response",
            "details": "Check that the model accepts short prompts.",
        }
    except Exception as inner:
        raise inner from None


# Map of supported provider names to their actual classes so we don't always
# use GenericLLMProvider (which defaults to openai internally).
# Kept for future extension; currently test_connection uses ChatOpenAI directly.
