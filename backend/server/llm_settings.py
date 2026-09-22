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
    """Set `key` to `value` in `.env`, or remove it when `value` is None."""
    if value is None:
        if not ENV_PATH.exists():
            return
        _remove_env_key(key)
        return

    if not ENV_PATH.exists():
        ENV_PATH.touch()
    set_key(str(ENV_PATH), key, value, quote_mode="always")
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
