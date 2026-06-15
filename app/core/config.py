import os

from pydantic_settings import BaseSettings

# All providers in one place. To add a new provider: add an entry in
# `provider()`, plus the env vars in docker-compose.yml, .env.example, and
# your local .env.
_PROVIDER_KEYS: tuple[str, ...] = ("sumopod", "deepseek", "bai")


def _csv(name: str, default: str) -> list[str]:
    """Read a comma-separated env var and return a clean lowercase list."""
    raw = os.getenv(name, default)
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


class Settings(BaseSettings):
    # ── Service auth ─────────────────────────────────
    AI_SERVICE_TOKEN: str = "default-service-token"

    # ── Active provider + fallback order ─────────────
    # ACTIVE_PROVIDER is tried first; if it fails the loop walks FALLBACK_ORDER.
    ACTIVE_PROVIDER: str = ""
    FALLBACK_ORDER: list[str] = []

    # ── Sumopod ──────────────────────────────────────
    SUMOPOD_BASE_URL: str = ""
    SUMOPOD_API_KEY: str = ""
    # Backward-compat alias: older configs used OPENAI_API_KEY for sumopod.
    OPENAI_API_KEY: str = ""
    SUMOPOD_MODEL: str = ""
    SUMOPOD_FALLBACK_MODEL: str = ""
    # Backward-compat alias: older code referenced MODEL_NAME.
    MODEL_NAME: str = ""

    # ── DeepSeek ─────────────────────────────────────
    DEEPSEEK_BASE_URL: str = ""
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = ""
    DEEPSEEK_FALLBACK_MODEL: str = ""

    # ── b.ai ─────────────────────────────────────────
    BAI_BASE_URL: str = ""
    BAI_API_KEY: str = ""
    BAI_MODEL: str = ""
    BAI_FALLBACK_MODEL: str = ""

    # ── Error handling ─────────────────────────────
    # Comma-separated patterns to detect auth errors (treated as retriable).
    AUTH_ERROR_PATTERNS: list[str] = []

    model_config = {"env_file": ".env", "extra": "ignore"}

    def model_post_init(self, _ctx) -> None:
        # Resolve aliases
        if not self.SUMOPOD_API_KEY and self.OPENAI_API_KEY:
            self.SUMOPOD_API_KEY = self.OPENAI_API_KEY
        if not self.MODEL_NAME:
            self.MODEL_NAME = self.SUMOPOD_MODEL

        # Defaults (derived from env vars, not hardcoded)
        if not self.ACTIVE_PROVIDER:
            self.ACTIVE_PROVIDER = _PROVIDER_KEYS[0]
        if not self.FALLBACK_ORDER:
            self.FALLBACK_ORDER = _csv("FALLBACK_ORDER", ",".join(_PROVIDER_KEYS))
        if not self.AUTH_ERROR_PATTERNS:
            self.AUTH_ERROR_PATTERNS = _csv(
                "AUTH_ERROR_PATTERNS", "authentication,invalid_api_key,permission"
            )

    # ── Provider registry (computed at access time) ──
    def provider(self, name: str) -> dict[str, str]:
        """Return base_url / api_key / model / fallback_model / display_name for a provider."""
        n = name.lower()
        if n not in _PROVIDER_KEYS:
            raise ValueError(
                f"Provider '{name}' tidak dikenal. Pilihan: {list(_PROVIDER_KEYS)}"
            )

        _REGISTRY = {
            "sumopod": {
                "name": "Sumopod",
                "base_url": self.SUMOPOD_BASE_URL,
                "api_key": self.SUMOPOD_API_KEY,
                "model": self.SUMOPOD_MODEL,
                "fallback_model": self.SUMOPOD_FALLBACK_MODEL,
            },
            "deepseek": {
                "name": "DeepSeek",
                "base_url": self.DEEPSEEK_BASE_URL,
                "api_key": self.DEEPSEEK_API_KEY,
                "model": self.DEEPSEEK_MODEL,
                "fallback_model": self.DEEPSEEK_FALLBACK_MODEL,
            },
            "bai": {
                "name": "b.ai",
                "base_url": self.BAI_BASE_URL,
                "api_key": self.BAI_API_KEY,
                "model": self.BAI_MODEL,
                "fallback_model": self.BAI_FALLBACK_MODEL,
            },
        }
        return _REGISTRY[n]

    def known_providers(self) -> list[str]:
        return list(_PROVIDER_KEYS)


settings = Settings()
