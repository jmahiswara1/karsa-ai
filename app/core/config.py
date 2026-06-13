import os

from pydantic_settings import BaseSettings

# All providers in one place. To add a new provider: add an entry in
# `provider()`, plus the env vars in docker-compose.yml, .env.example, and
# your local .env.
_PROVIDER_KEYS: tuple[str, ...] = ("bai", "deepseek", "sumopod")


def _csv(name: str, default: str) -> list[str]:
    """Read a comma-separated env var and return a clean lowercase list."""
    raw = os.getenv(name, default)
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


class Settings(BaseSettings):
    # ── Service auth ─────────────────────────────────
    AI_SERVICE_TOKEN: str = "default-service-token"

    # ── Active provider + fallback order ─────────────
    # ACTIVE_PROVIDER is tried first; if it fails the loop walks FALLBACK_ORDER.
    ACTIVE_PROVIDER: str = "bai"
    FALLBACK_ORDER: list[str] = []

    # ── b.ai ─────────────────────────────────────────
    BAI_BASE_URL: str = "https://api.b.ai/v1"
    BAI_API_KEY: str = ""
    BAI_MODEL: str = "DeepSeek-V4-Pro"

    # ── DeepSeek (direct) ───────────────────────────
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = "DeepSeek-V4-Pro"

    # ── Sumopod ──────────────────────────────────────
    SUMOPOD_BASE_URL: str = "https://ai.sumopod.com/v1"
    SUMOPOD_API_KEY: str = ""
    # Backward-compat alias: older configs used OPENAI_API_KEY for sumopod.
    OPENAI_API_KEY: str = ""
    SUMOPOD_MODEL: str = "gpt-4o-mini"
    # Backward-compat alias: older code referenced MODEL_NAME.
    MODEL_NAME: str = ""

    model_config = {"env_file": ".env", "extra": "ignore"}

    def model_post_init(self, _ctx) -> None:
        # Resolve aliases so downstream code can rely on a single name per provider.
        if not self.SUMOPOD_API_KEY and self.OPENAI_API_KEY:
            self.SUMOPOD_API_KEY = self.OPENAI_API_KEY
        if not self.MODEL_NAME:
            self.MODEL_NAME = self.SUMOPOD_MODEL
        if not self.FALLBACK_ORDER:
            self.FALLBACK_ORDER = _csv("FALLBACK_ORDER", "bai,deepseek,sumopod")

    # ── Provider registry (computed at access time) ──
    def provider(self, name: str) -> dict[str, str]:
        """Return base_url / api_key / model / display_name for a provider."""
        n = name.lower()
        if n == "bai":
            return {
                "name": "b.ai",
                "base_url": self.BAI_BASE_URL,
                "api_key": self.BAI_API_KEY,
                "model": self.BAI_MODEL,
            }
        if n == "deepseek":
            return {
                "name": "DeepSeek",
                "base_url": self.DEEPSEEK_BASE_URL,
                "api_key": self.DEEPSEEK_API_KEY,
                "model": self.DEEPSEEK_MODEL,
            }
        if n == "sumopod":
            return {
                "name": "Sumopod",
                "base_url": self.SUMOPOD_BASE_URL,
                "api_key": self.SUMOPOD_API_KEY or self.OPENAI_API_KEY,
                "model": self.SUMOPOD_MODEL,
            }
        raise ValueError(
            f"Provider '{name}' tidak dikenal. Pilihan: {list(_PROVIDER_KEYS)}"
        )

    def known_providers(self) -> list[str]:
        return list(_PROVIDER_KEYS)


settings = Settings()
