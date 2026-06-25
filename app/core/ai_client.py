"""Async LLM client dengan auto-fallback antar provider (sumopod / b.ai / deepseek)."""
import json
import logging
import re

from openai import (
    APIConnectionError,
    APIError,
    AsyncOpenAI,
    OpenAIError,
    RateLimitError,
)

from app.core.config import settings

logger = logging.getLogger(__name__)

# Errors that justify moving on to the next provider in the fallback chain.
RETRIABLE_ERRORS = (
    APIConnectionError,
    RateLimitError,
    APIError,
    TimeoutError,
    ConnectionError,
)


async def chat(
    messages: list[dict],
    *,
    provider: str | None = None,
    fallback: list[str] | None = None,
    response_format: dict | None = None,
    stream: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    tools: list[dict] | None = None,
    **kwargs,
):
    """Kirim chat ke LLM dengan auto-fallback.

    Args:
        messages:        list pesan OpenAI format
        provider:        provider utama (default = ACTIVE_PROVIDER dari .env)
        fallback:        override urutan fallback (default = FALLBACK_ORDER)
        response_format: mis. {"type": "json_object"} untuk mode JSON
        stream:          True untuk stream response
        temperature,
        max_tokens,
        tools:           list tool definitions (OpenAI function-calling format)
        **kwargs:        parameter lain yang diteruskan ke API

    Returns:
        - non-stream: `response.choices[0].message` (object dengan atribut `.content` dan `.tool_calls`)
        - stream: async iterator dari chunks

    Raises:
        ValueError: kalau nama provider tak dikenal atau API key kosong
        RuntimeError: kalau semua provider di chain gagal (network/rate limit)
    """
    primary = (provider or settings.ACTIVE_PROVIDER).lower()
    fb_list = fallback if fallback is not None else settings.FALLBACK_ORDER
    chain = [primary] + [p for p in fb_list if p.lower() != primary]

    last_error: Exception | None = None

    for p in chain:
        cfg = settings.provider(p)

        # Build model list: primary model + optional fallback model
        models = [cfg["model"]]
        if cfg.get("fallback_model"):
            fb_model = cfg["fallback_model"]
            if fb_model != cfg["model"]:
                models.append(fb_model)

        for i, model in enumerate(models):
            try:
                if not cfg["api_key"]:
                    logger.warning("[%s] API key belum di-set, lewati.", p)
                    last_error = ValueError(f"API key untuk provider '{p}' kosong")
                    break  # break inner loop, continue to next provider

                client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
                tag = f"{p}" if i == 0 else f"{p}/{model}"
                logger.info("[%s] %s (%s)", tag, cfg["name"], model)

                call_kwargs: dict = {
                    "model": model,
                    "messages": messages,
                    "stream": stream,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **kwargs,
                }
                if response_format:
                    call_kwargs["response_format"] = response_format
                if tools:
                    call_kwargs["tools"] = tools
                    call_kwargs["tool_choice"] = "auto"

                response = await client.chat.completions.create(**call_kwargs)

                if stream:
                    return response
                return response.choices[0].message

            except ValueError as e:
                raise

            except RETRIABLE_ERRORS as e:
                logger.warning("[%s] gagal: %s: %s", tag, type(e).__name__, e)
                last_error = e
                continue  # try next model in the list

            except OpenAIError as e:
                # Auth error (401/403): treat as retriable across the chain
                error_str = str(e).lower()
                if any(pattern in error_str for pattern in settings.AUTH_ERROR_PATTERNS):
                    logger.warning("[%s] auth error, coba provider/model lain: %s", tag, e)
                    last_error = e
                    break  # skip all models for this provider, try next provider
                # Other OpenAI error (bad request, etc). Fatal.
                logger.error("[%s] OpenAI error (fatal): %s", tag, e)
                raise

    raise RuntimeError(
        f"Semua provider gagal. Terakhir: {type(last_error).__name__}: {last_error}"
    ) from last_error


# ── JSON helper (dipakai semua endpoint) ─────────────────────

def parse_json_response(content: str) -> dict:
    """Parse JSON dari response LLM. Tangani ```json ... ``` wrapper."""
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
    if match:
        content = match.group(1)
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse AI response as JSON: {e}\nRaw content: {content}"
        )
