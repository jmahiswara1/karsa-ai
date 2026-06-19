"""Tests for tools/function-calling support in ai_client.chat()."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core import ai_client


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
def fake_completion():
    """A mock object that looks like a non-stream chat completion response."""
    message = MagicMock()
    message.content = "ok"
    message.tool_calls = None

    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message = message
    return completion


@pytest.fixture
def tool_call_completion():
    """A mock chat completion response that contains tool_calls."""
    tool_call = MagicMock()
    tool_call.id = "call_abc123"
    tool_call.function.name = "create_task"
    tool_call.function.arguments = '{"title": "Buy milk"}'

    message = MagicMock()
    message.content = None
    message.tool_calls = [tool_call]

    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message = message
    return completion


def _make_create_client(fake_response):
    """Return a fake AsyncOpenAI client whose .chat.completions.create returns fake_response."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=fake_response)
    return client


def _patched_provider(api_key: str = "fake-key"):
    """Build a single-provider config dict the way chat() expects."""
    return {
        "name": "Fake",
        "base_url": "https://fake.example/v1",
        "api_key": api_key,
        "model": "fake-model",
        "fallback_model": "",
    }


# ── Tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_without_tools_still_works(fake_completion):
    """Backward compat: chat() with no tools kwarg must not add tools/tool_choice."""
    fake_client = _make_create_client(fake_completion)

    fake_settings = MagicMock()
    fake_settings.ACTIVE_PROVIDER = "sumopod"
    fake_settings.FALLBACK_ORDER = ["sumopod"]
    fake_settings.provider.return_value = _patched_provider()
    fake_settings._get_auth_patterns.return_value = ["invalid_api_key"]

    with patch.object(ai_client, "settings", fake_settings), \
         patch.object(ai_client, "AsyncOpenAI", return_value=fake_client):
        msg = await ai_client.chat([{"role": "user", "content": "hi"}])

    assert msg.content == "ok"
    # tools / tool_choice must NOT be in the kwargs sent to the SDK
    call_kwargs = fake_client.chat.completions.create.await_args.kwargs
    assert "tools" not in call_kwargs
    assert "tool_choice" not in call_kwargs


@pytest.mark.asyncio
async def test_chat_with_tools_passes_tools_to_llm(fake_completion):
    """When tools is provided, chat() must forward it as `tools` to the SDK."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "create_task",
                "description": "Create a new task",
                "parameters": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                },
            },
        }
    ]

    fake_client = _make_create_client(fake_completion)
    fake_settings = MagicMock()
    fake_settings.ACTIVE_PROVIDER = "sumopod"
    fake_settings.FALLBACK_ORDER = ["sumopod"]
    fake_settings.provider.return_value = _patched_provider()
    fake_settings._get_auth_patterns.return_value = ["invalid_api_key"]

    with patch.object(ai_client, "settings", fake_settings), \
         patch.object(ai_client, "AsyncOpenAI", return_value=fake_client):
        msg = await ai_client.chat(
            [{"role": "user", "content": "Add a task to buy milk"}],
            tools=tools,
        )

    call_kwargs = fake_client.chat.completions.create.await_args.kwargs
    assert call_kwargs["tools"] is tools
    assert call_kwargs["tool_choice"] == "auto"
    assert msg.content == "ok"


@pytest.mark.asyncio
async def test_chat_with_empty_tools_list_does_not_send_tools(fake_completion):
    """Empty list is falsy -> tools/tool_choice must be omitted (matches `if tools:`)."""
    fake_client = _make_create_client(fake_completion)
    fake_settings = MagicMock()
    fake_settings.ACTIVE_PROVIDER = "sumopod"
    fake_settings.FALLBACK_ORDER = ["sumopod"]
    fake_settings.provider.return_value = _patched_provider()
    fake_settings._get_auth_patterns.return_value = ["invalid_api_key"]

    with patch.object(ai_client, "settings", fake_settings), \
         patch.object(ai_client, "AsyncOpenAI", return_value=fake_client):
        await ai_client.chat(
            [{"role": "user", "content": "hi"}],
            tools=[],
        )

    call_kwargs = fake_client.chat.completions.create.await_args.kwargs
    assert "tools" not in call_kwargs
    assert "tool_choice" not in call_kwargs


@pytest.mark.asyncio
async def test_chat_with_tools_returns_tool_calls(tool_call_completion):
    """When the LLM responds with tool_calls, chat() must return the message intact."""
    tools = [
        {
            "type": "function",
            "function": {"name": "create_task", "description": "x", "parameters": {}},
        }
    ]

    fake_client = _make_create_client(tool_call_completion)
    fake_settings = MagicMock()
    fake_settings.ACTIVE_PROVIDER = "sumopod"
    fake_settings.FALLBACK_ORDER = ["sumopod"]
    fake_settings.provider.return_value = _patched_provider()
    fake_settings._get_auth_patterns.return_value = ["invalid_api_key"]

    with patch.object(ai_client, "settings", fake_settings), \
         patch.object(ai_client, "AsyncOpenAI", return_value=fake_client):
        msg = await ai_client.chat(
            [{"role": "user", "content": "remind me to buy milk"}],
            tools=tools,
        )

    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].function.name == "create_task"
    assert msg.tool_calls[0].function.arguments == '{"title": "Buy milk"}'


@pytest.mark.asyncio
async def test_chat_signature_accepts_tools_keyword(fake_completion):
    """The chat() signature must expose `tools` as a keyword-only argument."""
    import inspect

    sig = inspect.signature(ai_client.chat)
    assert "tools" in sig.parameters
    tools_param = sig.parameters["tools"]
    # Must be keyword-only (after the `*,`).
    assert tools_param.kind is inspect.Parameter.KEYWORD_ONLY
    # Default should be None so callers can omit it.
    assert tools_param.default is None


@pytest.mark.asyncio
async def test_chat_tools_with_response_format(fake_completion):
    """tools and response_format should be combinable (response_format only set when truthy)."""
    tools = [
        {
            "type": "function",
            "function": {"name": "noop", "description": "d", "parameters": {}},
        }
    ]

    fake_client = _make_create_client(fake_completion)
    fake_settings = MagicMock()
    fake_settings.ACTIVE_PROVIDER = "sumopod"
    fake_settings.FALLBACK_ORDER = ["sumopod"]
    fake_settings.provider.return_value = _patched_provider()
    fake_settings._get_auth_patterns.return_value = ["invalid_api_key"]

    with patch.object(ai_client, "settings", fake_settings), \
         patch.object(ai_client, "AsyncOpenAI", return_value=fake_client):
        await ai_client.chat(
            [{"role": "user", "content": "hi"}],
            tools=tools,
            response_format={"type": "json_object"},
        )

    call_kwargs = fake_client.chat.completions.create.await_args.kwargs
    assert call_kwargs["tools"] is tools
    assert call_kwargs["tool_choice"] == "auto"
    assert call_kwargs["response_format"] == {"type": "json_object"}