"""Tests for POST /assistant/create-entities endpoint.

The endpoint accepts a user prompt, calls the LLM with function-calling
tools, and returns a structured reply + parsed tool_calls list.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.action import (
    CreateEntitiesRequest,
    CreateEntitiesResponse,
    ToolCallResult,
)
from app.core.security import settings as security_settings


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
def auth_headers():
    """Bypass the service-token dependency for direct endpoint tests."""
    return {"Authorization": f"Bearer {security_settings.AI_SERVICE_TOKEN}"}


def _make_tool_call(name: str, arguments: dict) -> MagicMock:
    """Build a mock OpenAI tool_call object."""
    tool_call = MagicMock()
    tool_call.id = f"call_{name}"
    tool_call.function.name = name
    tool_call.function.arguments = json.dumps(arguments)
    return tool_call


def _make_message(
    content: str | None = "Some reply",
    tool_calls: list | None = None,
) -> MagicMock:
    """Build a mock OpenAI chat message."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    return message


def _patched_chat(return_value: MagicMock):
    """Return a context-manager helper that patches app.api.routes.chat."""
    return patch("app.api.routes.chat", AsyncMock(return_value=return_value))


# ── Tests ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_entities_with_tool_calls(auth_headers):
    """Endpoint returns parsed tool_calls when the LLM emits them."""
    tool_calls = [
        _make_tool_call("create_task", {"title": "Design homepage", "projectName": "Website Redesign"}),
    ]
    fake_message = _make_message(
        content="Baik, saya akan membuat task 'Design homepage' untuk project Website Redesign.",
        tool_calls=tool_calls,
    )

    with _patched_chat(fake_message):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/assistant/create-entities",
                headers=auth_headers,
                json={
                    "prompt": "Buat task design homepage untuk project Website Redesign",
                    "context": {
                        "projects": [{"id": "abc-123", "title": "Website Redesign"}],
                        "folders": [],
                        "recentTasks": [],
                    },
                },
            )

    assert resp.status_code == 200
    body = resp.json()
    assert "reply" in body
    assert "tool_calls" in body
    assert body["reply"].startswith("Baik,")
    assert isinstance(body["tool_calls"], list)
    assert len(body["tool_calls"]) == 1

    tc = body["tool_calls"][0]
    assert tc["name"] == "create_task"
    assert tc["arguments"] == {
        "title": "Design homepage",
        "projectName": "Website Redesign",
    }


@pytest.mark.asyncio
async def test_create_entities_no_tool_calls(auth_headers):
    """Endpoint returns empty tool_calls list when the LLM only replies."""
    fake_message = _make_message(
        content="Halo! Ada yang bisa saya bantu?",
        tool_calls=None,
    )

    with _patched_chat(fake_message):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/assistant/create-entities",
                headers=auth_headers,
                json={"prompt": "Halo"},
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"] == "Halo! Ada yang bisa saya bantu?"
    assert body["tool_calls"] == []


@pytest.mark.asyncio
async def test_create_entities_invalid_request_missing_prompt(auth_headers):
    """Sending a body without the required `prompt` field yields 422."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/api/assistant/create-entities",
            headers=auth_headers,
            json={"context": {"projects": []}},
        )

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_entities_invalid_request_empty_prompt(auth_headers):
    """An empty string for `prompt` violates the min_length=1 constraint."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/api/assistant/create-entities",
            headers=auth_headers,
            json={"prompt": ""},
        )

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_entities_calls_chat_with_tools(auth_headers):
    """Endpoint must call chat() with the system prompt, user prompt, and tools."""
    fake_message = _make_message(content="ok", tool_calls=None)

    with _patched_chat(fake_message) as mock_chat, \
         patch("app.api.routes.get_tools", return_value=[{"type": "function"}]) as mock_tools:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/assistant/create-entities",
                headers=auth_headers,
                json={"prompt": "Buat task test"},
            )

    assert resp.status_code == 200
    mock_chat.assert_awaited_once()
    call_kwargs = mock_chat.await_args.kwargs
    assert call_kwargs["tools"] == [{"type": "function"}]

    sent_messages = call_kwargs["messages"]
    assert sent_messages[0]["role"] == "system"
    assert "Karsa" in sent_messages[0]["content"]
    assert sent_messages[1] == {"role": "user", "content": "Buat task test"}


@pytest.mark.asyncio
async def test_create_entities_parses_multiple_tool_calls(auth_headers):
    """Multiple tool_calls from the LLM are all preserved in order."""
    tool_calls = [
        _make_tool_call("create_task", {"title": "Task 1"}),
        _make_tool_call("create_note", {"title": "Note A", "content": "body"}),
    ]
    fake_message = _make_message(content="Membuat 2 hal", tool_calls=tool_calls)

    with _patched_chat(fake_message):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/assistant/create-entities",
                headers=auth_headers,
                json={"prompt": "Buat task dan note"},
            )

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["tool_calls"]) == 2
    assert body["tool_calls"][0]["name"] == "create_task"
    assert body["tool_calls"][1]["name"] == "create_note"
    assert body["tool_calls"][1]["arguments"] == {"title": "Note A", "content": "body"}


@pytest.mark.asyncio
async def test_create_entities_handles_invalid_json_arguments(auth_headers):
    """If the LLM returns malformed JSON in arguments, fall back to {}."""
    bad_tool_call = MagicMock()
    bad_tool_call.id = "call_bad"
    bad_tool_call.function.name = "create_task"
    bad_tool_call.function.arguments = "not valid json {"

    fake_message = _make_message(content="ok", tool_calls=[bad_tool_call])

    with _patched_chat(fake_message):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/assistant/create-entities",
                headers=auth_headers,
                json={"prompt": "Buat sesuatu"},
            )

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["tool_calls"]) == 1
    assert body["tool_calls"][0]["name"] == "create_task"
    assert body["tool_calls"][0]["arguments"] == {}


@pytest.mark.asyncio
async def test_create_entities_chat_failure_returns_500(auth_headers):
    """If chat() raises, the endpoint surfaces a 500 with the error message."""
    with patch("app.api.routes.chat", AsyncMock(side_effect=RuntimeError("LLM down"))):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/assistant/create-entities",
                headers=auth_headers,
                json={"prompt": "Buat task"},
            )

    assert resp.status_code == 500
    assert "LLM down" in resp.json()["detail"]
