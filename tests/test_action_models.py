"""Tests for app.models.action Pydantic models."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.action import (
    CreateEntitiesRequest,
    ToolCallResult,
    CreateEntitiesResponse,
)


# ── CreateEntitiesRequest ───────────────────────────────


class TestCreateEntitiesRequest:
    def test_parses_prompt_and_context(self):
        req = CreateEntitiesRequest(
            prompt="Buat task design homepage",
            context={"projects": [], "folders": []},
        )
        assert req.prompt == "Buat task design homepage"
        assert req.context == {"projects": [], "folders": []}

    def test_context_defaults_to_empty_dict(self):
        req = CreateEntitiesRequest(prompt="Hello")
        assert req.context == {}
        assert isinstance(req.context, dict)

    def test_context_accepts_nested_data(self):
        ctx = {
            "projects": [{"id": "p1", "name": "Karsa"}],
            "folders": [{"id": "f1", "name": "Inbox"}],
            "recentTasks": [{"id": "t1", "title": "Old task"}],
        }
        req = CreateEntitiesRequest(prompt="Test", context=ctx)
        assert req.context == ctx

    def test_empty_prompt_raises_validation_error(self):
        with pytest.raises(ValidationError):
            CreateEntitiesRequest(prompt="")

    def test_missing_prompt_raises_validation_error(self):
        with pytest.raises(ValidationError):
            CreateEntitiesRequest()  # type: ignore[call-arg]

    def test_non_string_prompt_raises_validation_error(self):
        with pytest.raises(ValidationError):
            CreateEntitiesRequest(prompt=123)  # type: ignore[arg-type]

    def test_whitespace_only_prompt_accepted_at_model_level(self):
        # min_length=1 allows whitespace; trimming is not enforced here.
        req = CreateEntitiesRequest(prompt="   ")
        assert req.prompt == "   "


# ── ToolCallResult ──────────────────────────────────────


class TestToolCallResult:
    def test_parses_name_and_arguments(self):
        tc = ToolCallResult(
            name="create_task",
            arguments={"title": "Test task"},
        )
        assert tc.name == "create_task"
        assert tc.arguments == {"title": "Test task"}

    def test_arguments_defaults_to_empty_dict(self):
        tc = ToolCallResult(name="create_task")
        assert tc.arguments == {}
        assert isinstance(tc.arguments, dict)

    def test_arguments_accepts_complex_nested_structures(self):
        args = {
            "title": "Task",
            "dueDate": "2026-06-25",
            "projectId": "p1",
            "tags": ["work", "urgent"],
            "metadata": {"source": "user", "priority": 1},
        }
        tc = ToolCallResult(name="create_task", arguments=args)
        assert tc.arguments == args

    def test_missing_name_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ToolCallResult()  # type: ignore[call-arg]

    def test_empty_name_accepted_at_model_level(self):
        # name is required but has no min_length constraint.
        tc = ToolCallResult(name="")
        assert tc.name == ""


# ── CreateEntitiesResponse ──────────────────────────────


class TestCreateEntitiesResponse:
    def test_includes_reply_and_tool_calls(self):
        resp = CreateEntitiesResponse(
            reply="Saya akan membuat task.",
            tool_calls=[
                ToolCallResult(name="create_task", arguments={"title": "X"}),
            ],
        )
        assert resp.reply == "Saya akan membuat task."
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "create_task"
        assert resp.tool_calls[0].arguments == {"title": "X"}

    def test_handles_empty_tool_calls_list(self):
        resp = CreateEntitiesResponse(reply="Done", tool_calls=[])
        assert resp.reply == "Done"
        assert resp.tool_calls == []

    def test_default_reply_is_empty_string(self):
        resp = CreateEntitiesResponse()
        assert resp.reply == ""
        assert isinstance(resp.reply, str)

    def test_default_tool_calls_is_empty_list(self):
        resp = CreateEntitiesResponse()
        assert resp.tool_calls == []
        assert isinstance(resp.tool_calls, list)

    def test_all_defaults_construct_empty_response(self):
        resp = CreateEntitiesResponse()
        assert resp.reply == ""
        assert resp.tool_calls == []
        assert resp.model_dump() == {"reply": "", "tool_calls": []}

    def test_multiple_tool_calls_preserved_in_order(self):
        calls = [
            ToolCallResult(name="create_task", arguments={"title": "A"}),
            ToolCallResult(name="create_folder", arguments={"name": "B"}),
            ToolCallResult(name="assign_task_to_folder", arguments={"taskId": "t1", "folderId": "f1"}),
        ]
        resp = CreateEntitiesResponse(reply="Multiple", tool_calls=calls)
        assert len(resp.tool_calls) == 3
        assert [c.name for c in resp.tool_calls] == [
            "create_task",
            "create_folder",
            "assign_task_to_folder",
        ]

    def test_tool_calls_default_factory_produces_independent_lists(self):
        # Regression: default_factory must give each instance its own list,
        # not a shared mutable default.
        r1 = CreateEntitiesResponse()
        r2 = CreateEntitiesResponse()
        r1.tool_calls.append(ToolCallResult(name="x"))
        assert r2.tool_calls == []

    def test_context_default_factory_produces_independent_dicts(self):
        r1 = CreateEntitiesRequest(prompt="a")
        r2 = CreateEntitiesRequest(prompt="b")
        r1.context["k"] = "v"
        assert r2.context == {}
