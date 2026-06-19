"""Tests for the OpenAI function-calling tool definitions in app.tools."""
from __future__ import annotations

import pytest

from app.tools import TOOLS


# ── Constants ──────────────────────────────────────────────

EXPECTED_TOOL_NAMES = {
    "create_task",
    "create_project",
    "create_note",
    "create_planner_entry",
}

VALID_PRIORITIES = {"LOW", "MEDIUM", "HIGH", "URGENT"}


# ── Helpers ────────────────────────────────────────────────


def _by_name(name: str) -> dict:
    """Return the raw tool dict for the given function name, or fail loudly."""
    for tool in TOOLS:
        if tool["function"]["name"] == name:
            return tool
    raise AssertionError(f"Tool {name!r} not found in TOOLS")


# ── Top-level structure ────────────────────────────────────


def test_tools_is_a_list():
    assert isinstance(TOOLS, list)
    assert len(TOOLS) > 0


def test_all_four_tools_are_defined():
    defined_names = {tool["function"]["name"] for tool in TOOLS}
    assert defined_names == EXPECTED_TOOL_NAMES, (
        f"Expected {EXPECTED_TOOL_NAMES}, got {defined_names}"
    )


def test_each_tool_has_openai_function_shape():
    """Every entry must follow OpenAI's documented tool-call shape."""
    expected_shape = "{type, function: {name, description, parameters}}"
    for tool in TOOLS:
        assert isinstance(tool, dict)
        assert "type" in tool, f"Tool missing 'type': {tool}"
        assert tool["type"] == "function", (
            f"Tool type must be 'function', got {tool['type']!r} "
            f"(expected {expected_shape})"
        )

        fn = tool["function"]
        assert isinstance(fn, dict)
        assert "name" in fn and isinstance(fn["name"], str) and fn["name"]
        assert (
            "description" in fn
            and isinstance(fn["description"], str)
            and fn["description"]
        )
        assert "parameters" in fn and isinstance(fn["parameters"], dict)


def test_tool_names_are_unique():
    names = [tool["function"]["name"] for tool in TOOLS]
    assert len(names) == len(set(names)), f"Duplicate tool names found: {names}"


# ── create_task ────────────────────────────────────────────


class TestCreateTask:
    @pytest.fixture
    def tool(self):
        return _by_name("create_task")

    def test_required_fields(self, tool):
        params = tool["function"]["parameters"]
        assert params["type"] == "object"
        assert params["required"] == ["title"]

    def test_title_is_required_string(self, tool):
        props = tool["function"]["parameters"]["properties"]
        assert "title" in props
        assert props["title"]["type"] == "string"

    def test_description_is_optional_string(self, tool):
        props = tool["function"]["parameters"]["properties"]
        assert "description" in props
        assert props["description"]["type"] == "string"

    def test_priority_enum_and_default(self, tool):
        props = tool["function"]["parameters"]["properties"]
        priority = props["priority"]
        assert priority["type"] == "string"
        assert set(priority["enum"]) == VALID_PRIORITIES
        assert priority["default"] == "MEDIUM"

    def test_deadline_is_string(self, tool):
        props = tool["function"]["parameters"]["properties"]
        assert props["deadline"]["type"] == "string"

    def test_project_name_is_string(self, tool):
        props = tool["function"]["parameters"]["properties"]
        assert props["projectName"]["type"] == "string"

    def test_tags_is_array_of_strings(self, tool):
        props = tool["function"]["parameters"]["properties"]
        tags = props["tags"]
        assert tags["type"] == "array"
        assert tags["items"] == {"type": "string"}

    def test_no_unexpected_required_fields(self, tool):
        params = tool["function"]["parameters"]
        # Only `title` should be in `required` — the rest are optional.
        for field in params["required"]:
            assert field in params["properties"], (
                f"Required field {field!r} missing from properties"
            )


# ── create_project ─────────────────────────────────────────


class TestCreateProject:
    @pytest.fixture
    def tool(self):
        return _by_name("create_project")

    def test_required_fields(self, tool):
        params = tool["function"]["parameters"]
        assert params["required"] == ["title"]

    def test_title_is_required_string(self, tool):
        props = tool["function"]["parameters"]["properties"]
        assert props["title"]["type"] == "string"

    def test_description_is_optional_string(self, tool):
        props = tool["function"]["parameters"]["properties"]
        assert props["description"]["type"] == "string"

    def test_priority_enum_and_default(self, tool):
        props = tool["function"]["parameters"]["properties"]
        priority = props["priority"]
        assert priority["type"] == "string"
        assert set(priority["enum"]) == VALID_PRIORITIES
        assert priority["default"] == "MEDIUM"

    def test_deadline_is_string(self, tool):
        props = tool["function"]["parameters"]["properties"]
        assert props["deadline"]["type"] == "string"

    def test_no_project_name_or_tags_for_project_tool(self, tool):
        """Projects don't carry tags or a parent project link."""
        props = tool["function"]["parameters"]["properties"]
        assert "projectName" not in props
        assert "tags" not in props


# ── create_note ────────────────────────────────────────────


class TestCreateNote:
    @pytest.fixture
    def tool(self):
        return _by_name("create_note")

    def test_required_fields(self, tool):
        params = tool["function"]["parameters"]
        assert set(params["required"]) == {"title", "content"}

    def test_title_is_required_string(self, tool):
        props = tool["function"]["parameters"]["properties"]
        assert props["title"]["type"] == "string"

    def test_content_is_required_string(self, tool):
        props = tool["function"]["parameters"]["properties"]
        assert props["content"]["type"] == "string"

    def test_project_name_is_optional_string(self, tool):
        props = tool["function"]["parameters"]["properties"]
        assert props["projectName"]["type"] == "string"

    def test_folder_name_is_optional_string(self, tool):
        props = tool["function"]["parameters"]["properties"]
        assert props["folderName"]["type"] == "string"

    def test_no_priority_or_deadline_for_note(self, tool):
        """Notes are informational, not scheduled/prioritized."""
        props = tool["function"]["parameters"]["properties"]
        assert "priority" not in props
        assert "deadline" not in props


# ── create_planner_entry ───────────────────────────────────


class TestCreatePlannerEntry:
    @pytest.fixture
    def tool(self):
        return _by_name("create_planner_entry")

    def test_required_fields(self, tool):
        params = tool["function"]["parameters"]
        assert set(params["required"]) == {"title", "date", "startTime", "endTime"}

    def test_title_is_required_string(self, tool):
        props = tool["function"]["parameters"]["properties"]
        assert props["title"]["type"] == "string"

    def test_date_is_string(self, tool):
        props = tool["function"]["parameters"]["properties"]
        assert props["date"]["type"] == "string"

    def test_start_time_is_string(self, tool):
        props = tool["function"]["parameters"]["properties"]
        assert props["startTime"]["type"] == "string"

    def test_end_time_is_string(self, tool):
        props = tool["function"]["parameters"]["properties"]
        assert props["endTime"]["type"] == "string"

    def test_task_id_is_optional_string(self, tool):
        props = tool["function"]["parameters"]["properties"]
        assert props["taskId"]["type"] == "string"

    def test_description_is_optional_string(self, tool):
        props = tool["function"]["parameters"]["properties"]
        assert props["description"]["type"] == "string"

    def test_no_priority_tags_or_project_name(self, tool):
        """Planner entries don't carry priority/tags/project link fields."""
        props = tool["function"]["parameters"]["properties"]
        assert "priority" not in props
        assert "tags" not in props
        assert "projectName" not in props


# ── Cross-tool sanity checks ───────────────────────────────


def test_every_required_field_is_typed_string():
    """All required fields across all tools should be typed `string`."""
    for tool in TOOLS:
        params = tool["function"]["parameters"]
        props = params["properties"]
        for field in params["required"]:
            assert field in props, (
                f"{tool['function']['name']}: required field {field!r} "
                f"missing from properties"
            )
            assert props[field]["type"] == "string", (
                f"{tool['function']['name']}.{field} should be 'string', "
                f"got {props[field]['type']!r}"
            )


def test_every_tool_has_a_non_empty_description():
    for tool in TOOLS:
        desc = tool["function"]["description"]
        assert desc.strip(), f"{tool['function']['name']} has empty description"
