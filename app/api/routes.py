import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Any

from app.core.ai_client import chat
from app.core.security import verify_service_token
from app.models.action import (
    CreateEntitiesRequest,
    CreateEntitiesResponse,
    ToolCallResult,
)
from app.tools import TOOLS

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(verify_service_token)])

class CaptureRequest(BaseModel):
    raw_input: str

class PriorityRequest(BaseModel):
    tasks: list[dict[str, Any]]
    projects: list[dict[str, Any]]

class PlannerRequest(BaseModel):
    energy_level: str
    mood: str
    tasks: list[dict[str, Any]]
    start_date: str | None = None
    end_date: str | None = None

class AssistantChatRequest(BaseModel):
    prompt: str
    context: dict[str, Any]


SYSTEM_PROMPT = """You are Karsa, a helpful AI assistant for task and project management.
Your job is to understand what the user wants to create and extract structured actions.

You have access to tools that can create tasks, projects, notes, and planner entries.
When the user describes what they want, use the appropriate tools to create those entities.

Guidelines:
- Understand the user's intent from their natural language description
- Use tools to create the entities they want
- Provide clear, concise tool calls with all necessary information
- Respond in the same language the user used (Indonesian or English)
- Be helpful and confirm what you're creating

For tasks:
- Extract a clear, concise title (3-7 words)
- Include description if provided
- Set priority if mentioned (LOW, MEDIUM, HIGH, URGENT), default to MEDIUM
- Include deadline if mentioned (ISO 8601 or relative like "besok", "tomorrow")
- Link to project if mentioned by name

For projects:
- Extract a clear, concise title
- Include description if provided
- Set priority if mentioned, default to MEDIUM
- Include deadline if mentioned

For notes:
- Extract a clear title
- Include content/body if provided
- Link to project if mentioned by name

For planner entries:
- Extract a clear title
- Include date in YYYY-MM-DD format
- Include start and end times in HH:MM 24-hour format
- Link to task if mentioned by name or ID
"""


def get_tools() -> list[dict]:
    """Return the tool definitions the LLM can invoke.

    Exposed as a module-level function (not a constant) so tests can patch it.
    """
    return TOOLS


def _parse_tool_arguments(raw: Any) -> dict:
    """Parse a tool call's `arguments` field into a dict.

    The OpenAI SDK returns `arguments` as a JSON-encoded string, but
    defensive code handles dicts and bad JSON without raising.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            logger.warning("Failed to parse tool_call arguments JSON: %r", raw)
            return {}
    return {}


@router.post("/capture/extract")
async def extract_capture(request: CaptureRequest):
    try:
        data = await chat(
            messages=[
                {"role": "system", "content": "You are a helpful data extraction assistant that outputs only valid JSON."},
                {"role": "user", "content": request.raw_input},
            ],
            response_format={"type": "json_object"},
        )
        return {"success": True, "message": "Extracted successfully", "data": _parse_tool_arguments(data.content)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/priority/suggest")
async def suggest_priority(request: PriorityRequest):
    try:
        data = await chat(
            messages=[
                {"role": "system", "content": "You are a productivity assistant that outputs only valid JSON."},
                {"role": "user", "content": f"Tasks: {json.dumps(request.tasks)}\nProjects: {json.dumps(request.projects)}"},
            ],
            response_format={"type": "json_object"},
        )
        return {"success": True, "message": "Priority suggested successfully", "data": _parse_tool_arguments(data.content)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/planner/generate")
async def generate_planner(request: PlannerRequest):
    try:
        data = await chat(
            messages=[
                {"role": "system", "content": "You are a daily planning assistant that outputs only valid JSON."},
                {"role": "user", "content": f"Energy: {request.energy_level}\nMood: {request.mood}\nTasks: {json.dumps(request.tasks)}"},
            ],
            response_format={"type": "json_object"},
        )
        return {"success": True, "message": "Planner generated successfully", "data": _parse_tool_arguments(data.content)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/assistant/chat")
async def assistant_chat(request: AssistantChatRequest):
    try:
        data = await chat(
            messages=[
                {"role": "system", "content": "You are Karsa, a calm and helpful personal productivity assistant."},
                {"role": "user", "content": f"Context: {json.dumps(request.context)}\n\nUser: {request.prompt}"},
            ],
            response_format={"type": "json_object"},
        )
        return {"success": True, "message": "Chat completed", "data": _parse_tool_arguments(data.content)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/assistant/create-entities", response_model=CreateEntitiesResponse)
async def create_entities(request: CreateEntitiesRequest):
    """Accept a user prompt, call the LLM with function-calling tools, and
    return the assistant's reply plus any structured tool calls the LLM emitted.
    """
    try:
        message = await chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.prompt},
            ],
            tools=get_tools(),
        )

        tool_calls: list[ToolCallResult] = []
        for tc in (message.tool_calls or []):
            tool_calls.append(
                ToolCallResult(
                    name=tc.function.name,
                    arguments=_parse_tool_arguments(tc.function.arguments),
                )
            )

        return CreateEntitiesResponse(
            reply=(message.content or ""),
            tool_calls=tool_calls,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
