"""Pydantic models for AI assistant create-entities endpoint."""
from pydantic import BaseModel, Field


class CreateEntitiesRequest(BaseModel):
    """Request body for POST /assistant/create-entities."""

    prompt: str = Field(..., min_length=1, description="User prompt in natural language")
    context: dict = Field(
        default_factory=dict,
        description="User context: projects, folders, recentTasks for resolution",
    )


class ToolCallResult(BaseModel):
    """Single tool call from LLM."""

    name: str = Field(..., description="Tool function name (e.g. create_task)")
    arguments: dict = Field(
        default_factory=dict,
        description="Arguments dict parsed from JSON string LLM",
    )


class CreateEntitiesResponse(BaseModel):
    """Response from AI service to Backend."""

    reply: str = Field(default="", description="Natural language reply from assistant")
    tool_calls: list[ToolCallResult] = Field(
        default_factory=list,
        description="List of tool calls LLM wants to execute",
    )
