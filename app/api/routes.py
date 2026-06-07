from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Any
from app.core.security import verify_service_token
import app.core.ai_client as ai_client

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

@router.post("/capture/extract")
async def extract_capture(request: CaptureRequest):
    try:
        data = await ai_client.extract_task_from_text(request.raw_input)
        return {"success": True, "message": "Extracted successfully", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/priority/suggest")
async def suggest_priority(request: PriorityRequest):
    try:
        data = await ai_client.suggest_priorities(request.tasks, request.projects)
        return {"success": True, "message": "Priority suggested successfully", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/planner/generate")
async def generate_planner(request: PlannerRequest):
    try:
        data = await ai_client.generate_plan(request.energy_level, request.mood, request.tasks, request.start_date, request.end_date)
        return {"success": True, "message": "Planner generated successfully", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/assistant/chat")
async def assistant_chat(request: AssistantChatRequest):
    try:
        data = await ai_client.assistant_chat(request.prompt, request.context)
        return {"success": True, "message": "Chat completed", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
