from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Any
from app.core.security import verify_service_token

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

@router.post("/capture/extract")
async def extract_capture(request: CaptureRequest):
    # Placeholder for Phase 3
    return {"success": True, "message": "Extracted", "data": {}}

@router.post("/priority/suggest")
async def suggest_priority(request: PriorityRequest):
    # Placeholder for Phase 3
    return {"success": True, "message": "Priority suggested", "data": {}}

@router.post("/planner/generate")
async def generate_planner(request: PlannerRequest):
    # Placeholder for Phase 3
    return {"success": True, "message": "Planner generated", "data": {}}
