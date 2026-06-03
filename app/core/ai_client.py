import json
import re
from openai import AsyncOpenAI
from app.core.config import settings

client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.SUMOPOD_BASE_URL
)

async def get_chat_completion(messages: list, response_format=None):
    kwargs = {
        "model": settings.MODEL_NAME,
        "messages": messages,
    }
    if response_format:
        kwargs["response_format"] = response_format

    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message

def parse_json_response(content: str) -> dict:
    # Try to find JSON block if the model wrapped it in markdown
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    if match:
        content = match.group(1)
    
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse AI response as JSON: {e}\nRaw content: {content}")

async def extract_task_from_text(raw_input: str) -> dict:
    prompt = f"""
    You are an AI assistant for a productivity app. Your task is to extract task information from the following raw text input.
    Extract the 'title', 'description' (if any), 'priority' (enum: LOW, MEDIUM, HIGH, URGENT - default to MEDIUM if not specified), and 'tags' (array of strings).
    
    Raw input: "{raw_input}"
    
    Respond STRICTLY in JSON format with the following keys:
    {{
        "title": "string",
        "description": "string or null",
        "priority": "MEDIUM",
        "tags": ["tag1"]
    }}
    Do not include any other text, only the JSON block.
    """
    
    messages = [
        {"role": "system", "content": "You are a helpful data extraction assistant that outputs only valid JSON."},
        {"role": "user", "content": prompt}
    ]
    
    response = await get_chat_completion(messages, response_format={"type": "json_object"})
    return parse_json_response(response.content)

async def suggest_priorities(tasks: list[dict], projects: list[dict]) -> dict:
    prompt = f"""
    You are an AI priority advisor. Analyze the following tasks and projects, and suggest the top 3 tasks to focus on right now.
    
    Tasks: {json.dumps(tasks)}
    Projects: {json.dumps(projects)}
    
    Respond STRICTLY in JSON format with the following keys:
    {{
        "suggested_task_ids": ["id1", "id2", "id3"],
        "reasoning": "A brief paragraph explaining why these tasks were chosen."
    }}
    Do not include any other text.
    """
    
    messages = [
        {"role": "system", "content": "You are a productivity assistant that outputs only valid JSON."},
        {"role": "user", "content": prompt}
    ]
    
    response = await get_chat_completion(messages, response_format={"type": "json_object"})
    return parse_json_response(response.content)

async def generate_daily_plan(energy_level: str, mood: str, tasks: list[dict]) -> dict:
    prompt = f"""
    You are an AI daily planner. Create a daily plan based on the user's current state.
    Energy Level: {energy_level}
    Mood: {mood}
    Available Tasks: {json.dumps(tasks)}
    
    Select a realistic set of tasks from the available list that fits their current energy and mood.
    If energy is low, suggest easier tasks.
    
    Respond STRICTLY in JSON format with the following keys:
    {{
        "focus_message": "A short, encouraging message tailored to their mood.",
        "planned_task_ids": ["id1", "id2"],
        "estimated_total_minutes": 120
    }}
    Do not include any other text.
    """
    
    messages = [
        {"role": "system", "content": "You are a daily planning assistant that outputs only valid JSON."},
        {"role": "user", "content": prompt}
    ]
    
    response = await get_chat_completion(messages, response_format={"type": "json_object"})
    return parse_json_response(response.content)
