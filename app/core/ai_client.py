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

async def generate_plan(energy_level: str, mood: str, tasks: list[dict], start_date: str = None, end_date: str = None) -> dict:
    date_context = "Create a time-blocked daily schedule for today."
    block_format = '{"task_id": "task-uuid", "title": "Task name", "start_time": "08:00", "end_time": "09:00", "reason": "Short motivation"}'
    
    if start_date and end_date:
        date_context = f"Create a time-blocked schedule spanning from {start_date} to {end_date}."
        block_format = '{"date": "YYYY-MM-DD", "task_id": "task-uuid", "title": "Task name", "start_time": "08:00", "end_time": "09:00", "reason": "Short motivation"}'

    prompt = f"""
    You are an AI planner for a calm productivity app. {date_context}
    Energy Level: {energy_level}
    Mood: {mood}
    Available Tasks: {json.dumps(tasks)}

    Create a realistic schedule with time blocks starting from a reasonable morning hour.
    - Low energy: start later (9:00), shorter blocks (30-45 min), more breaks.
    - Medium energy: start at 8:00, normal blocks (45-60 min).
    - High energy: start at 7:00, longer blocks (60-90 min), tackle hardest tasks first.
    - Include short breaks between focus sessions.
    - Every task given must appear in a block with its task_id.
    - End by a reasonable time based on energy and total task load.
    - Include a calm, encouraging focus_message.
    - Set workload_level: LOW (3 or fewer blocks per day), MEDIUM (4-6 blocks per day), HIGH (7+ blocks per day).

    Respond STRICTLY in JSON format:
    {{
        "focus_message": "A warm, encouraging sentence tailored to their mood and energy.",
        "blocks": [
            {block_format},
            {block_format.replace('"task-uuid"', 'null').replace('Task name', 'Short Break')}
        ],
        "workload_level": "MEDIUM"
    }}
    Do not include any other text.
    """

    messages = [
        {"role": "system", "content": "You are a daily planning assistant that outputs only valid JSON."},
        {"role": "user", "content": prompt}
    ]

    response = await get_chat_completion(messages, response_format={"type": "json_object"})
    return parse_json_response(response.content)

async def assistant_chat(prompt: str, context: dict) -> dict:
    system_prompt = """
    You are Karsa, a calm and helpful personal productivity assistant.
    You communicate naturally with the user, but you also have the ability to structure tasks or suggest priorities if asked.
    
    You will receive the user's prompt and a context containing their current active tasks and projects.
    
    If the user asks to prioritize tasks, you must include 'action': 'PRIORITIZE' and fill 'action_data' with suggested task IDs.
    If the user asks to create a task or dumps raw thoughts, you must include 'action': 'EXTRACT_TASK' and fill 'action_data' with extracted task details.
    Otherwise, just respond naturally and set 'action' to null.
    
    Respond STRICTLY in JSON format with the following keys:
    {
        "reply": "Your natural language response to the user.",
        "action": "PRIORITIZE | EXTRACT_TASK | null",
        "action_data": {} // Context-specific data or null
    }
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context: {json.dumps(context)}\n\nUser: {prompt}"}
    ]
    
    response = await get_chat_completion(messages, response_format={"type": "json_object"})
    return parse_json_response(response.content)
