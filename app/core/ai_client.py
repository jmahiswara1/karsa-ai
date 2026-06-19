"""Async LLM client dengan auto-fallback antar provider (sumopod / b.ai / deepseek)."""
import json
import re

from openai import (
    APIConnectionError,
    APIError,
    AsyncOpenAI,
    OpenAIError,
    RateLimitError,
)

from app.core.config import settings

# Errors that justify moving on to the next provider in the fallback chain.
RETRIABLE_ERRORS = (
    APIConnectionError,
    RateLimitError,
    APIError,
    TimeoutError,
    ConnectionError,
)


async def chat(
    messages: list[dict],
    *,
    provider: str | None = None,
    fallback: list[str] | None = None,
    response_format: dict | None = None,
    stream: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    tools: list[dict] | None = None,
    **kwargs,
):
    """Kirim chat ke LLM dengan auto-fallback.

    Args:
        messages:        list pesan OpenAI format
        provider:        provider utama (default = ACTIVE_PROVIDER dari .env)
        fallback:        override urutan fallback (default = FALLBACK_ORDER)
        response_format: mis. {"type": "json_object"} untuk mode JSON
        stream:          True untuk stream response
        temperature,
        max_tokens,
        tools:           list tool definitions (OpenAI function-calling format)
        **kwargs:        parameter lain yang diteruskan ke API

    Returns:
        - non-stream: `response.choices[0].message` (object dengan atribut `.content` dan `.tool_calls`)
        - stream: async iterator dari chunks

    Raises:
        ValueError: kalau nama provider tak dikenal atau API key kosong
        RuntimeError: kalau semua provider di chain gagal (network/rate limit)
    """
    primary = (provider or settings.ACTIVE_PROVIDER).lower()
    fb_list = fallback if fallback is not None else settings.FALLBACK_ORDER
    chain = [primary] + [p for p in fb_list if p.lower() != primary]

    last_error: Exception | None = None

    for p in chain:
        cfg = settings.provider(p)

        # Build model list: primary model + optional fallback model
        models = [cfg["model"]]
        if cfg.get("fallback_model"):
            fb_model = cfg["fallback_model"]
            if fb_model != cfg["model"]:
                models.append(fb_model)

        for i, model in enumerate(models):
            try:
                if not cfg["api_key"]:
                    print(f"  ✗ [{p}] API key belum di-set, lewati.")
                    last_error = ValueError(f"API key untuk provider '{p}' kosong")
                    break  # break inner loop, continue to next provider

                client = AsyncOpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
                tag = f"{p}" if i == 0 else f"{p}/{model}"
                print(f"  → [{tag}] {cfg['name']} ({model})")

                call_kwargs: dict = {
                    "model": model,
                    "messages": messages,
                    "stream": stream,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **kwargs,
                }
                if response_format:
                    call_kwargs["response_format"] = response_format
                if tools:
                    call_kwargs["tools"] = tools
                    call_kwargs["tool_choice"] = "auto"

                response = await client.chat.completions.create(**call_kwargs)

                if stream:
                    return response
                return response.choices[0].message

            except ValueError as e:
                raise

            except RETRIABLE_ERRORS as e:
                print(f"  ✗ [{tag}] gagal: {type(e).__name__}: {e}")
                last_error = e
                continue  # try next model in the list

            except OpenAIError as e:
                # Auth error (401/403): treat as retriable across the chain
                error_str = str(e).lower()
                auth_patterns = settings._get_auth_patterns()
                if any(pattern in error_str for pattern in auth_patterns):
                    print(f"  ✗ [{tag}] auth error, coba provider/model lain: {e}")
                    last_error = e
                    break  # skip all models for this provider, try next provider
                # Other OpenAI error (bad request, etc). Fatal.
                print(f"  ✗ [{tag}] OpenAI error (fatal): {e}")
                raise

    raise RuntimeError(
        f"Semua provider gagal. Terakhir: {type(last_error).__name__}: {last_error}"
    ) from last_error


# ── JSON helper (dipakai semua endpoint) ─────────────────────

def parse_json_response(content: str) -> dict:
    """Parse JSON dari response LLM. Tangani ```json ... ``` wrapper."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if match:
        content = match.group(1)
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse AI response as JSON: {e}\nRaw content: {content}"
        )


# ── Endpoint helpers (signature tidak berubah) ────────────────

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
        {"role": "user", "content": prompt},
    ]

    message = await chat(
        messages,
        response_format={"type": "json_object"},
    )
    return parse_json_response(message.content)


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
        {"role": "user", "content": prompt},
    ]

    message = await chat(
        messages,
        response_format={"type": "json_object"},
    )
    return parse_json_response(message.content)


async def generate_plan(
    energy_level: str,
    mood: str,
    tasks: list[dict],
    start_date: str = None,
    end_date: str = None,
) -> dict:
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
        {"role": "user", "content": prompt},
    ]

    message = await chat(
        messages,
        response_format={"type": "json_object"},
    )
    return parse_json_response(message.content)


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
        {"role": "user", "content": f"Context: {json.dumps(context)}\n\nUser: {prompt}"},
    ]

    message = await chat(
        messages,
        response_format={"type": "json_object"},
    )
    return parse_json_response(message.content)
