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
