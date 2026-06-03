import asyncio
from app.core.ai_client import client, settings

async def main():
    print("Testing connection to Sumopod...")
    print(f"Base URL: {settings.SUMOPOD_BASE_URL}")
    print(f"Model: {settings.MODEL_NAME}")
    
    try:
        response = await client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, simply reply with 'PONG' and nothing else."}
            ]
        )
        print("\nConnection Successful!")
        print("Response:", response.choices[0].message.content)
    except Exception as e:
        print("\nConnection Failed!")
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
