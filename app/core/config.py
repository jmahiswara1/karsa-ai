from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    AI_SERVICE_TOKEN: str = "default-service-token"
    OPENAI_API_KEY: str = ""
    SUMOPOD_BASE_URL: str = "https://ai.sumopod.com/v1"
    MODEL_NAME: str = "kimi-k2.6"

    class Config:
        env_file = ".env"

settings = Settings()
