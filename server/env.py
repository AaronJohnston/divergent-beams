from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    cohere_api_key: str
    model_config = SettingsConfigDict(env_file=".env")


@lru_cache
def get_env():
    return Settings()
