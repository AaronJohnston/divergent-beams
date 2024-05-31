from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(BaseSettings):
    cohere_api_key: str
    model_config = SettingsConfigDict(env_file=".env")
