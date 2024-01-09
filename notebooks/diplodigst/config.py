from typing import Any

from dotenv import find_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class DiploConfig(BaseSettings):
    """
    Represents the settings for the application.
    """

    PROJECT_NAME: str = "DiploDigst"

    OPENAI_API_KEY: str = "sk-<your-key>"

    WEAVIATE_HOST: str = "localhost"
    WEAVIATE_PORT: int = 8080

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    NEO4J_HOST: str = "localhost"
    NEO4J_PORT: int = 7687
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    model_config: Any = SettingsConfigDict(
        env_file=find_dotenv(), env_file_encoding="utf-8", extra="allow"
    )
