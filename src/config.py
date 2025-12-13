from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any, Dict, Optional
from pydantic import PostgresDsn, field_validator, model_validator


class Settings(BaseSettings):
    PROJECT_NAME: str = "API FastAPI"
    LOG_LEVEL: str = "DEBUG"

    OPENAI_API_KEY: str
    OPENAI_MODEL: str
    GEMINI_MODEL: str

    # Database
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    DATABASE_URL: PostgresDsn

    @model_validator(mode="before")
    @classmethod
    def assemble_db_connection(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "DATABASE_URL" not in values:
            db_url = PostgresDsn.build(
                scheme="postgresql+asyncpg",
                username=values.get("POSTGRES_USER"),
                password=values.get("POSTGRES_PASSWORD"),
                host=values.get("POSTGRES_HOST"),
                port=int(values.get("POSTGRES_PORT")),
                path=f"{values.get('POSTGRES_DB') or ''}",
            )
            values["DATABASE_URL"] = str(db_url)
        return values

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def strip_quotes_from_db_url(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip('"')
        return v

    # Firecrawl
    FIRECRAWL_API_KEY: Optional[str] = None

    # Chroma Cloud
    CHROMA_CLOUD_API_KEY: Optional[str] = None
    CHROMA_CLOUD_TENANT: Optional[str] = None
    CHROMA_CLOUD_DATABASE: Optional[str] = None
    CHROMA_CLOUD_COLLECTION: Optional[str] = None


    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
