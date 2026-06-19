from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    APP_ENV: str = Field(
        "development", description="Environment: development or production"
    )
    APP_PORT: int = Field(8000, description="Port the application runs on")
    DEBUG: bool = Field(False, description="Enable debug mode")

    # Database
    DATABASE_URL: str = Field(
        "sqlite+aiosqlite:///reporag.db",
        description="PostgreSQL or SQLite connection string",
    )

    # Neo4j
    NEO4J_URI: str = Field(..., description="Neo4j connection URI")
    NEO4J_USER: str = Field(..., description="Neo4j username")
    NEO4J_PASSWORD: SecretStr = Field(..., description="Neo4j password")

    # Qdrant
    QDRANT_HOST: str = Field("localhost", description="Qdrant host")
    QDRANT_PORT: int = Field(6333, description="Qdrant port")
    QDRANT_API_KEY: SecretStr | None = Field(
        None, description="Qdrant API key if secured"
    )

    # LLM
    OPENAI_API_KEY: SecretStr | None = Field(None, description="OpenAI API key")
    ANTHROPIC_API_KEY: SecretStr | None = Field(None, description="Anthropic API key")

    # Auth
    JWT_SECRET: SecretStr = Field(..., description="Secret key for signing JWTs")
    JWT_ALGORITHM: str = Field("HS256", description="JWT signing algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        1440, description="Token expiration time in minutes"
    )

    # OAuth
    GOOGLE_CLIENT_ID: str | None = Field(None, description="Google OAuth Client ID")
    GOOGLE_CLIENT_SECRET: SecretStr | None = Field(
        None, description="Google OAuth Client Secret"
    )


settings = Settings()
