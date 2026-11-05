from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
  DEBUG: bool = Field(default=False)
  DATABASE_URL: str = Field(...)
  SECRET_KEY: str = Field(...)
  API_TIMEOUT: int = Field(default=60)

  OPENAI_API_KEY: str = Field(default="")
  ANTHROPIC_API_KEY: str = Field(default="")
  GEMINI_API_KEY: str = Field(...)
  GEMINI_MODEL: str = Field(default="gemini-2.5-flash")
  DEFAULT_MODEL: str = Field(default="gemini-flash-latest")
  DEFAULT_TEMPERATURE: float = Field(default=0.7)

  # AI Gateway Settings (Lab 02)
  GATEWAY_MAX_INPUT_TOKENS: int = Field(default=2048)
  GATEWAY_PRIMARY_MODEL: str = Field(default="gemini-2.5-flash")
  GATEWAY_FALLBACK_MODELS: str = Field(default="gemini-3.5-flash-lite,gemini-flash-latest")
  GATEWAY_TIMEOUT_SECONDS: float = Field(default=15.0)

  # Semantic Cache & Feature Tagger (Lab 03)
  SEMANTIC_CACHE_THRESHOLD: float = Field(default=0.85)
  SEMANTIC_CACHE_MAX_ENTRIES: int = Field(default=1000)
  EMBEDDING_MODEL: str = Field(default="models/gemini-embedding-001")

  # read from .env file
  model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
