from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
  DEBUG: bool = Field(default=False, env="DEBUG")
  DATABASE_URL: str = Field(..., env="DATABASE_URL")
  SECRET_KEY: str = Field(..., env="SECRET_KEY")
  API_TIMEOUT: int = Field(default=60, env="API_TIMEOUT")

  OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
  ANTHROPIC_API_KEY: str = Field(default="", env="ANTHROPIC_API_KEY")
  GEMINI_API_KEY: str = Field(..., env="GEMINI_API_KEY")
  GEMINI_MODEL: str = Field(default="gemini-2.5-flash", env="GEMINI_MODEL")
  DEFAULT_MODEL: str = Field(default="gemini-flash-latest", env="DEFAULT_MODEL")
  DEFAULT_TEMPERATURE: float = Field(default=0.7, env="DEFAULT_TEMPERATURE")

  # read from .env file
  model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
