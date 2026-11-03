from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
  DEBUG: bool = False
  DATABASE_URL: str
  SECRET_KEY: str
  API_TIMEOUT: int = 60

  # Tự động đọc từ file .env
  model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
