from pydantic import BaseSettings, Field, SecretStr
from functools import lru_cache

class Settings(BaseSettings):
    app_name : str = "Life Coach AI"
    env: str = Field(default="dev")
    db_url: str = Field(default="sqlite:///data/dev.db", env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    log_level: str = "INFO"
    secret_key: SecretStr
    offline_mode: bool = False
    timezone: str = "UTC"
    plan_timeout_s: int = 5
    enable_mfa: bool = True
    telemetry_opt_in: bool = Field(default=False)
    cache_dir: str = "data/cache"
    max_reminders_per_hours: int = 5
    
    class Config:
        env_file = ".env"

@lru_cache
def get_settings():
    return Settings()