from pydantic import BaseSettings, Field
from functools import lru_cache

class Settings(BaseSettings):
    env: str = Field(default="dev")
    db_url: str 
    redis_url: str
    log_level: str = "INFO"
    secret_key: str = "___"
    
    class Config:
        env_file = ".env"

@lru_cache
def get_settings():
    return Settings()