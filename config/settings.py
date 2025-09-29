from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Risk Analysis API"
    VERSION: str = "1.0.0"
    
    # Development settings
    DEBUG: bool = True
    TESTING: bool = False
    
    # CORS settings
    ALLOWED_ORIGINS: list = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    class Config:
        env_file = ".env"

settings = Settings()