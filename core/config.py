"""
core/config.py - Working Configuration
====================================

Simple configuration that bypasses Pydantic validation issues.
"""

import os
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class Settings:
    """Simple settings class without Pydantic validation"""
    
    def __init__(self):
        # Application settings
        self.app_name = "Risk Analysis Backend"
        self.app_version = "2.0.0"
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # API settings
        self.api_host = "0.0.0.0"
        self.api_port = int(os.getenv("API_PORT", "8000"))
        
        # Legacy compatibility - read but don't validate
        self.project_name = os.getenv("PROJECT_NAME", "Risk Analysis API")
        self.api_v1_str = os.getenv("API_V1_STR", "/api/v1")
        self.alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        
        # FMP Data Provider settings
        self.fmp_api_key = os.getenv("FMP_API_KEY")
        self.fmp_enabled = self.fmp_api_key is not None
        self.fmp_base_url = "https://financialmodelingprep.com/api/v3"
        self.fmp_request_timeout = 30
        
        # Risk analysis settings
        self.risk_free_rate = float(os.getenv("RISK_FREE_RATE", "0.02"))
        self.confidence_levels = [0.95, 0.99]
        self.enable_caching = os.getenv("ENABLE_CACHING", "true").lower() == "true"
        self.cache_ttl_seconds = 3600
        
        # Data settings
        self.default_analysis_period = "1year"
        self.max_symbols_per_request = 50
        self.synthetic_data_seed = 42
        
        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # CORS settings
        self.cors_origins = ["*"]
        
        self._log_configuration()
    
    def _log_configuration(self):
        """Log current configuration (without sensitive data)"""
        config_summary = {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "debug": self.debug,
            "fmp_enabled": self.fmp_enabled,
            "fmp_api_key_set": bool(self.fmp_api_key),
            "alpha_vantage_key_set": bool(self.alpha_vantage_api_key),
            "risk_free_rate": self.risk_free_rate,
            "confidence_levels": self.confidence_levels,
            "enable_caching": self.enable_caching,
            "log_level": self.log_level
        }
        
        logger.info(f"Configuration loaded: {config_summary}")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"
    
    @property
    def database_url(self) -> Optional[str]:
        """Get database URL if configured (for future use)"""
        return os.getenv("DATABASE_URL")

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get application settings"""
    return settings
