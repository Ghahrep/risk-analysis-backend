# core/dependencies.py - Complete Dependencies for Four-Way Integration
"""
Core Dependencies - Updated for Four-Way Service Integration
===========================================================

Following Backend Refactoring Handbook dependency injection patterns
with comprehensive error handling and graceful degradation.
"""

import logging
from typing import Optional, Any
from functools import lru_cache
import os

logger = logging.getLogger(__name__)

# Global service instances for dependency injection
_risk_service_instance: Optional[Any] = None
_behavioral_service_instance: Optional[Any] = None
_portfolio_service_instance: Optional[Any] = None
_forecasting_service_instance: Optional[Any] = None
_fmp_provider_instance: Optional[Any] = None
_data_manager_instance: Optional[Any] = None

# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

@lru_cache()
def get_settings():
    """Get application settings (cached with fallback)"""
    try:
        from core.config import Settings
        settings = Settings()
        logger.info("Settings loaded successfully")
        return settings
    except Exception as e:
        logger.warning(f"Failed to load settings, using defaults: {e}")
        
        # Return minimal settings for graceful degradation
        class MinimalSettings:
            fmp_enabled = False
            fmp_api_key = None
            risk_free_rate = 0.02
            confidence_levels = [0.95, 0.99]
            enable_caching = False
            
        return MinimalSettings()

# =============================================================================
# FMP PROVIDER DEPENDENCY
# =============================================================================

def get_fmp_provider():
    """Get FMP data provider with error handling"""
    global _fmp_provider_instance
    
    if _fmp_provider_instance is None:
        try:
            settings = get_settings()
            if getattr(settings, 'fmp_enabled', False):
                # Try to import and initialize FMP provider
                from data.providers.fmp_integration import FMPDataProvider
                _fmp_provider_instance = FMPDataProvider()
                logger.info("FMP provider initialized")
            else:
                logger.info("FMP provider disabled in settings")
                return None
        except ImportError:
            logger.warning("FMP provider not available - module not found")
            return None
        except Exception as e:
            logger.warning(f"FMP provider initialization failed: {e}")
            return None
    
    return _fmp_provider_instance

def get_data_manager():
    """Get data manager with FMP integration"""
    global _data_manager_instance
    
    if _data_manager_instance is None:
        try:
            # Try to get FMP provider first
            fmp_provider = get_fmp_provider()
            
            # Create data manager (this might be a simple wrapper or mock)
            class DataManager:
                def __init__(self, fmp_provider=None):
                    self.fmp_provider = fmp_provider
                
                async def get_returns_data(self, symbols, period="1year"):
                    if self.fmp_provider:
                        try:
                            data = await self.fmp_provider.get_historical_data(symbols, period)
                            return data, "FMP Real Data"
                        except Exception as e:
                            logger.warning(f"FMP data retrieval failed: {e}")
                            return None, "Synthetic"
                    else:
                        logger.warning("No FMP provider available, using synthetic data")
                        return None, "Synthetic"
            
            _data_manager_instance = DataManager(fmp_provider)
            logger.info("Data manager initialized")
            
        except Exception as e:
            logger.warning(f"Data manager initialization failed: {e}")
            _data_manager_instance = None
    
    return _data_manager_instance

# =============================================================================
# RISK ANALYSIS SERVICE DEPENDENCY
# =============================================================================

def get_risk_service():
    """Get risk analysis service instance"""
    global _risk_service_instance
    
    if _risk_service_instance is None:
        try:
            from services.risk_service import RiskAnalysisService
            
            settings = get_settings()
            data_manager = get_data_manager()
            
            _risk_service_instance = RiskAnalysisService(
                data_manager=data_manager,
                risk_free_rate=getattr(settings, 'risk_free_rate', 0.02),
                confidence_levels=getattr(settings, 'confidence_levels', [0.95, 0.99]),
                enable_caching=getattr(settings, 'enable_caching', False)
            )
            
            logger.info("Risk Analysis Service initialized successfully")
            
        except ImportError as e:
            logger.error(f"Risk service import failed: {e}")
            raise ImportError("Risk Analysis Service is required but not available")
        except Exception as e:
            logger.error(f"Risk service initialization failed: {e}")
            raise
    
    return _risk_service_instance

# =============================================================================
# BEHAVIORAL ANALYSIS SERVICE DEPENDENCY
# =============================================================================

def get_behavioral_service():
    """Get behavioral analysis service instance"""
    global _behavioral_service_instance
    
    if _behavioral_service_instance is None:
        try:
            from services.behavioral_service_updated import BehavioralAnalysisService
            
            settings = get_settings()
            data_manager = get_data_manager()
            
            _behavioral_service_instance = BehavioralAnalysisService(
                data_manager=data_manager,
                **{k: v for k, v in settings.__dict__.items() if not k.startswith('_')}
            )
            
            logger.info("Behavioral Analysis Service initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Behavioral service not available: {e}")
            return None
        except Exception as e:
            logger.warning(f"Behavioral service initialization failed: {e}")
            return None
    
    return _behavioral_service_instance

# =============================================================================
# PORTFOLIO MANAGEMENT SERVICE DEPENDENCY
# =============================================================================

def get_portfolio_service():
    """Get portfolio management service instance"""
    global _portfolio_service_instance
    
    if _portfolio_service_instance is None:
        try:
            from services.portfolio_service_direct import PortfolioManagementService
            
            settings = get_settings()
            data_manager = get_data_manager()
            
            _portfolio_service_instance = PortfolioManagementService(
                data_manager=data_manager,
                **{k: v for k, v in settings.__dict__.items() if not k.startswith('_')}
            )
            
            logger.info("Portfolio Management Service initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Portfolio service not available: {e}")
            return None
        except Exception as e:
            logger.warning(f"Portfolio service initialization failed: {e}")
            return None
    
    return _portfolio_service_instance

# =============================================================================
# FORECASTING SERVICE DEPENDENCY
# =============================================================================

def get_forecasting_service():
    """Get forecasting service instance"""
    global _forecasting_service_instance
    
    if _forecasting_service_instance is None:
        try:
            from services.forecasting_service_updated import ForecastingService
            
            settings = get_settings()
            data_manager = get_data_manager()
            
            _forecasting_service_instance = ForecastingService(
                data_manager=data_manager,
                **{k: v for k, v in settings.__dict__.items() if not k.startswith('_')}
            )
            
            logger.info("Forecasting Service initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Forecasting service not available: {e}")
            return None
        except Exception as e:
            logger.warning(f"Forecasting service initialization failed: {e}")
            return None
    
    return _forecasting_service_instance

# =============================================================================
# FASTAPI DEPENDENCY PROVIDERS
# =============================================================================

async def get_risk_service_dependency():
    """FastAPI dependency provider for risk service"""
    return get_risk_service()

async def get_behavioral_service_dependency():
    """FastAPI dependency provider for behavioral service"""
    service = get_behavioral_service()
    if service is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Behavioral Analysis Service not available")
    return service

async def get_portfolio_service_dependency():
    """FastAPI dependency provider for portfolio service"""
    service = get_portfolio_service()
    if service is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Portfolio Management Service not available")
    return service

async def get_forecasting_service_dependency():
    """FastAPI dependency provider for forecasting service"""
    service = get_forecasting_service()
    if service is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Forecasting Service not available")
    return service

async def get_data_manager_dependency():
    """FastAPI dependency provider for data manager"""
    return get_data_manager()

async def get_fmp_provider_dependency():
    """FastAPI dependency provider for FMP provider"""
    return get_fmp_provider()

# =============================================================================
# SERVICE HEALTH CHECK UTILITIES
# =============================================================================

def check_service_availability() -> dict:
    """Check availability of all services"""
    services_status = {}
    
    # Risk service (critical)
    try:
        risk_service = get_risk_service()
        services_status["risk"] = "available" if risk_service else "unavailable"
    except Exception as e:
        services_status["risk"] = f"error: {str(e)}"
    
    # Behavioral service (optional)
    try:
        behavioral_service = get_behavioral_service()
        services_status["behavioral"] = "available" if behavioral_service else "unavailable"
    except Exception as e:
        services_status["behavioral"] = f"error: {str(e)}"
    
    # Portfolio service (optional)
    try:
        portfolio_service = get_portfolio_service()
        services_status["portfolio"] = "available" if portfolio_service else "unavailable"
    except Exception as e:
        services_status["portfolio"] = f"error: {str(e)}"
    
    # Forecasting service (optional)
    try:
        forecasting_service = get_forecasting_service()
        services_status["forecasting"] = "available" if forecasting_service else "unavailable"
    except Exception as e:
        services_status["forecasting"] = f"error: {str(e)}"
    
    # FMP provider
    try:
        fmp_provider = get_fmp_provider()
        services_status["fmp_provider"] = "available" if fmp_provider else "unavailable"
    except Exception as e:
        services_status["fmp_provider"] = f"error: {str(e)}"
    
    return services_status

# =============================================================================
# CLEANUP UTILITIES
# =============================================================================

def reset_dependencies():
    """Reset all dependency instances for testing or reinitialization"""
    global _risk_service_instance, _behavioral_service_instance
    global _portfolio_service_instance, _forecasting_service_instance
    global _fmp_provider_instance, _data_manager_instance
    
    _risk_service_instance = None
    _behavioral_service_instance = None
    _portfolio_service_instance = None
    _forecasting_service_instance = None
    _fmp_provider_instance = None
    _data_manager_instance = None
    
    logger.info("All dependency instances reset")

def cleanup_resources():
    """Cleanup resources on application shutdown"""
    try:
        # Close any open connections, clear caches, etc.
        reset_dependencies()
        logger.info("Resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during resource cleanup: {e}")

# =============================================================================
# DIAGNOSTIC UTILITIES
# =============================================================================

def get_dependency_status():
    """Get detailed status of all dependencies for diagnostics"""
    status = {
        "timestamp": "now",
        "services": {},
        "providers": {},
        "configuration": {}
    }
    
    # Check services
    for service_name in ["risk", "behavioral", "portfolio", "forecasting"]:
        try:
            if service_name == "risk":
                service = get_risk_service()
            elif service_name == "behavioral":
                service = get_behavioral_service()
            elif service_name == "portfolio":
                service = get_portfolio_service()
            elif service_name == "forecasting":
                service = get_forecasting_service()
            
            status["services"][service_name] = {
                "available": service is not None,
                "type": type(service).__name__ if service else None
            }
        except Exception as e:
            status["services"][service_name] = {
                "available": False,
                "error": str(e)
            }
    
    # Check providers
    try:
        fmp_provider = get_fmp_provider()
        status["providers"]["fmp"] = {
            "available": fmp_provider is not None,
            "type": type(fmp_provider).__name__ if fmp_provider else None
        }
    except Exception as e:
        status["providers"]["fmp"] = {
            "available": False,
            "error": str(e)
        }
    
    # Check configuration
    try:
        settings = get_settings()
        status["configuration"] = {
            "loaded": True,
            "fmp_enabled": getattr(settings, 'fmp_enabled', False),
            "type": type(settings).__name__
        }
    except Exception as e:
        status["configuration"] = {
            "loaded": False,
            "error": str(e)
        }
    
    return status

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Core dependency functions
    "get_risk_service",
    "get_behavioral_service", 
    "get_portfolio_service",
    "get_forecasting_service",
    "get_fmp_provider",
    "get_data_manager",
    "get_settings",
    
    # FastAPI dependency providers
    "get_risk_service_dependency",
    "get_behavioral_service_dependency",
    "get_portfolio_service_dependency", 
    "get_forecasting_service_dependency",
    "get_data_manager_dependency",
    "get_fmp_provider_dependency",
    
    # Utilities
    "check_service_availability",
    "reset_dependencies",
    "cleanup_resources",
    "get_dependency_status"
]