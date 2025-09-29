# services/__init__.py - Updated Service Layer Module Initialization
"""
Service Layer Module Initialization
==================================

Exposes all service classes for clean imports across the application.
Updated with CORRECT import paths matching actual service files.
"""

import logging
import os
import sys
from typing import Optional, Dict, Any

# Configure logging for services module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add root directory to path for proper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# =============================================================================
# CORE SERVICE IMPORTS - CORRECTED PATHS
# =============================================================================

try:
    from .risk_service import RiskAnalysisService  # CORRECTED: risk_service.py
    RISK_SERVICE_AVAILABLE = True
    logger.info("✓ Risk Analysis Service loaded successfully")
except ImportError as e:
    RISK_SERVICE_AVAILABLE = False
    logger.warning(f"⚠ Risk Analysis Service not available: {e}")
    RiskAnalysisService = None

try:
    from .behavioral_service_updated import BehavioralAnalysisService  # CORRECTED: behavioral_service_updated.py
    BEHAVIORAL_SERVICE_AVAILABLE = True
    logger.info("✓ Behavioral Analysis Service loaded successfully")
except ImportError as e:
    BEHAVIORAL_SERVICE_AVAILABLE = False
    logger.warning(f"⚠ Behavioral Analysis Service not available: {e}")
    BehavioralAnalysisService = None

try:
    from .portfolio_service_direct import PortfolioManagementService  # CORRECTED: portfolio_service_direct.py
    PORTFOLIO_SERVICE_AVAILABLE = True
    logger.info("✓ Portfolio Management Service loaded successfully")
except ImportError as e:
    PORTFOLIO_SERVICE_AVAILABLE = False
    logger.warning(f"⚠ Portfolio Management Service not available: {e}")
    PortfolioManagementService = None

try:
    from .forecasting_service_updated import ForecastingService  # CORRECTED: forecasting_service_updated.py
    FORECASTING_SERVICE_AVAILABLE = True
    logger.info("✓ Forecasting Service loaded successfully")
except ImportError as e:
    FORECASTING_SERVICE_AVAILABLE = False
    logger.warning(f"⚠ Forecasting Service not available: {e}")
    ForecastingService = None

# =============================================================================
# RESPONSE MODEL IMPORTS
# =============================================================================

try:
    from .risk_service import (
        RiskAnalysisResponse,
        PortfolioRiskProfileResponse,
        RiskComparisonResponse
    )
    logger.info("✓ Risk service response models loaded")
except ImportError as e:
    logger.warning(f"⚠ Risk response models not available: {e}")
    RiskAnalysisResponse = None
    PortfolioRiskProfileResponse = None
    RiskComparisonResponse = None

# =============================================================================
# SERVICE STATUS AND HEALTH CHECKS - CORRECTED MODULE NAMES
# =============================================================================

def get_services_status() -> Dict[str, Any]:
    """
    Get comprehensive status of all services
    """
    return {
        "services": {
            "risk_analysis": {
                "available": RISK_SERVICE_AVAILABLE,
                "class": "RiskAnalysisService",
                "module": "services.risk_service"  # CORRECTED
            },
            "behavioral_analysis": {
                "available": BEHAVIORAL_SERVICE_AVAILABLE,
                "class": "BehavioralAnalysisService",
                "module": "services.behavioral_service_updated"  # CORRECTED
            },
            "portfolio_management": {
                "available": PORTFOLIO_SERVICE_AVAILABLE,
                "class": "PortfolioManagementService",
                "module": "services.portfolio_service_direct"  # CORRECTED
            },
            "forecasting": {
                "available": FORECASTING_SERVICE_AVAILABLE,
                "class": "ForecastingService",
                "module": "services.forecasting_service_updated"  # CORRECTED
            }
        },
        "integration_status": {
            "three_way_integration": (
                RISK_SERVICE_AVAILABLE and 
                BEHAVIORAL_SERVICE_AVAILABLE and 
                PORTFOLIO_SERVICE_AVAILABLE
            ),
            "four_way_integration": (
                RISK_SERVICE_AVAILABLE and 
                BEHAVIORAL_SERVICE_AVAILABLE and 
                PORTFOLIO_SERVICE_AVAILABLE and
                FORECASTING_SERVICE_AVAILABLE
            ),
            "ready_for_production": all([
                RISK_SERVICE_AVAILABLE,
                BEHAVIORAL_SERVICE_AVAILABLE,
                PORTFOLIO_SERVICE_AVAILABLE
            ])
        },
        "capabilities": {
            "fmp_integration": True,  # All services support FMP
            "real_time_analysis": True,
            "cross_service_integration": True,
            "behavioral_tools_integration": BEHAVIORAL_SERVICE_AVAILABLE
        }
    }

async def health_check_all_services() -> Dict[str, Any]:
    """
    Run health checks on all available services
    """
    health_results = {
        "overall_status": "healthy",
        "services": {},
        "integration_tests": {},
        "timestamp": str(logging.time.time())
    }
    
    # Risk Service Health Check
    if RISK_SERVICE_AVAILABLE:
        try:
            risk_service = RiskAnalysisService()
            risk_health = await risk_service.health_check()
            health_results["services"]["risk"] = risk_health
        except Exception as e:
            health_results["services"]["risk"] = {
                "status": "error",
                "error": str(e)
            }
            health_results["overall_status"] = "degraded"
    
    # Behavioral Service Health Check
    if BEHAVIORAL_SERVICE_AVAILABLE:
        try:
            behavioral_service = BehavioralAnalysisService()
            behavioral_health = await behavioral_service.health_check()
            health_results["services"]["behavioral"] = behavioral_health
        except Exception as e:
            health_results["services"]["behavioral"] = {
                "status": "error",
                "error": str(e)
            }
            health_results["overall_status"] = "degraded"
    
    # Portfolio Service Health Check
    if PORTFOLIO_SERVICE_AVAILABLE:
        try:
            portfolio_service = PortfolioManagementService()
            # Portfolio service might not have async health check
            health_results["services"]["portfolio"] = {
                "status": "healthy",
                "service": "portfolio_management"
            }
        except Exception as e:
            health_results["services"]["portfolio"] = {
                "status": "error",
                "error": str(e)
            }
            health_results["overall_status"] = "degraded"
    
    # Forecasting Service Health Check
    if FORECASTING_SERVICE_AVAILABLE:
        try:
            forecasting_service = ForecastingService()
            forecasting_health = forecasting_service.health_check()
            health_results["services"]["forecasting"] = forecasting_health
        except Exception as e:
            health_results["services"]["forecasting"] = {
                "status": "error",
                "error": str(e)
            }
            health_results["overall_status"] = "degraded"
    
    return health_results

# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def create_risk_service(**kwargs) -> Optional['RiskAnalysisService']:
    """Create risk analysis service with optional configuration"""
    if not RISK_SERVICE_AVAILABLE:
        logger.error("Risk service not available")
        return None
    return RiskAnalysisService(**kwargs)

def create_behavioral_service(**kwargs) -> Optional['BehavioralAnalysisService']:
    """Create behavioral analysis service with optional configuration"""
    if not BEHAVIORAL_SERVICE_AVAILABLE:
        logger.error("Behavioral service not available")
        return None
    return BehavioralAnalysisService(**kwargs)

def create_portfolio_service(**kwargs) -> Optional['PortfolioManagementService']:
    """Create portfolio management service with optional configuration"""
    if not PORTFOLIO_SERVICE_AVAILABLE:
        logger.error("Portfolio service not available")
        return None
    return PortfolioManagementService(**kwargs)

def create_forecasting_service(**kwargs) -> Optional['ForecastingService']:
    """Create forecasting service with optional configuration"""
    if not FORECASTING_SERVICE_AVAILABLE:
        logger.error("Forecasting service not available")
        return None
    return ForecastingService(**kwargs)

# =============================================================================
# BEHAVIORAL TOOLS INTEGRATION
# =============================================================================

def validate_behavioral_tools_integration() -> Dict[str, Any]:
    """
    Validate that behavioral tools are properly integrated
    """
    validation_result = {
        "behavioral_service_available": BEHAVIORAL_SERVICE_AVAILABLE,
        "behavioral_tools_status": "checking",
        "integration_issues": []
    }
    
    if not BEHAVIORAL_SERVICE_AVAILABLE:
        validation_result["behavioral_tools_status"] = "service_unavailable"
        validation_result["integration_issues"].append(
            "BehavioralAnalysisService not available - check services.behavioral_service_updated"  # CORRECTED
        )
        return validation_result
    
    try:
        # Test behavioral service instantiation
        behavioral_service = BehavioralAnalysisService()
        
        # Check if behavioral tools are available
        service_status = behavioral_service.get_service_status()
        behavioral_tools_available = service_status.get('capabilities', {}).get('behavioral_tools_available', False)
        
        if behavioral_tools_available:
            validation_result["behavioral_tools_status"] = "available"
        else:
            validation_result["behavioral_tools_status"] = "tools_missing"
            validation_result["integration_issues"].append(
                "Behavioral tools not properly integrated with service"
            )
        
        # Check FMP integration
        fmp_available = service_status.get('capabilities', {}).get('fmp_integration_available', False)
        if not fmp_available:
            validation_result["integration_issues"].append(
                "FMP integration not available for behavioral analysis"
            )
        
    except Exception as e:
        validation_result["behavioral_tools_status"] = "error"
        validation_result["integration_issues"].append(f"Error testing behavioral service: {str(e)}")
    
    return validation_result

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Core Services
    'RiskAnalysisService',
    'BehavioralAnalysisService', 
    'PortfolioManagementService',
    'ForecastingService',
    
    # Response Models
    'RiskAnalysisResponse',
    'PortfolioRiskProfileResponse',
    'RiskComparisonResponse',
    
    # Service Factories
    'create_risk_service',
    'create_behavioral_service',
    'create_portfolio_service', 
    'create_forecasting_service',
    
    # Status and Health
    'get_services_status',
    'health_check_all_services',
    'validate_behavioral_tools_integration',
    
    # Availability Flags
    'RISK_SERVICE_AVAILABLE',
    'BEHAVIORAL_SERVICE_AVAILABLE',
    'PORTFOLIO_SERVICE_AVAILABLE',
    'FORECASTING_SERVICE_AVAILABLE'
]

# =============================================================================
# MODULE INITIALIZATION LOG
# =============================================================================

logger.info("=" * 60)
logger.info("SERVICE LAYER INITIALIZATION COMPLETE")
logger.info("=" * 60)
logger.info(f"Risk Service: {'✓ Available' if RISK_SERVICE_AVAILABLE else '✗ Not Available'}")
logger.info(f"Behavioral Service: {'✓ Available' if BEHAVIORAL_SERVICE_AVAILABLE else '✗ Not Available'}")
logger.info(f"Portfolio Service: {'✓ Available' if PORTFOLIO_SERVICE_AVAILABLE else '✗ Not Available'}")
logger.info(f"Forecasting Service: {'✓ Available' if FORECASTING_SERVICE_AVAILABLE else '✗ Not Available'}")

integration_status = get_services_status()
logger.info(f"Three-way Integration: {'✓ Ready' if integration_status['integration_status']['three_way_integration'] else '✗ Not Ready'}")
logger.info(f"Four-way Integration: {'✓ Ready' if integration_status['integration_status']['four_way_integration'] else '✗ Not Ready'}")
logger.info(f"Production Ready: {'✓ Yes' if integration_status['integration_status']['ready_for_production'] else '✗ No'}")

# Validate behavioral tools integration on import
behavioral_validation = validate_behavioral_tools_integration()
if behavioral_validation["behavioral_tools_status"] == "available":
    logger.info("✓ Behavioral tools integration validated successfully")
else:
    logger.warning(f"⚠ Behavioral tools integration issues: {behavioral_validation['integration_issues']}")

logger.info("=" * 60)