from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Union
import logging
import time

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/risk", tags=["risk"])

# Import risk service and models - THIS WAS MISSING
try:
    from core.dependencies import get_risk_service
    risk_service = get_risk_service()  # Gets properly configured instance
    SERVICE_AVAILABLE = True
except Exception as e:
    logger.error(f"❌ Risk service not available: {e}")
    risk_service = None
    SERVICE_AVAILABLE = False

# Import centralized models if available
try:
    from models.requests import RiskAnalysisRequest
    CENTRALIZED_MODELS = True
except ImportError:
    CENTRALIZED_MODELS = False

@router.get("/health")
async def risk_health():
    """Risk service health check"""
    return {
        "service": "risk_analysis", 
        "status": "healthy" if SERVICE_AVAILABLE else "degraded",
        "centralized_models": CENTRALIZED_MODELS,
        "accepted_parameters": ['symbols', 'weights', 'portfolio_id', 'period', 'include_stress_testing', 'include_volatility_forecast', 'use_real_data'],
        "timestamp": time.time()
    }

@router.post("/analyze")
async def analyze_risk(request_data: Dict[str, Any]):
    """Risk analysis endpoint - simplified"""
    
    if not SERVICE_AVAILABLE or not risk_service:
        raise HTTPException(status_code=503, detail="Risk service not available")
    
    try:
        # Direct call to service method that actually exists
        result = await risk_service.analyze_portfolio_risk(request_data)
        return result
        
    except Exception as e:
        logger.error(f"Risk analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Risk analysis failed: {str(e)}")