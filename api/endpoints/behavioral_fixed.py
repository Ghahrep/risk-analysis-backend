# api/endpoints/behavioral_fixed.py - Updated with Centralized Models
"""
Behavioral Analysis API Endpoints - Updated with Centralized Models
===================================================================

RESTful API endpoints for behavioral finance analysis with real FMP market data.
Updated to use centralized models/requests.py instead of inline models.
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

# Import centralized models
from models.requests import (
    BehavioralAnalysisRequest,
    BiasDetectionRequest, 
    SentimentAnalysisRequest,
    BehavioralProfileRequest,
    PortfolioContextRequest,
    BehavioralAnalysisResponse,
    HealthResponse
)

# Import the fixed service layer
from services.behavioral_service_direct import BehavioralAnalysisService

logger = logging.getLogger(__name__)

# Create router with consistent prefix pattern
router = APIRouter(prefix="/api/v1/behavioral", tags=["behavioral-analysis"])

# =============================================================================
# MAIN BEHAVIORAL ANALYSIS ENDPOINTS
# =============================================================================

@router.post("/analyze", response_model=BehavioralAnalysisResponse,
             summary="Comprehensive Behavioral Analysis",
             description="Complete behavioral analysis with bias detection, sentiment analysis, and profile assessment")
async def comprehensive_behavioral_analysis(request: BehavioralAnalysisRequest):
    """
    MAIN: Comprehensive behavioral analysis with FMP integration
    
    Analyzes conversation patterns to detect cognitive biases, assess market sentiment,
    and create behavioral profile. Enhanced with real portfolio data when symbols provided.
    """
    try:
        logger.info(f"Starting comprehensive behavioral analysis for {len(request.conversation_messages)} messages")
        
        # Initialize service with dependency injection
        service = BehavioralAnalysisService(
            risk_free_rate=0.02,
            confidence_levels=[0.95, 0.99],
            default_period=request.period.value
        )
        
        # Convert conversation messages to expected format
        messages = [
            {"role": msg.role, "content": msg.content} 
            for msg in request.conversation_messages
        ]
        
        # Call the service
        analysis_result = await service.comprehensive_behavioral_analysis(
            conversation_messages=messages,
            symbols=request.symbols or [],
            period=request.period.value,
            use_real_data=request.use_real_data
        )
        
        if not analysis_result.success:
            raise HTTPException(
                status_code=400,
                detail=f"Behavioral analysis failed: {analysis_result.error}"
            )
        
        # Return standardized response
        return BehavioralAnalysisResponse(
            success=True,
            message="Comprehensive behavioral analysis completed successfully",
            data_source=analysis_result.data_source,
            execution_time=analysis_result.execution_time,
            bias_count=analysis_result.bias_count,
            overall_risk_score=analysis_result.overall_risk_score,
            detected_biases=analysis_result.detected_biases or [],
            recommendations=analysis_result.recommendations or []
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comprehensive behavioral analysis endpoint failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Behavioral analysis service error: {str(e)}"
        )

@router.post("/bias-detection", response_model=BehavioralAnalysisResponse,
             summary="Cognitive Bias Detection", 
             description="Detect specific cognitive biases in investment decision-making")
async def detect_behavioral_biases(request: BiasDetectionRequest):
    """
    SPECIALIZED: Targeted cognitive bias detection with FMP enhancement
    """
    try:
        logger.info(f"Starting bias detection for {len(request.conversation_messages)} messages")
        
        service = BehavioralAnalysisService(default_period=request.period.value)
        
        # Convert conversation messages
        messages = [
            {"role": msg.role, "content": msg.content} 
            for msg in request.conversation_messages
        ]
        
        # Call service
        analysis_result = await service.detect_specific_biases(
            conversation_messages=messages,
            bias_types=request.bias_types or [],
            symbols=request.symbols,
            period=request.period.value,
            use_real_data=request.use_real_data
        )
        
        if not analysis_result.get('success', False):
            raise HTTPException(
                status_code=400,
                detail=f"Bias detection failed: {analysis_result.get('error', 'Unknown error')}"
            )
        
        return BehavioralAnalysisResponse(
            success=True,
            message="Bias detection completed successfully",
            data_source=analysis_result.get('data_source', 'Unknown'),
            execution_time=analysis_result.get('execution_time', 0),
            bias_count=analysis_result.get('bias_count', 0),
            overall_risk_score=analysis_result.get('overall_risk_score', 0),
            detected_biases=analysis_result.get('targeted_biases', []),
            recommendations=analysis_result.get('recommendations', [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bias detection endpoint failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Bias detection service error: {str(e)}"
        )

@router.post("/sentiment-analysis", response_model=BehavioralAnalysisResponse,
             summary="Market Sentiment Analysis",
             description="Analyze investor sentiment and market timing risk from conversation patterns")
async def analyze_market_sentiment_endpoint(request: SentimentAnalysisRequest):
    """
    SPECIALIZED: Market sentiment analysis with FMP market context
    """
    try:
        logger.info(f"Starting sentiment analysis for {len(request.conversation_messages)} messages")
        
        service = BehavioralAnalysisService(default_period=request.period.value)
        
        # Convert conversation messages
        messages = [
            {"role": msg.role, "content": msg.content} 
            for msg in request.conversation_messages
        ]
        
        # Call service
        sentiment_result = await service.sentiment_analysis_with_market_context(
            conversation_messages=messages,
            symbols=request.symbols,
            period=request.period.value,
            use_real_data=request.use_real_data
        )
        
        if not sentiment_result.success:
            raise HTTPException(
                status_code=400,
                detail=f"Sentiment analysis failed: {sentiment_result.error}"
            )
        
        return BehavioralAnalysisResponse(
            success=True,
            message="Market sentiment analysis completed successfully",
            data_source=sentiment_result.data_source,
            execution_time=sentiment_result.execution_time,
            bias_count=0,  # Sentiment analysis doesn't count biases
            overall_risk_score=sentiment_result.market_timing_risk,
            detected_biases=[],
            recommendations=sentiment_result.recommendations or []
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sentiment analysis endpoint failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment analysis service error: {str(e)}"
        )

@router.post("/profile-assessment", response_model=BehavioralAnalysisResponse,
             summary="Behavioral Profile Assessment",
             description="Comprehensive behavioral maturity and risk profile assessment")
async def assess_behavioral_profile_endpoint(request: BehavioralProfileRequest):
    """
    SPECIALIZED: Comprehensive behavioral profile assessment with FMP context
    """
    try:
        logger.info(f"Starting behavioral profile assessment for {len(request.conversation_messages)} messages")
        
        service = BehavioralAnalysisService(default_period=request.period.value)
        
        # Convert conversation messages
        messages = [
            {"role": msg.role, "content": msg.content} 
            for msg in request.conversation_messages
        ]
        
        # Call service
        profile_result = await service.behavioral_profile_assessment(
            conversation_messages=messages,
            symbols=request.symbols,
            period=request.period.value,
            user_demographics=request.user_demographics,
            use_real_data=request.use_real_data
        )
        
        if not profile_result.success:
            raise HTTPException(
                status_code=400,
                detail=f"Profile assessment failed: {profile_result.error}"
            )
        
        return BehavioralAnalysisResponse(
            success=True,
            message="Behavioral profile assessment completed successfully",
            data_source=profile_result.data_source,
            execution_time=profile_result.execution_time,
            bias_count=len(profile_result.dominant_biases) if profile_result.dominant_biases else 0,
            overall_risk_score=profile_result.overall_risk_score,
            detected_biases=profile_result.dominant_biases or [],
            recommendations=profile_result.recommendations or []
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile assessment endpoint failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Profile assessment service error: {str(e)}"
        )

@router.post("/portfolio-context", 
             summary="Portfolio Behavioral Context",
             description="Analyze portfolio-specific behavioral risk factors without conversation")
async def get_portfolio_behavioral_context(request: PortfolioContextRequest):
    """
    UTILITY: Get portfolio behavioral context analysis
    """
    try:
        logger.info(f"Getting portfolio behavioral context for {len(request.symbols)} symbols")
        
        service = BehavioralAnalysisService(default_period=request.period.value)
        
        # Call service
        context_result = await service.get_portfolio_behavioral_context(
            symbols=request.symbols,
            period=request.period.value
        )
        
        if not context_result.get('success', False):
            raise HTTPException(
                status_code=400,
                detail=f"Portfolio context analysis failed: {context_result.get('error', 'Unknown error')}"
            )
        
        response_data = {
            "success": True,
            "message": "Portfolio behavioral context analysis completed successfully",
            "data_source": context_result.get('data_source', 'FMP'),
            "execution_time": context_result.get('execution_time', 0),
            "results": {
                "behavioral_risk_factors": context_result.get('behavioral_risk_factors', {}),
                "performance_metrics": context_result.get('performance_metrics', {}),
                "sector_analysis": context_result.get('sector_analysis', {}),
                "key_concerns": context_result.get('key_concerns', [])
            },
            "metadata": {
                "symbols_analyzed": request.symbols,
                "period": request.period.value,
                "analysis_timestamp": context_result.get('analysis_timestamp')
            }
        }
        
        logger.info(f"Portfolio context analysis completed for {len(request.symbols)} symbols")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Portfolio context endpoint failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Portfolio context service error: {str(e)}"
        )

# =============================================================================
# UTILITY AND STATUS ENDPOINTS
# =============================================================================

@router.get("/health", response_model=HealthResponse,
            summary="Service Health Check",
            description="Check behavioral analysis service health and FMP integration status")
async def behavioral_health_check():
    """Health check for behavioral analysis service"""
    try:
        service = BehavioralAnalysisService()
        
        health_status = await service.health_check()
        
        return HealthResponse(
            service="behavioral_analysis",
            status=health_status.get("status", "healthy"),
            version="2.0",
            timestamp=datetime.now(),
            error=health_status.get("error")
        )
        
    except Exception as e:
        logger.error(f"Behavioral health check failed: {e}")
        return HealthResponse(
            service="behavioral_analysis",
            status="unhealthy",
            version="2.0",
            timestamp=datetime.now(),
            error=str(e)
        )

@router.get("/capabilities",
            summary="Service Capabilities",
            description="Get detailed information about behavioral analysis capabilities")
async def get_behavioral_capabilities():
    """Get comprehensive behavioral analysis capabilities"""
    try:
        service = BehavioralAnalysisService()
        service_status = service.get_service_status()
        
        return {
            "service_info": service_status,
            "api_endpoints": {
                "/analyze": "Comprehensive behavioral analysis with bias detection, sentiment, and profiling",
                "/bias-detection": "Targeted cognitive bias detection",
                "/sentiment-analysis": "Market sentiment and emotional state analysis",
                "/profile-assessment": "Behavioral maturity and decision-making confidence assessment",
                "/portfolio-context": "Portfolio behavioral risk factors analysis"
            },
            "supported_bias_types": [
                "loss_aversion", "overconfidence", "herding_fomo", "anchoring", "confirmation"
            ],
            "fmp_enhancements": service_status['capabilities'].get('fmp_integration_available', False)
        }
        
    except Exception as e:
        logger.error(f"Failed to get behavioral capabilities: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve service capabilities")