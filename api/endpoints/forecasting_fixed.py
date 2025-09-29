# api/endpoints/forecasting_fixed.py - Fixed Forecasting API Router
"""
Forecasting Analysis API Endpoints - Fixed and Production Ready
==============================================================

Complete version of forecasting API that resolves all import and model issues.
Follows the exact proven pattern from behavioral_fixed.py.
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

# Fixed centralized models imports (no duplicate imports)
from models.requests import (
    ReturnForecastRequest, VolatilityForecastRequest, RegimeAnalysisRequest,
    ScenarioAnalysisRequest, ComprehensiveForecastRequest,
    IntegratedForecastRiskRequest, IntegratedForecastBehavioralRequest,
    FourWayIntegratedAnalysisRequest,
    ForecastingAnalysisResponse, RegimeAnalysisResponse, IntegratedAnalysisResponse,
    ForecastingHealthResponse
)

# Import service (fixed path)
from services.forecasting_service_updated import ForecastingService

# Configure logging
logger = logging.getLogger(__name__)

# Create router with proper export
router = APIRouter(
    prefix="/api/v1/forecasting",
    tags=["forecasting", "time-series", "regimes", "fmp-integration"]
)

# ============================================================================
# CORE FORECASTING ENDPOINTS
# ============================================================================

@router.post("/returns", response_model=ForecastingAnalysisResponse,
             summary="Return Forecasting Analysis",
             description="Generate portfolio return forecasts with Auto-ARIMA and regime conditioning")
async def forecast_returns(request: ReturnForecastRequest):
    """
    Generate return forecasts with FMP integration following proven pattern
    """
    try:
        logger.info(f"Return forecasting: symbols={request.symbols}, horizon={request.forecast_horizon}")
        
        # Direct service instantiation (proven pattern)
        service = ForecastingService()
        
        # Call service with proper parameter handling
        result = service.forecast_returns(
            symbols=request.symbols,
            period=request.period,  # Pass enum directly, service will normalize
            use_real_data=request.use_real_data,
            forecast_horizon=request.forecast_horizon,  # Pass enum directly
            model_type=request.model_type,  # Pass enum directly
            confidence_levels=request.confidence_levels,
            include_regime_conditioning=request.include_regime_conditioning
        )
        
        # Check result success (handles both centralized and legacy responses)
        success = getattr(result, 'success', False) if hasattr(result, 'success') else result.get('success', False)
        
        if not success:
            error = getattr(result, 'error', 'Unknown error') if hasattr(result, 'error') else result.get('error', 'Unknown error')
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Return forecasting failed: {error}"
            )
        
        logger.info(f"Return forecasting completed successfully")
        
        # Return result directly (service handles response format)
        return result
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Return forecasting endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during return forecasting: {str(e)}"
        )

@router.post("/volatility", response_model=ForecastingAnalysisResponse,
             summary="Volatility Forecasting Analysis", 
             description="Generate volatility forecasts with GARCH and regime-switching models")
async def forecast_volatility(request: VolatilityForecastRequest):
    """
    Generate volatility forecasts with FMP integration
    """
    try:
        logger.info(f"Volatility forecasting: symbols={request.symbols}, model={request.volatility_model}")
        
        service = ForecastingService()
        
        result = service.forecast_volatility(
            symbols=request.symbols,
            period=request.period,
            use_real_data=request.use_real_data,
            forecast_horizon=request.forecast_horizon,
            volatility_model=request.volatility_model,
            include_regime_switching=request.include_regime_switching
        )
        
        success = getattr(result, 'success', False) if hasattr(result, 'success') else result.get('success', False)
        
        if not success:
            error = getattr(result, 'error', 'Unknown error') if hasattr(result, 'error') else result.get('error', 'Unknown error')
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Volatility forecasting failed: {error}"
            )
        
        logger.info(f"Volatility forecasting completed")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Volatility forecasting endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during volatility forecasting: {str(e)}"
        )

@router.post("/regimes", response_model=RegimeAnalysisResponse,
             summary="Market Regime Analysis",
             description="Detect market regimes with HMM and volatility-based methods")
async def analyze_market_regimes(request: RegimeAnalysisRequest):
    """
    Market regime detection with FMP integration
    """
    try:
        logger.info(f"Regime analysis: symbols={request.symbols}, method={request.regime_method}")
        
        service = ForecastingService()
        
        result = service.analyze_market_regimes(
            symbols=request.symbols,
            period=request.period,
            use_real_data=request.use_real_data,
            regime_method=request.regime_method,
            n_regimes=request.n_regimes,
            include_transitions=request.include_transitions
        )
        
        success = getattr(result, 'success', False) if hasattr(result, 'success') else result.get('success', False)
        
        if not success:
            error = getattr(result, 'error', 'Unknown error') if hasattr(result, 'error') else result.get('error', 'Unknown error')
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Regime analysis failed: {error}"
            )
        
        logger.info(f"Regime analysis completed")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Regime analysis endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during regime analysis: {str(e)}"
        )

@router.post("/scenarios", response_model=ForecastingAnalysisResponse,
             summary="Scenario Analysis",
             description="Generate scenarios with Monte Carlo simulation and stress testing")
async def generate_scenarios(request: ScenarioAnalysisRequest):
    """
    Scenario analysis with Monte Carlo simulation
    """
    try:
        logger.info(f"Scenario analysis: symbols={request.symbols}, paths={request.monte_carlo_paths}")
        
        service = ForecastingService()
        
        result = service.generate_scenarios(
            symbols=request.symbols,
            period=request.period,
            use_real_data=request.use_real_data,
            forecast_horizon=request.forecast_horizon,
            scenarios=request.scenarios,
            monte_carlo_paths=request.monte_carlo_paths
        )
        
        success = getattr(result, 'success', False) if hasattr(result, 'success') else result.get('success', False)
        
        if not success:
            error = getattr(result, 'error', 'Unknown error') if hasattr(result, 'error') else result.get('error', 'Unknown error')
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Scenario analysis failed: {error}"
            )
        
        logger.info(f"Scenario analysis completed")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scenario analysis endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during scenario analysis: {str(e)}"
        )

@router.post("/comprehensive", response_model=ForecastingAnalysisResponse,
             summary="Comprehensive Forecasting Analysis",
             description="Complete forecasting suite combining returns, volatility, regimes, and scenarios")
async def comprehensive_forecast_analysis(request: ComprehensiveForecastRequest):
    """
    Comprehensive forecasting analysis combining all components
    """
    try:
        logger.info(f"Comprehensive forecasting: symbols={request.symbols}")
        
        service = ForecastingService()
        
        result = service.comprehensive_forecast_analysis(
            symbols=request.symbols,
            period=request.period,
            use_real_data=request.use_real_data,
            forecast_horizon=request.forecast_horizon,
            include_returns=request.include_returns,
            include_volatility=request.include_volatility,
            include_regimes=request.include_regimes,
            include_scenarios=request.include_scenarios
        )
        
        success = getattr(result, 'success', False) if hasattr(result, 'success') else result.get('success', False)
        
        if not success:
            error = getattr(result, 'error', 'Unknown error') if hasattr(result, 'error') else result.get('error', 'Unknown error')
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Comprehensive forecasting failed: {error}"
            )
        
        logger.info(f"Comprehensive forecasting completed")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comprehensive forecasting endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during comprehensive forecasting: {str(e)}"
        )

# ============================================================================
# INTEGRATED ANALYSIS ENDPOINTS
# ============================================================================

@router.post("/integrated/with-risk", response_model=IntegratedAnalysisResponse,
             summary="Integrated Forecasting + Risk Analysis",
             description="Combine forecasting with risk analysis for regime-aware risk assessment")
async def integrated_forecast_risk_analysis(request: IntegratedForecastRiskRequest):
    """
    Integrated forecasting with risk analysis
    """
    try:
        logger.info(f"Integrated forecast+risk: symbols={request.symbols}")
        
        service = ForecastingService()
        
        result = service.integrated_analysis_with_risk_context(
            symbols=request.symbols,
            period=request.period,
            use_real_data=request.use_real_data,
            forecast_horizon=request.forecast_horizon,
            risk_analysis_type=request.risk_analysis_type
        )
        
        if not result.get('success', False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Integrated forecast-risk analysis failed: {result.get('error', 'Unknown error')}"
            )
        
        logger.info(f"Integrated forecast-risk analysis completed")
        
        # Convert dict result to proper response format
        return IntegratedAnalysisResponse(
            success=True,
            service="forecasting_analysis",
            analysis_type="integrated_forecast_risk",
            timestamp=datetime.now(),
            service_metadata={
                "service_name": "forecasting",
                "version": "5.1.0",
                "analysis_duration": result.get('execution_time', 0),
                "fmp_integration_used": "FMP" in result.get('data_source', ''),
                "data_source": result.get('data_source', 'Unknown')
            },
            symbols=request.symbols,
            period=request.period.value if hasattr(request.period, 'value') else str(request.period),
            integration_level="TWO_WAY",
            data_source=result.get('data_source', 'FMP'),
            fmp_integration="FMP" in result.get('data_source', ''),
            forecasting_analysis=result.get('forecasting_analysis', {}),
            risk_analysis=result.get('risk_analysis', {}),
            cross_analysis_insights=result.get('cross_analysis_insights', {}),
            integrated_insights=result.get('integrated_insights', []),
            analysis_duration=result.get('execution_time', 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Integrated forecast-risk endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during integrated forecast-risk analysis: {str(e)}"
        )

@router.post("/integrated/with-behavioral", response_model=IntegratedAnalysisResponse,
             summary="Integrated Forecasting + Behavioral Analysis",
             description="Combine forecasting with behavioral analysis for bias-aware forecasting")
async def integrated_forecast_behavioral_analysis(request: IntegratedForecastBehavioralRequest):
    """
    Integrated forecasting with behavioral analysis
    """
    try:
        logger.info(f"Integrated forecast+behavioral: symbols={request.symbols}, messages={len(request.conversation_messages)}")
        
        service = ForecastingService()
        
        # Convert ConversationMessage objects to dict format (proven pattern)
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.conversation_messages
        ]
        
        result = service.integrated_analysis_with_behavioral_context(
            symbols=request.symbols,
            conversation_messages=messages,
            period=request.period,
            use_real_data=request.use_real_data,
            forecast_horizon=request.forecast_horizon
        )
        
        if not result.get('success', False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Integrated forecast-behavioral analysis failed: {result.get('error', 'Unknown error')}"
            )
        
        logger.info(f"Integrated forecast-behavioral analysis completed")
        
        return IntegratedAnalysisResponse(
            success=True,
            service="forecasting_analysis",
            analysis_type="integrated_forecast_behavioral",
            timestamp=datetime.now(),
            service_metadata={
                "service_name": "forecasting",
                "version": "5.1.0",
                "analysis_duration": result.get('execution_time', 0),
                "fmp_integration_used": "FMP" in result.get('data_source', ''),
                "data_source": result.get('data_source', 'Unknown')
            },
            symbols=request.symbols,
            period=request.period.value if hasattr(request.period, 'value') else str(request.period),
            integration_level="TWO_WAY",
            data_source=result.get('data_source', 'FMP'),
            fmp_integration="FMP" in result.get('data_source', ''),
            forecasting_analysis=result.get('forecasting_analysis', {}),
            behavioral_analysis=result.get('behavioral_analysis', {}),
            cross_analysis_insights=result.get('cross_analysis_insights', {}),
            integrated_insights=result.get('integrated_insights', []),
            analysis_duration=result.get('execution_time', 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Integrated forecast-behavioral endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during integrated forecast-behavioral analysis: {str(e)}"
        )

@router.post("/integrated/four-way", response_model=IntegratedAnalysisResponse,
             summary="Four-Way Integrated Analysis",
             description="Ultimate analysis combining forecasting, risk, behavioral, and portfolio analysis")
async def four_way_integrated_analysis(request: FourWayIntegratedAnalysisRequest):
    """
    Complete four-way integrated analysis
    """
    try:
        logger.info(f"Four-way integration: symbols={request.symbols}, depth={request.integration_depth}")
        
        service = ForecastingService()
        
        # Convert ConversationMessage objects to dict format
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.conversation_messages
        ]
        
        result = service.four_way_integrated_analysis(
            symbols=request.symbols,
            conversation_messages=messages,
            portfolio_request=request.portfolio_request,
            period=request.period,
            use_real_data=request.use_real_data,
            forecast_horizon=request.forecast_horizon,
            integration_depth=request.integration_depth
        )
        
        if not result.get('success', False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Four-way integrated analysis failed: {result.get('error', 'Unknown error')}"
            )
        
        analysis_duration = result.get('analysis_duration', 0)
        logger.info(f"Four-way analysis completed in {analysis_duration:.2f}s")
        
        return IntegratedAnalysisResponse(
            success=True,
            service="forecasting_analysis",
            analysis_type="four_way_integrated",
            timestamp=datetime.now(),
            service_metadata={
                "service_name": "forecasting",
                "version": "5.1.0",
                "analysis_duration": analysis_duration,
                "fmp_integration_used": "FMP" in result.get('data_source', ''),
                "data_source": result.get('data_source', 'Unknown')
            },
            symbols=request.symbols,
            period=request.period.value if hasattr(request.period, 'value') else str(request.period),
            integration_level="FOUR_WAY",
            data_source=result.get('data_source', 'Multi-service Integration'),
            fmp_integration=result.get('fmp_integration_used', False),
            analysis_components=result.get('analysis_components', {}),
            forecasting_analysis=result.get('forecasting_analysis', {}),
            risk_analysis=result.get('risk_analysis', {}),
            behavioral_analysis=result.get('behavioral_analysis', {}),
            portfolio_analysis=result.get('portfolio_analysis', {}),
            cross_analysis_insights=result.get('cross_analysis_insights', {}),
            integrated_insights=result.get('integrated_insights', []),
            unified_recommendations=result.get('unified_recommendations', []),
            analysis_duration=analysis_duration
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Four-way integrated analysis endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during four-way integrated analysis: {str(e)}"
        )

# ============================================================================
# HEALTH CHECK AND UTILITY ENDPOINTS
# ============================================================================

@router.get("/health", response_model=ForecastingHealthResponse,
            summary="Forecasting Service Health Check",
            description="Check forecasting service health and integration status")
async def forecasting_service_health():
    """
    Forecasting service health check following proven pattern
    """
    try:
        service = ForecastingService()
        health_status = service.health_check()
        
        return ForecastingHealthResponse(
            service="forecasting_analysis",
            status=health_status.get("status", "healthy"),
            version=health_status.get("version", "5.1.0"),
            timestamp=datetime.now(),
            fmp_integration_status=health_status.get("fmp_integration_status", {}),
            tools_status=health_status.get("tools_status", {}),
            integrated_services_status=health_status.get("integrated_services_status", {}),
            four_way_integration=health_status.get("four_way_integration", "unknown"),
            capabilities={
                "forecasting": ["return_forecasting", "volatility_forecasting", "regime_analysis", "scenario_analysis"],
                "integration": ["risk_integration", "behavioral_integration", "portfolio_integration"],
                "data_sources": ["fmp_real_data", "synthetic_fallback"],
                "models": ["auto_arima", "garch", "hmm_regimes", "monte_carlo"]
            }
        )
        
    except Exception as e:
        logger.error(f"Forecasting service health check error: {e}")
        return ForecastingHealthResponse(
            service="forecasting_analysis",
            status="error",
            version="5.1.0",
            timestamp=datetime.now(),
            fmp_integration_status={},
            tools_status={},
            integrated_services_status={},
            error=str(e)
        )

@router.get("/capabilities",
            summary="Forecasting Service Capabilities",
            description="Get detailed forecasting service capabilities and integration status")
async def get_forecasting_capabilities():
    """
    Get detailed forecasting service capabilities
    """
    try:
        service = ForecastingService()
        health_status = service.health_check()
        
        return {
            "service": "forecasting_analysis",
            "version": "5.1.0",
            "core_capabilities": {
                "return_forecasting": {
                    "methods": ["auto_arima", "regime_conditional", "simple_mean"],
                    "confidence_intervals": True,
                    "regime_conditioning": True,
                    "fmp_data_support": True
                },
                "volatility_forecasting": {
                    "methods": ["garch", "regime_switching", "rolling_volatility"],
                    "regime_switching": True,
                    "confidence_bands": True,
                    "fmp_data_support": True
                },
                "regime_analysis": {
                    "methods": ["hmm", "volatility_based", "comprehensive"],
                    "transition_analysis": True,
                    "n_regimes_range": [2, 5],
                    "fmp_data_support": True
                },
                "scenario_analysis": {
                    "monte_carlo": True,
                    "custom_scenarios": True,
                    "tail_risk": True,
                    "stress_testing": True,
                    "fmp_data_support": True
                }
            },
            "integration_capabilities": {
                "risk_integration": health_status.get("integrated_services_status", {}).get("risk_service") == "available",
                "behavioral_integration": health_status.get("integrated_services_status", {}).get("behavioral_service") == "available",
                "portfolio_integration": health_status.get("integrated_services_status", {}).get("portfolio_service") == "available",
                "four_way_integration": health_status.get("four_way_integration", "").startswith("ready")
            },
            "data_capabilities": {
                "fmp_real_data": health_status.get("fmp_integration_status", {}).get("forecasting", {}).get("fmp_integration", False),
                "synthetic_fallback": True,
                "supported_periods": ["3months", "6months", "1year", "2years", "3years", "5years"],
                "supported_symbols": "Any valid stock symbols"
            },
            "performance_characteristics": {
                "forecast_horizons": "1-252 days",
                "typical_response_time": "1-5 seconds",
                "four_way_analysis_time": "5-15 seconds",
                "concurrent_requests": "Supported with proper resource management"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Capabilities endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving capabilities: {str(e)}"
        )

@router.get("/status",
            summary="Quick Forecasting Status Check",
            description="Quick status check for forecasting service operational state")
async def get_forecasting_status():
    """
    Quick status check for forecasting service
    """
    try:
        service = ForecastingService()
        health_status = service.health_check()
        
        return {
            "service": "forecasting_analysis",
            "status": health_status.get("status", "unknown"),
            "tools_available": health_status.get("tools_status", {}).get("forecasting_tools") == "available",
            "regime_tools_available": health_status.get("tools_status", {}).get("regime_tools") == "integrated",
            "fmp_integration": health_status.get("fmp_integration_status", {}).get("forecasting", {}).get("fmp_integration", False),
            "integration_ready": health_status.get("four_way_integration", "").startswith("ready"),
            "integration_level": health_status.get("four_way_integration", "unknown"),
            "centralized_models": service.centralized_models_available,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Status endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving status: {str(e)}"
        )