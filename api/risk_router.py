"""
api/risk_router.py - Clean Risk Analysis API Router
===================================================

Clean, properly structured API router with proper error handling
and request validation.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging
import time

from models.risk_models import (
    RiskAnalysisRequest, 
    RiskAnalysisResponse,
    RiskComparisonRequest,
    RiskComparisonResponse
)
from services.risk_service import RiskAnalysisService
from core.dependencies import get_risk_service_dependency
from core.exceptions import RiskAnalysisError, DataProviderError

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/risk", tags=["Risk Analysis"])

@router.get("/health")
async def health_check(
    risk_service: RiskAnalysisService = Depends(get_risk_service_dependency)
):
    """
    Comprehensive health check for risk analysis service
    
    Returns detailed status of all risk analysis capabilities
    """
    try:
        health_result = await risk_service.health_check()
        
        # Add router-specific health info
        health_result.update({
            "api_router": "healthy",
            "endpoints": {
                "analyze": "/api/risk/analyze",
                "compare": "/api/risk/compare", 
                "health": "/api/risk/health"
            }
        })
        
        return health_result
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "service": "risk_analysis_api",
                "error": str(e),
                "timestamp": time.time()
            }
        )

@router.post("/analyze", response_model=RiskAnalysisResponse)
async def analyze_portfolio_risk(
    request: RiskAnalysisRequest,
    risk_service: RiskAnalysisService = Depends(get_risk_service_dependency)
):
    """
    Comprehensive portfolio risk analysis
    
    Performs comprehensive risk analysis including:
    - Value at Risk (VaR) and Conditional VaR (CVaR) calculations
    - Risk-adjusted performance metrics (Sharpe, Sortino ratios)
    - Maximum drawdown analysis
    - Distribution characteristics (skewness, kurtosis)
    - Optional stress testing scenarios
    - Risk insights and recommendations
    
    Args:
        request: Risk analysis request with portfolio details
        
    Returns:
        RiskAnalysisResponse: Comprehensive risk analysis results
        
    Raises:
        HTTPException: 400 for validation errors, 500 for service errors
    """
    try:
        logger.info(
            f"Risk analysis request: {len(request.symbols)} symbols, "
            f"period={request.period}, real_data={request.use_real_data}"
        )
        
        # Perform analysis
        result = await risk_service.analyze_portfolio_risk(request)
        
        # Log result summary
        if result.success:
            logger.info(
                f"Risk analysis completed: data_source={result.data_source}, "
                f"execution_time={result.execution_time:.2f}s"
            )
        else:
            logger.warning(f"Risk analysis failed: {result.error}")
        
        return result
        
    except RiskAnalysisError as e:
        logger.error(f"Risk analysis error: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Risk Analysis Error",
                "message": str(e),
                "type": "validation_error"
            }
        )
    
    except DataProviderError as e:
        logger.error(f"Data provider error: {e}")
        # Return degraded service instead of error
        return RiskAnalysisResponse(
            success=False,
            message="Data provider unavailable - using synthetic data",
            data_source="Synthetic (provider error)",
            execution_time=0.0,
            timestamp=time.time(),
            error=str(e),
            warnings=["Real market data unavailable, using synthetic fallback"]
        )
    
    except Exception as e:
        logger.error(f"Unexpected error in risk analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal Server Error",
                "message": "Risk analysis service encountered an unexpected error",
                "type": "service_error"
            }
        )

@router.post("/compare", response_model=RiskComparisonResponse)  
async def compare_portfolio_risk(
    request: RiskComparisonRequest,
    risk_service: RiskAnalysisService = Depends(get_risk_service_dependency)
):
    """
    Compare portfolio risk vs benchmark
    
    Compares risk metrics between a portfolio and benchmark to assess
    relative performance and risk characteristics.
    
    Args:
        request: Risk comparison request with portfolio and benchmark details
        
    Returns:
        RiskComparisonResponse: Comparative risk analysis results
    """
    try:
        logger.info(
            f"Risk comparison: {len(request.portfolio_symbols)} portfolio symbols "
            f"vs {len(request.benchmark_symbols)} benchmark symbols"
        )
        
        # Analyze portfolio
        portfolio_request = RiskAnalysisRequest(
            symbols=request.portfolio_symbols,
            weights=request.portfolio_weights,
            period=request.period,
            use_real_data=request.use_real_data,
            include_stress_testing=False  # Skip for comparison
        )
        
        portfolio_result = await risk_service.analyze_portfolio_risk(portfolio_request)
        
        # Analyze benchmark
        benchmark_request = RiskAnalysisRequest(
            symbols=request.benchmark_symbols,
            weights=request.benchmark_weights,
            period=request.period,
            use_real_data=request.use_real_data,
            include_stress_testing=False
        )
        
        benchmark_result = await risk_service.analyze_portfolio_risk(benchmark_request)
        
        # Check if both analyses succeeded
        if not portfolio_result.success or not benchmark_result.success:
            return RiskComparisonResponse(
                success=False,
                relative_performance={},
                outperformance_probability=0.0,
                data_source="Error",
                error="Failed to analyze portfolio or benchmark"
            )
        
        # Calculate relative performance
        port_sharpe = portfolio_result.risk_metrics.sharpe_ratio
        bench_sharpe = benchmark_result.risk_metrics.sharpe_ratio
        sharpe_diff = port_sharpe - bench_sharpe
        
        # Estimate outperformance probability (simplified)
        outperformance_prob = max(0.0, min(1.0, 0.5 + sharpe_diff * 0.3))
        
        return RiskComparisonResponse(
            success=True,
            portfolio_metrics=portfolio_result.risk_metrics,
            benchmark_metrics=benchmark_result.risk_metrics,
            relative_performance={
                "sharpe_ratio_difference": sharpe_diff,
                "volatility_difference": (
                    portfolio_result.risk_metrics.annualized_volatility - 
                    benchmark_result.risk_metrics.annualized_volatility
                ),
                "return_difference": (
                    portfolio_result.risk_metrics.annualized_return -
                    benchmark_result.risk_metrics.annualized_return
                )
            },
            outperformance_probability=outperformance_prob,
            data_source=portfolio_result.data_source
        )
        
    except Exception as e:
        logger.error(f"Risk comparison failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Risk comparison failed",
                "message": str(e),
                "type": "service_error"
            }
        )

@router.get("/cache/stats")
async def get_cache_statistics(
    risk_service: RiskAnalysisService = Depends(get_risk_service_dependency)
):
    """Get cache statistics for risk analysis service"""
    try:
        # This would need to be implemented in the service
        return {
            "message": "Cache statistics endpoint",
            "note": "Implementation needed in risk service"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cache/clear") 
async def clear_cache(
    risk_service: RiskAnalysisService = Depends(get_risk_service_dependency)
):
    """Clear the risk analysis cache"""
    try:
        risk_service.clear_cache()
        return {
            "message": "Cache cleared successfully",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Backwards compatibility endpoint
@router.post("/legacy/analyze")
async def legacy_analyze_risk(
    request_data: Dict[str, Any],
    risk_service: RiskAnalysisService = Depends(get_risk_service_dependency)
):
    """
    Legacy risk analysis endpoint for backwards compatibility
    
    Accepts raw dictionary input and converts to proper request model.
    This endpoint provides compatibility with existing integrations.
    """
    try:
        logger.info("Legacy risk analysis request received")
        
        # Convert legacy request to new format
        try:
            # Handle legacy field names
            symbols = request_data.get("symbols", [])
            weights = request_data.get("weights")
            portfolio_id = request_data.get("portfolio_id") 
            period = request_data.get("period", "1year")
            use_real_data = request_data.get("use_real_data", True)
            include_stress_testing = request_data.get("include_stress_testing", True)
            
            # Create validated request
            validated_request = RiskAnalysisRequest(
                symbols=symbols,
                weights=weights,
                portfolio_id=portfolio_id,
                period=period,
                use_real_data=use_real_data,
                include_stress_testing=include_stress_testing
            )
            
        except Exception as validation_error:
            logger.error(f"Legacy request validation failed: {validation_error}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid request format",
                    "message": str(validation_error),
                    "suggestion": "Use /api/risk/analyze endpoint with proper request model"
                }
            )
        
        # Perform analysis using validated request
        result = await risk_service.analyze_portfolio_risk(validated_request)
        
        logger.info("Legacy risk analysis completed")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Legacy risk analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Legacy analysis failed", 
                "message": str(e)
            }
        )

# Router metadata
router.tags = ["Risk Analysis"]
router.prefix = "/api/risk"