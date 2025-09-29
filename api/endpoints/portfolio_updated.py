# api/endpoints/portfolio_updated.py - Updated with Centralized Models
"""
Portfolio Management API Endpoints - Updated with Centralized Models
===================================================================

Fixed API endpoints using centralized models/requests.py structure.
Eliminates import path issues and inline model definitions.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

# Import centralized models (clean import structure)
from models.requests import (
    PortfolioOptimizationRequest, RebalancingRequest, 
    PortfolioRiskAnalysisRequest, PortfolioSummaryRequest,
    PortfolioAnalysisResponse, HealthResponse
)

# Import service (clean path)
from services.portfolio_service_fixed import PortfolioManagementService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/portfolio", tags=["Portfolio Management"])

# =============================================================================
# MOCK DEPENDENCIES (for testing when real dependencies unavailable)
# =============================================================================

async def mock_get_current_user():
    """Mock user for testing"""
    return {"user_id": "test_user", "email": "test@example.com"}

async def mock_check_api_limits():
    """Mock API limits check"""
    return {"tier": "professional", "calls_remaining": 1000}

async def mock_track_api_usage(user_id: str, endpoint: str, success: bool, execution_time: float):
    """Mock usage tracking"""
    logger.info(f"API Usage: {user_id} called {endpoint}, success={success}, time={execution_time:.2f}s")

# Simple dependency functions (no complex dependency injection)
async def get_current_user():
    return await mock_get_current_user()

async def check_api_limits():
    return await mock_check_api_limits()

# =============================================================================
# PORTFOLIO API ENDPOINTS (Updated with centralized models)
# =============================================================================

@router.post("/optimize", response_model=PortfolioAnalysisResponse,
             summary="Portfolio Optimization",
             description="Optimize portfolio allocation with real market data")
async def comprehensive_portfolio_optimization(
    request: PortfolioOptimizationRequest,
    background_tasks: BackgroundTasks
):
    """
    Portfolio Optimization with Real Market Data using centralized models
    """
    start_time = datetime.utcnow()
    current_user = await get_current_user()
    subscription = await check_api_limits()
    
    try:
        # Validate request using centralized model validation
        if len(request.symbols) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 symbols required for portfolio optimization"
            )
        
        # Create service instance (following your proven pattern)
        portfolio_service = PortfolioManagementService(risk_free_rate=0.02, default_period=request.period.value)
        
        # Prepare service request with enum values properly extracted
        service_request = {
            'portfolio_id': request.portfolio_id,
            'symbols': request.symbols,
            'optimization_method': request.optimization_method.value,
            'period': request.period.value,
            'constraints': request.constraints,
            'current_holdings': request.current_holdings,
            'include_efficient_frontier': request.include_efficient_frontier,
            'use_real_data': request.use_real_data
        }
        
        # Call service
        optimization_response = await portfolio_service.comprehensive_portfolio_optimization(service_request)
        
        if not optimization_response.success:
            raise HTTPException(
                status_code=500,
                detail=f"Portfolio optimization failed: {optimization_response.error}"
            )
        
        # Track API usage
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        background_tasks.add_task(
            mock_track_api_usage,
            user_id=current_user["user_id"],
            endpoint="portfolio_optimization",
            success=True,
            execution_time=execution_time
        )
        
        # Return properly structured response using centralized model
        return PortfolioAnalysisResponse(
            success=True,
            service="portfolio_management",
            analysis_type="portfolio_optimization",
            timestamp=datetime.now().isoformat(),
            service_metadata={
                "service_name": "portfolio_management",
                "version": "2.0",
                "analysis_duration": execution_time,
                "fmp_integration_used": True,
                "data_source": optimization_response.optimization_result.data_source if optimization_response.optimization_result else "FMP"
            },
            portfolio_id=request.portfolio_id,
            symbols=request.symbols,
            data_source=optimization_response.optimization_result.data_source if optimization_response.optimization_result else "FMP",
            fmp_integration=True,
            optimization_results={
                "optimal_weights": optimization_response.optimization_result.optimal_weights,
                "expected_return": optimization_response.optimization_result.expected_return,
                "expected_volatility": optimization_response.optimization_result.expected_volatility,
                "sharpe_ratio": optimization_response.optimization_result.sharpe_ratio,
                "convergence_status": optimization_response.optimization_result.convergence_status
            } if optimization_response.optimization_result else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Track failed API call
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        background_tasks.add_task(
            mock_track_api_usage,
            user_id=current_user.get("user_id", "unknown"),
            endpoint="portfolio_optimization", 
            success=False,
            execution_time=execution_time
        )
        
        logger.error(f"Portfolio optimization error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Portfolio optimization error: {str(e)}"
        )

@router.post("/rebalancing", response_model=PortfolioAnalysisResponse,
             summary="Portfolio Rebalancing",
             description="Analyze portfolio rebalancing needs with cost-benefit analysis")
async def intelligent_rebalancing_analysis(
    request: RebalancingRequest,
    background_tasks: BackgroundTasks
):
    """
    Intelligent Portfolio Rebalancing with Real Market Data
    """
    start_time = datetime.utcnow()
    current_user = await get_current_user()
    
    try:
        # Validate request
        if not request.current_holdings:
            raise HTTPException(
                status_code=400,
                detail="Current holdings required for rebalancing analysis"
            )
        
        # Create service instance
        portfolio_service = PortfolioManagementService(risk_free_rate=0.02, default_period=request.period.value)
        
        # Prepare service request
        service_request = {
            'portfolio_id': request.portfolio_id,
            'current_holdings': request.current_holdings,
            'target_allocation': request.target_allocation,
            'rebalance_threshold': request.rebalance_threshold,
            'period': request.period.value,
            'execution_preferences': request.execution_preferences,
            'use_real_data': request.use_real_data
        }
        
        # Call service
        rebalancing_response = await portfolio_service.intelligent_rebalancing_analysis(service_request)
        
        if not rebalancing_response.success:
            raise HTTPException(
                status_code=500,
                detail=f"Rebalancing analysis failed: {rebalancing_response.error}"
            )
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        background_tasks.add_task(
            mock_track_api_usage,
            user_id=current_user["user_id"],
            endpoint="portfolio_rebalancing",
            success=True,
            execution_time=execution_time
        )
        
        return PortfolioAnalysisResponse(
            success=True,
            service="portfolio_management",
            analysis_type="portfolio_rebalancing",
            timestamp=datetime.now().isoformat(),
            service_metadata={
                "service_name": "portfolio_management",
                "version": "2.0",
                "analysis_duration": execution_time,
                "fmp_integration_used": True,
                "data_source": "FMP"
            },
            portfolio_id=request.portfolio_id,
            symbols=list(request.current_holdings.keys()),
            data_source="FMP",
            fmp_integration=True,
            rebalancing_recommendations={
                "rebalancing_needed": rebalancing_response.rebalancing_needed,
                "trades": rebalancing_response.trades,
                "insights": rebalancing_response.rebalancing_insights,
                "cost_benefit": rebalancing_response.cost_benefit_analysis
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        background_tasks.add_task(
            mock_track_api_usage,
            user_id=current_user.get("user_id", "unknown"),
            endpoint="portfolio_rebalancing",
            success=False,
            execution_time=execution_time
        )
        
        logger.error(f"Rebalancing analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Rebalancing analysis error: {str(e)}"
        )

@router.post("/risk-analysis", response_model=PortfolioAnalysisResponse,
             summary="Portfolio Risk Analysis",
             description="Comprehensive risk analysis with benchmark comparisons")
async def comprehensive_risk_analysis(
    request: PortfolioRiskAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Comprehensive Risk Analysis with Real Market Data
    """
    start_time = datetime.utcnow()
    current_user = await get_current_user()
    
    try:
        # Validate request
        if not request.holdings:
            raise HTTPException(
                status_code=400,
                detail="Portfolio holdings required for risk analysis"
            )
        
        # Create service instance
        portfolio_service = PortfolioManagementService(risk_free_rate=0.02, default_period=request.period.value)
        
        # Prepare service request
        service_request = {
            'portfolio_id': request.portfolio_id,
            'holdings': request.holdings,
            'period': request.period.value,
            'benchmark_symbols': request.benchmark_symbols,
            'use_real_data': request.use_real_data
        }
        
        # Call service
        risk_response = await portfolio_service.comprehensive_risk_analysis(service_request)
        
        if not risk_response.success:
            raise HTTPException(
                status_code=500,
                detail=f"Risk analysis failed: {risk_response.error}"
            )
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        background_tasks.add_task(
            mock_track_api_usage,
            user_id=current_user["user_id"],
            endpoint="portfolio_risk_analysis",
            success=True,
            execution_time=execution_time
        )
        
        return PortfolioAnalysisResponse(
            success=True,
            service="portfolio_management",
            analysis_type="portfolio_risk_analysis",
            timestamp=datetime.now().isoformat(),
            service_metadata={
                "service_name": "portfolio_management",
                "version": "2.0",
                "analysis_duration": execution_time,
                "fmp_integration_used": True,
                "data_source": "FMP"
            },
            portfolio_id=request.portfolio_id,
            symbols=list(request.holdings.keys()),
            data_source="FMP",
            fmp_integration=True,
            optimization_results=None,  # Not applicable for risk analysis
            rebalancing_recommendations=None  # Not applicable for risk analysis
        )
        
    except HTTPException:
        raise
    except Exception as e:
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        background_tasks.add_task(
            mock_track_api_usage,
            user_id=current_user.get("user_id", "unknown"),
            endpoint="portfolio_risk_analysis",
            success=False,
            execution_time=execution_time
        )
        
        logger.error(f"Risk analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Risk analysis error: {str(e)}"
        )

@router.post("/summary", 
             summary="Portfolio Summary",
             description="Quick portfolio summary with diversification metrics")
async def portfolio_summary_analysis(request: PortfolioSummaryRequest):
    """
    Portfolio Summary with Real Holdings Data
    """
    try:
        # Validate request
        if not request.holdings:
            raise HTTPException(
                status_code=400,
                detail="Portfolio holdings required for summary"
            )
        
        # Calculate summary metrics
        total_value = sum(request.holdings.values())
        holdings_count = len(request.holdings)
        
        # Calculate weights and details
        holdings_details = []
        for symbol, value in request.holdings.items():
            weight = value / total_value
            holdings_details.append({
                "symbol": symbol,
                "value": float(value),
                "weight": float(weight),
                "weight_pct": float(weight * 100)
            })
        
        # Sort by value
        holdings_details.sort(key=lambda x: x["value"], reverse=True)
        
        # Calculate concentration metrics
        weights = [h["weight"] for h in holdings_details]
        top_5_weight = sum(weights[:5]) if len(weights) >= 5 else sum(weights)
        
        # Herfindahl index for diversification
        herfindahl_index = sum(w**2 for w in weights)
        effective_number_stocks = 1 / herfindahl_index if herfindahl_index > 0 else 0
        
        return {
            "success": True,
            "portfolio_id": request.portfolio_id,
            "summary": {
                "portfolio_name": f"Portfolio {request.portfolio_id}",
                "total_value": float(total_value),
                "holdings_count": holdings_count,
                "holdings": holdings_details,
                "concentration_metrics": {
                    "top_5_weight_pct": float(top_5_weight * 100),
                    "largest_position_pct": holdings_details[0]["weight_pct"] if holdings_details else 0.0,
                    "herfindahl_index": float(herfindahl_index),
                    "effective_number_stocks": float(effective_number_stocks)
                },
                "diversification_score": min(100.0, effective_number_stocks * 10),
                "risk_assessment": {
                    "concentration_risk": "High" if top_5_weight > 0.6 else ("Medium" if top_5_weight > 0.4 else "Low"),
                    "diversification_level": "High" if effective_number_stocks > 10 else ("Medium" if effective_number_stocks > 5 else "Low")
                }
            },
            "as_of": datetime.utcnow().isoformat(),
            "response_type": "portfolio_summary"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Portfolio summary error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Portfolio summary error: {str(e)}"
        )

@router.get("/health", response_model=HealthResponse,
            summary="Service Health Check",
            description="Check portfolio management service health")
async def portfolio_service_health():
    """
    Health check for portfolio management service
    """
    try:
        # Test service availability
        portfolio_service = PortfolioManagementService()
        
        return HealthResponse(
            service="portfolio_management",
            status="healthy",
            version="2.0",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        return HealthResponse(
            service="portfolio_management",
            status="unhealthy",
            version="2.0",
            timestamp=datetime.now(),
            error=str(e)
        )

@router.get("/capabilities",
            summary="Service Capabilities",
            description="Get portfolio management service capabilities")
async def get_portfolio_capabilities():
    """
    Get portfolio management service capabilities
    """
    return {
        "service": "portfolio_management",
        "version": "2.0",
        "features_available": [
            "portfolio_optimization_with_real_data",
            "intelligent_rebalancing",
            "comprehensive_risk_analysis", 
            "portfolio_summary"
        ],
        "optimization_methods": [
            "max_sharpe",
            "min_variance",
            "equal_weight",
            "max_return",
            "risk_parity"
        ],
        "data_integration": {
            "fmp_enabled": True,
            "fallback_available": True,
            "cache_enabled": True
        },
        "supported_periods": ["1month", "3months", "6months", "1year", "2years", "3years", "5years"],
        "timestamp": datetime.now().isoformat()
    }

@router.get("/status",
            summary="Quick Status Check",
            description="Quick status check for portfolio service")
async def get_portfolio_status():
    """
    Quick status check for portfolio service
    """
    try:
        portfolio_service = PortfolioManagementService()
        
        return {
            "service": "portfolio_management",
            "status": "healthy",
            "fmp_integration": True,
            "optimization_available": True,
            "rebalancing_available": True,
            "risk_analysis_available": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Status endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving status: {str(e)}"
        )