# models/portfolio_models.py
"""
Portfolio Management Models
===========================

Models for portfolio optimization, analysis, rebalancing,
and performance attribution with FMP integration.
"""

from typing import Dict, List, Optional
from pydantic import Field, field_validator
from enum import Enum
from .base_models import (
    BaseAnalysisRequest, BaseAnalysisResponse, BaseRequestModel,
    AnalysisPeriod, AnalysisDepth
)

# =============================================================================
# PORTFOLIO-SPECIFIC ENUMS
# =============================================================================

class OptimizationMethod(str, Enum):
    """Portfolio optimization methods"""
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    EQUAL_WEIGHT = "equal_weight"
    MAX_RETURN = "max_return"
    RISK_PARITY = "risk_parity"
    MAX_DIVERSIFICATION = "maximize_diversification"

# =============================================================================
# REQUEST MODELS - Portfolio Management
# =============================================================================

class PortfolioOptimizationRequest(BaseRequestModel):
    """Request for portfolio optimization"""
    portfolio_id: Optional[int] = Field(None, description="Portfolio ID (optional)")
    symbols: List[str] = Field(..., description="Stock symbols for optimization")
    period: AnalysisPeriod = Field(AnalysisPeriod.ONE_YEAR, description="Analysis period")
    optimization_method: OptimizationMethod = Field(OptimizationMethod.MAX_SHARPE, description="Optimization method")
    target_return: Optional[float] = Field(None, description="Target return for optimization", ge=0, le=1)
    use_real_data: bool = Field(True, description="Use real FMP market data")
    risk_tolerance: Optional[float] = Field(0.5, description="Risk tolerance level", ge=0, le=1)

class PortfolioAnalysisRequest(BaseRequestModel):
    """Portfolio analysis request"""
    portfolio_id: str = Field(..., description="Portfolio identifier")
    holdings: Dict[str, float] = Field(..., description="Portfolio holdings {symbol: weight}")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    benchmark_symbols: List[str] = Field(default=["SPY"], description="Benchmark symbols")
    use_real_data: bool = Field(default=True, description="Use real market data")

class RebalancingRequest(BaseRequestModel):
    """Portfolio rebalancing request"""
    portfolio_id: str = Field(..., description="Portfolio identifier")
    current_holdings: Dict[str, float] = Field(..., description="Current portfolio holdings")
    target_allocation: Optional[Dict[str, float]] = Field(default=None, description="Target allocation")
    rebalancing_method: str = Field(default="threshold", description="Rebalancing method")
    threshold: float = Field(default=0.05, description="Rebalancing threshold", ge=0, le=1)
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    use_real_data: bool = Field(default=True, description="Use real market data")

class PortfolioRiskAnalysisRequest(BaseRequestModel):
    """Portfolio risk analysis request"""
    portfolio_id: str = Field(..., description="Portfolio identifier")
    holdings: Dict[str, float] = Field(..., description="Portfolio holdings")
    benchmark_symbols: List[str] = Field(default=["SPY"], description="Benchmark symbols")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    confidence_levels: List[float] = Field(default=[0.95, 0.99], description="Risk confidence levels")
    use_real_data: bool = Field(default=True, description="Use real market data")

class PortfolioSummaryRequest(BaseRequestModel):
    """Portfolio summary request"""
    portfolio_id: str = Field(..., description="Portfolio identifier")
    symbols: Optional[List[str]] = Field(default=None, description="Portfolio symbols")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    include_performance: bool = Field(default=True, description="Include performance metrics")
    use_real_data: bool = Field(default=True, description="Use real market data")

class EfficientFrontierRequest(BaseRequestModel):
    """Efficient frontier calculation request"""
    symbols: List[str] = Field(..., description="Stock symbols for frontier calculation")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    num_portfolios: int = Field(default=100, description="Number of portfolios to generate", ge=10, le=1000)
    min_weight: float = Field(default=0.0, description="Minimum weight per asset", ge=0, le=0.5)
    max_weight: float = Field(default=1.0, description="Maximum weight per asset", ge=0.5, le=1.0)
    use_real_data: bool = Field(default=True, description="Use real FMP data")

class PortfolioBacktestRequest(BaseRequestModel):
    """Portfolio backtesting request"""
    strategy_config: Dict = Field(..., description="Strategy configuration")
    symbols: List[str] = Field(..., description="Universe of symbols")
    start_date: str = Field(..., description="Backtest start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Backtest end date (YYYY-MM-DD)")
    initial_capital: float = Field(default=100000, description="Initial capital", ge=1000)
    benchmark_symbol: str = Field(default="SPY", description="Benchmark symbol")
    rebalance_frequency: str = Field(default="monthly", description="Rebalancing frequency")
    use_real_data: bool = Field(default=True, description="Use real FMP data")

class PortfolioScreeningRequest(BaseRequestModel):
    """Portfolio screening and selection request"""
    screening_criteria: Dict = Field(..., description="Screening criteria")
    universe: Optional[List[str]] = Field(default=None, description="Symbol universe")
    max_positions: int = Field(default=20, description="Maximum number of positions", ge=1, le=100)
    sector_constraints: Optional[Dict[str, float]] = Field(default=None, description="Sector allocation constraints")
    use_real_data: bool = Field(default=True, description="Use real FMP data")

# =============================================================================
# RESPONSE MODELS - Portfolio Management
# =============================================================================

class PortfolioAnalysisResponse(BaseAnalysisResponse):
    """Portfolio analysis response"""
    optimization_result: Optional[Dict] = Field(default=None, description="Optimization results")
    trade_analysis: Optional[Dict] = Field(default=None, description="Trade analysis")
    rebalancing_needed: Optional[bool] = Field(default=None, description="Rebalancing needed flag")

class OptimizationResult(BaseRequestModel):
    """Portfolio optimization result"""
    optimal_weights: Dict[str, float] = Field(..., description="Optimal portfolio weights")
    expected_return: float = Field(..., description="Expected portfolio return")
    expected_volatility: float = Field(..., description="Expected portfolio volatility")
    sharpe_ratio: float = Field(..., description="Portfolio Sharpe ratio")
    optimization_method: str = Field(..., description="Method used for optimization")
    convergence_status: bool = Field(..., description="Whether optimization converged")

class PerformanceMetrics(BaseRequestModel):
    """Portfolio performance metrics"""
    total_return: float = Field(..., description="Total return over period")
    annualized_return: float = Field(..., description="Annualized return")
    volatility: float = Field(..., description="Return volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    calmar_ratio: float = Field(..., description="Calmar ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")

# =============================================================================
# EXAMPLE REQUESTS FOR TESTING
# =============================================================================

PORTFOLIO_EXAMPLE_REQUESTS = {
    'portfolio_optimization': {
        "portfolio_id": 1,
        "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN"],
        "optimization_method": "max_sharpe",
        "period": "1year",
        "use_real_data": True
    },
    'portfolio_analysis': {
        "portfolio_id": "test_portfolio_001",
        "holdings": {
            "AAPL": 0.3,
            "GOOGL": 0.25,
            "MSFT": 0.25,
            "AMZN": 0.2
        },
        "benchmark_symbols": ["SPY", "QQQ"],
        "period": "1year"
    },
    'rebalancing': {
        "portfolio_id": "test_portfolio_001",
        "current_holdings": {
            "AAPL": 0.35,
            "GOOGL": 0.30,
            "MSFT": 0.20,
            "AMZN": 0.15
        },
        "target_allocation": {
            "AAPL": 0.25,
            "GOOGL": 0.25,
            "MSFT": 0.25,
            "AMZN": 0.25
        },
        "threshold": 0.05
    },
    'efficient_frontier': {
        "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
        "num_portfolios": 100,
        "period": "2years",
        "min_weight": 0.05,
        "max_weight": 0.4
    },
    'portfolio_screening': {
        "screening_criteria": {
            "min_market_cap": 10000000000,
            "max_pe_ratio": 25,
            "min_dividend_yield": 0.02,
            "sectors": ["Technology", "Healthcare", "Financial Services"]
        },
        "max_positions": 15,
        "sector_constraints": {
            "Technology": 0.4,
            "Healthcare": 0.3,
            "Financial Services": 0.3
        }
    }
}

# Export all models
__all__ = [
    # Enums
    "OptimizationMethod",
    
    # Request Models
    "PortfolioOptimizationRequest", "PortfolioAnalysisRequest", "RebalancingRequest",
    "PortfolioRiskAnalysisRequest", "PortfolioSummaryRequest", "EfficientFrontierRequest",
    "PortfolioBacktestRequest", "PortfolioScreeningRequest",
    
    # Response Models
    "PortfolioAnalysisResponse", "OptimizationResult", "PerformanceMetrics",
    
    # Example Data
    "PORTFOLIO_EXAMPLE_REQUESTS"
]