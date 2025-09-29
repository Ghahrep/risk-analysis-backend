# models/advanced_analytics_models.py
"""
Advanced Analytics Models
=========================

Models for advanced analytics including risk attribution,
performance attribution, correlation analysis, and sophisticated
portfolio metrics with factor analysis integration.
"""

from typing import Dict, List, Optional
from pydantic import Field
from datetime import datetime
from .base_models import BaseRequestModel, BaseResponseModel

# =============================================================================
# REQUEST MODELS - Advanced Analytics
# =============================================================================

class RiskAttributionRequest(BaseRequestModel):
    """Request model for portfolio risk attribution analysis"""
    symbols: List[str] = Field(..., description="List of stock symbols", example=["AAPL", "GOOGL", "MSFT"])
    weights: List[float] = Field(..., description="Portfolio weights (must sum to 1.0)", example=[0.33, 0.33, 0.34])
    factor_model: str = Field(default="fama_french_3", description="Factor model to use", example="fama_french_3")
    period: str = Field(default="1year", description="Analysis period", example="1year")
    use_real_data: bool = Field(default=True, description="Use real market data if available")

class PerformanceAttributionRequest(BaseRequestModel):
    """Request model for portfolio performance attribution analysis"""
    symbols: List[str] = Field(..., description="List of stock symbols", example=["AAPL", "GOOGL", "MSFT"])
    weights: List[float] = Field(..., description="Portfolio weights (must sum to 1.0)", example=[0.33, 0.33, 0.34])
    benchmark: str = Field(default="SPY", description="Benchmark symbol", example="SPY")
    factor_model: str = Field(default="fama_french_3", description="Factor model to use", example="fama_french_3")
    period: str = Field(default="1year", description="Analysis period", example="1year")
    use_real_data: bool = Field(default=True, description="Use real market data if available")

class AdvancedAnalyticsRequest(BaseRequestModel):
    """Request model for advanced portfolio analytics"""
    symbols: List[str] = Field(..., description="List of stock symbols", example=["AAPL", "GOOGL", "MSFT"])
    weights: List[float] = Field(..., description="Portfolio weights (must sum to 1.0)", example=[0.33, 0.33, 0.34])
    period: str = Field(default="1year", description="Analysis period", example="1year")
    use_real_data: bool = Field(default=True, description="Use real market data if available")

class CorrelationAnalysisRequest(BaseRequestModel):
    """Request model for correlation analysis"""
    symbols: List[str] = Field(..., description="List of stock symbols", example=["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"])
    period: str = Field(default="1year", description="Analysis period", example="1year")
    use_real_data: bool = Field(default=True, description="Use real market data if available")

# =============================================================================
# RESPONSE COMPONENT MODELS
# =============================================================================

class ConcentrationMetrics(BaseResponseModel):
    """Concentration risk metrics"""
    herfindahl_index: float = Field(..., description="Herfindahl concentration index")
    top_5_concentration_pct: float = Field(..., description="Top 5 holdings concentration percentage")
    top_10_concentration_pct: float = Field(..., description="Top 10 holdings concentration percentage")
    largest_position_pct: float = Field(..., description="Largest single position percentage")

class TailRiskMetrics(BaseResponseModel):
    """Tail risk and extreme scenario metrics"""
    var_99_pct: float = Field(..., description="99% Value at Risk percentage")
    cvar_99_pct: float = Field(..., description="99% Conditional Value at Risk percentage")
    stress_scenarios: Dict[str, float] = Field(..., description="Stress scenario returns")
    tail_correlation_risk: float = Field(..., description="Tail correlation risk measure")

class RiskAttributionData(BaseResponseModel):
    """Risk attribution analysis data"""
    total_risk_pct: float = Field(..., description="Total portfolio risk percentage")
    factor_contributions: Dict[str, float] = Field(..., description="Factor contributions to risk")
    systematic_risk_pct: float = Field(..., description="Systematic risk percentage")
    idiosyncratic_risk_pct: float = Field(..., description="Idiosyncratic risk percentage")
    concentration_metrics: ConcentrationMetrics = Field(..., description="Portfolio concentration metrics")
    tail_risk_metrics: TailRiskMetrics = Field(..., description="Tail risk metrics")

class RiskAttributionMetadata(BaseResponseModel):
    """Risk attribution analysis metadata"""
    data_source: str = Field(..., description="Data source used for analysis")
    analysis_date: str = Field(..., description="Date of analysis")
    symbols_analyzed: int = Field(..., description="Number of symbols analyzed")
    factor_model: str = Field(..., description="Factor model used")

class AttributionEffects(BaseResponseModel):
    """Performance attribution effects (Brinson model)"""
    selection_effect: float = Field(..., description="Security selection effect percentage")
    allocation_effect: float = Field(..., description="Asset allocation effect percentage")
    interaction_effect: float = Field(..., description="Interaction effect percentage")

class RiskAdjustedMetrics(BaseResponseModel):
    """Risk-adjusted performance metrics"""
    tracking_error: float = Field(..., description="Tracking error percentage")
    information_ratio: float = Field(..., description="Information ratio")

class PerformanceAttributionData(BaseResponseModel):
    """Performance attribution analysis data"""
    total_return_pct: float = Field(..., description="Total portfolio return percentage")
    factor_contributions: Dict[str, float] = Field(..., description="Factor contributions to performance")
    alpha_pct: float = Field(..., description="Alpha percentage")
    alpha_significance: str = Field(..., description="Alpha statistical significance")
    attribution_effects: AttributionEffects = Field(..., description="Attribution effects")
    risk_adjusted_metrics: RiskAdjustedMetrics = Field(..., description="Risk-adjusted metrics")

class PerformanceAttributionMetadata(BaseResponseModel):
    """Performance attribution analysis metadata"""
    data_source: str = Field(..., description="Data source used for analysis")
    analysis_period: str = Field(..., description="Analysis period")
    benchmark: str = Field(..., description="Benchmark used")
    alpha_tstat: float = Field(..., description="Alpha t-statistic")

class DiversificationMetrics(BaseResponseModel):
    """Portfolio diversification metrics"""
    diversification_ratio: float = Field(..., description="Diversification ratio")
    effective_num_assets: float = Field(..., description="Effective number of assets")
    avg_correlation: float = Field(..., description="Average correlation")
    correlation_clusters: int = Field(..., description="Number of correlation clusters")

class RiskAdjustedPerformance(BaseResponseModel):
    """Risk-adjusted performance measures"""
    calmar_ratio: float = Field(..., description="Calmar ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    omega_ratio: float = Field(..., description="Omega ratio")

class TailRiskMeasures(BaseResponseModel):
    """Tail risk measures"""
    var_95_pct: float = Field(..., description="95% Value at Risk percentage")
    cvar_95_pct: float = Field(..., description="95% Conditional Value at Risk percentage")
    max_drawdown_pct: float = Field(..., description="Maximum drawdown percentage")

class AdvancedAnalyticsData(BaseResponseModel):
    """Advanced analytics data"""
    diversification_metrics: DiversificationMetrics = Field(..., description="Diversification metrics")
    risk_adjusted_performance: RiskAdjustedPerformance = Field(..., description="Risk-adjusted performance")
    tail_risk_measures: TailRiskMeasures = Field(..., description="Tail risk measures")

class AdvancedAnalyticsMetadata(BaseResponseModel):
    """Advanced analytics metadata"""
    data_source: str = Field(..., description="Data source used")
    symbols_analyzed: int = Field(..., description="Number of symbols analyzed")
    analysis_period: str = Field(..., description="Analysis period")

class CorrelationPair(BaseResponseModel):
    """Correlation pair information"""
    symbol1: str = Field(..., description="First symbol")
    symbol2: str = Field(..., description="Second symbol")
    correlation: float = Field(..., description="Correlation coefficient")

class CorrelationCluster(BaseResponseModel):
    """Correlation cluster information"""
    cluster_id: int = Field(..., description="Cluster identifier")
    symbols: List[str] = Field(..., description="Symbols in cluster")
    avg_correlation: float = Field(..., description="Average correlation within cluster")

class CorrelationAnalysisData(BaseResponseModel):
    """Correlation analysis data"""
    correlation_matrix: Dict[str, Dict[str, float]] = Field(..., description="Correlation matrix")
    average_correlation: float = Field(..., description="Average portfolio correlation")
    highest_correlation: CorrelationPair = Field(..., description="Highest correlation pair")
    correlation_clusters: List[CorrelationCluster] = Field(..., description="Correlation clusters")
    diversification_score: float = Field(..., description="Portfolio diversification score")

class CorrelationAnalysisMetadata(BaseResponseModel):
    """Correlation analysis metadata"""
    symbols_analyzed: int = Field(..., description="Number of symbols analyzed")
    data_source: str = Field(..., description="Data source used")
    analysis_period: str = Field(..., description="Analysis period")

# =============================================================================
# COMPLETE RESPONSE MODELS
# =============================================================================

class RiskAttributionResponse(BaseResponseModel):
    """Complete risk attribution response"""
    status: str = Field(..., description="Response status")
    risk_attribution: RiskAttributionData = Field(..., description="Risk attribution analysis results")
    metadata: RiskAttributionMetadata = Field(..., description="Analysis metadata")

class PerformanceAttributionResponse(BaseResponseModel):
    """Complete performance attribution response"""
    status: str = Field(..., description="Response status")
    performance_attribution: PerformanceAttributionData = Field(..., description="Performance attribution results")
    metadata: PerformanceAttributionMetadata = Field(..., description="Analysis metadata")

class AdvancedAnalyticsResponse(BaseResponseModel):
    """Complete advanced analytics response"""
    status: str = Field(..., description="Response status")
    advanced_analytics: AdvancedAnalyticsData = Field(..., description="Advanced analytics results")
    metadata: AdvancedAnalyticsMetadata = Field(..., description="Analysis metadata")

class CorrelationAnalysisResponse(BaseResponseModel):
    """Complete correlation analysis response"""
    status: str = Field(..., description="Response status")
    correlation_analysis: CorrelationAnalysisData = Field(..., description="Correlation analysis results")
    metadata: CorrelationAnalysisMetadata = Field(..., description="Analysis metadata")

# =============================================================================
# ERROR RESPONSE MODEL
# =============================================================================

class AdvancedAnalyticsErrorResponse(BaseResponseModel):
    """Error response for advanced analytics endpoints"""
    status: str = Field(default="error", description="Response status")
    message: str = Field(..., description="Error message")
    fallback_used: bool = Field(default=False, description="Whether fallback data was used")
    error_code: Optional[str] = Field(None, description="Specific error code")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Error timestamp")

# =============================================================================
# EXAMPLE REQUESTS FOR TESTING
# =============================================================================

ADVANCED_ANALYTICS_EXAMPLE_REQUESTS = {
    'risk_attribution': {
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "weights": [0.4, 0.35, 0.25],
        "factor_model": "fama_french_3",
        "period": "1year",
        "use_real_data": True
    },
    'performance_attribution': {
        "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN"],
        "weights": [0.25, 0.25, 0.25, 0.25],
        "benchmark": "SPY",
        "factor_model": "fama_french_5",
        "period": "2years",
        "use_real_data": True
    },
    'advanced_analytics': {
        "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
        "weights": [0.3, 0.25, 0.2, 0.15, 0.1],
        "period": "1year",
        "use_real_data": True
    },
    'correlation_analysis': {
        "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META"],
        "period": "2years",
        "use_real_data": True
    }
}

# Export all models
__all__ = [
    # Request Models
    "RiskAttributionRequest", "PerformanceAttributionRequest", 
    "AdvancedAnalyticsRequest", "CorrelationAnalysisRequest",
    
    # Response Component Models
    "ConcentrationMetrics", "TailRiskMetrics", "RiskAttributionData", "RiskAttributionMetadata",
    "AttributionEffects", "RiskAdjustedMetrics", "PerformanceAttributionData", "PerformanceAttributionMetadata",
    "DiversificationMetrics", "RiskAdjustedPerformance", "TailRiskMeasures", "AdvancedAnalyticsData", "AdvancedAnalyticsMetadata",
    "CorrelationPair", "CorrelationCluster", "CorrelationAnalysisData", "CorrelationAnalysisMetadata",
    
    # Complete Response Models
    "RiskAttributionResponse", "PerformanceAttributionResponse", 
    "AdvancedAnalyticsResponse", "CorrelationAnalysisResponse",
    
    # Error Model
    "AdvancedAnalyticsErrorResponse",
    
    # Example Data
    "ADVANCED_ANALYTICS_EXAMPLE_REQUESTS"
]