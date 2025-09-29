# models/risk_models.py
"""
Risk Analysis Models
====================

Models for risk analysis service including VaR calculations,
stress testing, volatility forecasting, and factor analysis.
"""

from typing import Dict, List, Optional
from pydantic import Field, field_validator
from enum import Enum
from .base_models import (
    BaseAnalysisRequest, BaseAnalysisResponse, 
    AnalysisPeriod, AnalysisDepth
)

# =============================================================================
# RISK-SPECIFIC ENUMS
# =============================================================================

class RiskAnalysisType(str, Enum):
    """Risk analysis types"""
    VAR_ANALYSIS = "var_analysis"
    STRESS_TESTING = "stress_testing"
    CORRELATION_ANALYSIS = "correlation_analysis"
    COMPREHENSIVE = "comprehensive"

class VolatilityModel(str, Enum):
    """Available volatility models"""
    GARCH = "garch"
    ROLLING_VOLATILITY = "rolling_volatility"
    REGIME_SWITCHING = "regime_switching"

# =============================================================================
# REQUEST MODELS - Risk Analysis
# =============================================================================

class RiskAnalysisRequest(BaseAnalysisRequest):
    """Comprehensive risk analysis request with enhanced validation"""
    portfolio_id: Optional[str] = Field(None, description="Portfolio identifier")
    confidence_levels: List[float] = Field(default=[0.95, 0.99], description="VaR confidence levels")
    risk_analysis_type: RiskAnalysisType = Field(default=RiskAnalysisType.COMPREHENSIVE, description="Type of risk analysis")
    benchmark_symbols: Optional[List[str]] = Field(default=["SPY"], description="Benchmark symbols")
    include_stress_testing: bool = Field(default=True, description="Include stress testing scenarios")
    include_correlation_analysis: bool = Field(default=True, description="Include correlation analysis")
    include_regime_analysis: bool = Field(default=True, description="Include regime-conditional analysis")
    analysis_horizon_days: int = Field(default=252, ge=30, le=1000, description="Analysis horizon in days")
    
    @field_validator('confidence_levels')
    @classmethod
    def validate_confidence_levels(cls, v):
        if not v:
            return [0.95, 0.99]
        for level in v:
            if not 0.5 <= level <= 0.999:
                raise ValueError("Confidence levels must be between 0.5 and 0.999")
        return sorted(list(set(v)))

class VaRAnalysisRequest(BaseAnalysisRequest):
    """Value at Risk analysis request"""
    symbols: List[str] = Field(..., description="Stock symbols for VaR analysis")
    confidence_levels: List[float] = Field(default=[0.95, 0.99], description="VaR confidence levels")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    portfolio_weights: Optional[List[float]] = Field(None, description="Portfolio weights")
    use_real_data: bool = Field(default=True, description="Use real FMP data")

class StressTestRequest(BaseAnalysisRequest):
    """Stress testing request"""
    symbols: List[str] = Field(..., description="Stock symbols for stress testing")
    scenarios: Optional[List[str]] = Field(default=None, description="Specific stress scenarios")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    portfolio_weights: Optional[List[float]] = Field(None, description="Portfolio weights")
    use_real_data: bool = Field(default=True, description="Use real FMP data")

class VolatilityForecastRequest(BaseAnalysisRequest):
    """Volatility forecasting request"""
    symbols: List[str] = Field(..., description="Stock symbols for volatility forecasting")
    forecast_horizon: int = Field(30, description="Forecast horizon in days", ge=1, le=365)
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Historical data period")
    volatility_model: VolatilityModel = Field(default=VolatilityModel.GARCH, description="Volatility model")
    confidence_level: float = Field(0.95, description="Confidence level", ge=0.5, le=0.999)
    use_real_data: bool = Field(default=True, description="Use real FMP data")

class FactorAnalysisRequest(BaseAnalysisRequest):
    """Fama-French factor analysis request"""
    symbols: List[str] = Field(..., description="Stock symbols for factor analysis")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    model_type: str = Field(default="3factor", description="Factor model type: 3factor or 5factor")
    
    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v):
        if v not in ["3factor", "5factor"]:
            raise ValueError("model_type must be '3factor' or '5factor'")
        return v

class StyleAnalysisRequest(BaseAnalysisRequest):
    """Style analysis request"""
    portfolio_returns: List[float] = Field(..., min_length=30, description="Portfolio return series")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    benchmark_symbol: Optional[str] = Field(default=None, description="Benchmark symbol")

class PCAAnalysisRequest(BaseAnalysisRequest):
    """PCA factor analysis request"""
    symbols: List[str] = Field(..., min_length=2, description="Stock symbols for PCA analysis")
    period: str = Field(default="1year", description="Analysis period")
    n_components: int = Field(default=5, ge=2, le=10, description="Number of principal components")
    
    @field_validator('symbols')
    @classmethod
    def validate_symbols(cls, v):
        if not v or len(v) < 2:
            raise ValueError("At least 2 symbols required for PCA analysis")
        return v
    
    @field_validator('n_components')
    @classmethod
    def validate_components(cls, v, info):
        values = info.data if hasattr(info, 'data') else {}
        symbols = values.get('symbols', [])
        if symbols and v > len(symbols):
            raise ValueError("Number of components cannot exceed number of symbols")
        return v

class RollingFactorRequest(BaseAnalysisRequest):
    """Rolling factor analysis request"""
    symbol: str = Field(..., description="Stock symbol for rolling analysis")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    window_days: int = Field(default=60, ge=30, le=252, description="Rolling window size in days")
    model_type: str = Field(default="3factor", description="Factor model type")
    
    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v):
        if v not in ["3factor", "5factor"]:
            raise ValueError("model_type must be '3factor' or '5factor'")
        return v

# =============================================================================
# RESPONSE MODELS - Risk Analysis
# =============================================================================

class RiskAnalysisResponse(BaseAnalysisResponse):
    """Risk analysis response"""
    risk_metrics: Optional[Dict] = Field(default=None, description="Risk metrics")
    value_at_risk: Optional[Dict] = Field(default=None, description="VaR calculations")
    stress_test_results: Optional[Dict] = Field(default=None, description="Stress test results")
    risk_insights: List[str] = Field(default_factory=list, description="Risk insights")

# =============================================================================
# EXAMPLE REQUESTS FOR TESTING
# =============================================================================

RISK_EXAMPLE_REQUESTS = {
    'risk_analysis': {
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "use_real_data": True,
        "period": "1year",
        "confidence_levels": [0.95, 0.99],
        "analysis_depth": "comprehensive",
        "include_stress_testing": True
    },
    'var_analysis': {
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "confidence_levels": [0.95, 0.99],
        "portfolio_weights": [0.4, 0.35, 0.25],
        "period": "1year",
        "use_real_data": True
    },
    'stress_test': {
        "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA"],
        "scenarios": ["market_crash", "interest_rate_shock", "covid_scenario"],
        "portfolio_weights": [0.25, 0.25, 0.25, 0.25],
        "period": "2years"
    },
    'volatility_forecast': {
        "symbols": ["AAPL", "GOOGL"],
        "forecast_horizon": 30,
        "volatility_model": "garch",
        "period": "1year",
        "confidence_level": 0.95
    },
    'factor_analysis': {
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "model_type": "3factor",
        "period": "1year"
    }
}

# Export all models
__all__ = [
    # Enums
    "RiskAnalysisType", "VolatilityModel",
    
    # Request Models
    "RiskAnalysisRequest", "VaRAnalysisRequest", "StressTestRequest", 
    "VolatilityForecastRequest", "FactorAnalysisRequest", "StyleAnalysisRequest",
    "PCAAnalysisRequest", "RollingFactorRequest",
    
    # Response Models
    "RiskAnalysisResponse",
    
    # Example Data
    "RISK_EXAMPLE_REQUESTS"
]