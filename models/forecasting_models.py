# models/forecasting_models.py
"""
Forecasting Analysis Models
===========================

Models for financial forecasting including return prediction,
volatility forecasting, regime analysis, and scenario modeling.
"""

from typing import Dict, List, Optional, Any
from pydantic import Field, field_validator
from enum import Enum
from datetime import datetime
from .base_models import (
    BaseAnalysisRequest, BaseAnalysisResponse, BaseRequestModel, BaseResponseModel,
    AnalysisPeriod, AnalysisDepth
)
from .base_models import ConversationMessage
# =============================================================================
# FORECASTING-SPECIFIC ENUMS
# =============================================================================

class ForecastingMethod(str, Enum):
    """Available forecasting methods"""
    AUTO_ARIMA = "auto_arima"
    SIMPLE_MEAN = "simple_mean"
    REGIME_CONDITIONAL = "regime_conditional"
    MONTE_CARLO = "monte_carlo"

class ModelType(str, Enum):
    """Forecasting model types"""
    ARIMA = "arima"
    AUTO_ARIMA = "auto_arima"
    GARCH = "garch"
    SIMPLE_MEAN = "simple_mean"
    MONTE_CARLO = "monte_carlo"

class RegimeMethod(str, Enum):
    """Available regime detection methods"""
    HMM = "hmm"
    VOLATILITY = "volatility_based"
    COMPREHENSIVE = "comprehensive"

class ForecastHorizon(int, Enum):
    """Supported forecast horizons in days"""
    ONE_WEEK = 7
    ONE_MONTH = 30
    THREE_MONTHS = 90
    SIX_MONTHS = 180
    ONE_YEAR = 252

# =============================================================================
# REQUEST MODELS - Forecasting Analysis
# =============================================================================

class ForecastingRequest(BaseAnalysisRequest):
    """Request for financial forecasting analysis"""
    symbols: List[str] = Field(..., description="Stock symbols to forecast")
    forecast_horizon: int = Field(30, description="Number of days to forecast", ge=1, le=365)
    period: AnalysisPeriod = Field(AnalysisPeriod.ONE_YEAR, description="Historical data period")
    include_confidence_intervals: bool = Field(True, description="Include confidence intervals")
    use_real_data: bool = Field(True, description="Use real FMP data")
    model_type: ModelType = Field(ModelType.ARIMA, description="Forecasting model type")

class ComprehensiveForecastRequest(BaseAnalysisRequest):
    """Comprehensive forecasting request combining all components"""
    forecast_horizon: ForecastHorizon = Field(default=ForecastHorizon.THREE_MONTHS, description="Forecast horizon")
    include_returns: bool = Field(default=True, description="Include return forecasting")
    include_volatility: bool = Field(default=True, description="Include volatility forecasting")
    include_regimes: bool = Field(default=True, description="Include regime analysis")
    include_scenarios: bool = Field(default=True, description="Include scenario analysis")

class ReturnForecastRequest(BaseAnalysisRequest):
    """Return forecasting request"""
    symbols: List[str] = Field(..., description="Stock symbols to forecast")
    forecast_horizon: int = Field(30, description="Forecast horizon in days", ge=1, le=365)
    period: AnalysisPeriod = Field(AnalysisPeriod.ONE_YEAR, description="Historical data period")
    model_type: ModelType = Field(ModelType.AUTO_ARIMA, description="Forecasting model")
    confidence_level: float = Field(0.95, description="Confidence level", ge=0.5, le=0.999)
    use_real_data: bool = Field(True, description="Use real FMP data")

class VolatilityForecastRequest(BaseAnalysisRequest):
    """Volatility forecasting request"""
    symbols: List[str] = Field(..., description="Stock symbols for volatility forecasting")
    forecast_horizon: int = Field(30, description="Forecast horizon in days", ge=1, le=365)
    period: AnalysisPeriod = Field(AnalysisPeriod.ONE_YEAR, description="Historical data period")
    confidence_level: float = Field(0.95, description="Confidence level", ge=0.5, le=0.999)
    use_real_data: bool = Field(True, description="Use real FMP data")

class RegimeAnalysisRequest(BaseAnalysisRequest):
    """Regime analysis and detection request"""
    symbols: List[str] = Field(..., description="Stock symbols for regime analysis")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    regime_method: RegimeMethod = Field(default=RegimeMethod.HMM, description="Regime detection method")
    num_regimes: int = Field(default=2, description="Number of regimes to detect", ge=2, le=5)
    use_real_data: bool = Field(default=True, description="Use real FMP data")

class ScenarioAnalysisRequest(BaseAnalysisRequest):
    """Scenario analysis request"""
    symbols: List[str] = Field(..., description="Stock symbols for scenario analysis")
    scenarios: Optional[List[str]] = Field(default=None, description="Specific scenarios to analyze")
    period: AnalysisPeriod = Field(AnalysisPeriod.ONE_YEAR, description="Historical data period")
    num_simulations: int = Field(1000, description="Number of Monte Carlo simulations", ge=100, le=10000)
    confidence_level: float = Field(0.95, description="Confidence level", ge=0.5, le=0.999)
    use_real_data: bool = Field(True, description="Use real FMP data")

class FourWayIntegratedAnalysisRequest(BaseAnalysisRequest):
    """Complete four-way integrated analysis request"""
   
    conversation_messages: List[ConversationMessage] = Field(..., min_length=1, description="Conversation history")
    portfolio_request: Optional[Dict] = Field(default=None, description="Portfolio analysis parameters")
    forecast_horizon: ForecastHorizon = Field(default=ForecastHorizon.THREE_MONTHS, description="Forecast horizon")
    integration_depth: AnalysisDepth = Field(default=AnalysisDepth.COMPREHENSIVE, description="Integration depth")

class BacktestRequest(BaseAnalysisRequest):
    """Backtesting request for forecasting models"""
    symbols: List[str] = Field(..., description="Stock symbols for backtesting")
    start_date: str = Field(..., description="Backtest start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Backtest end date (YYYY-MM-DD)")
    model_configs: List[Dict] = Field(..., description="Model configurations to test")
    rebalance_frequency: str = Field(default="monthly", description="Rebalancing frequency")
    use_real_data: bool = Field(default=True, description="Use real FMP data")

# =============================================================================
# RESPONSE MODELS - Forecasting Analysis
# =============================================================================

class ForecastingAnalysisResponse(BaseAnalysisResponse):
    """Forecasting analysis response"""
    forecasts: Optional[Dict] = Field(default=None, description="Forecast results")
    regime_analysis: Optional[Dict] = Field(default=None, description="Regime analysis")
    scenario_results: Optional[Dict] = Field(default=None, description="Scenario analysis")

class ForecastingResponse(BaseResponseModel):
    """Response from forecasting analysis"""
    success: bool = Field(..., description="Analysis success status")
    data_source: str = Field(..., description="Data source used")
    forecast_data: Dict[str, Any] = Field(..., description="Forecast results")
    confidence_intervals: Optional[Dict[str, Any]] = Field(None, description="Confidence intervals")
    model_accuracy: Optional[float] = Field(None, description="Model accuracy metrics")
    execution_time: float = Field(..., description="Execution time in seconds")
    error: Optional[str] = Field(None, description="Error message if failed")

class RegimeAnalysisResponse(BaseResponseModel):
    """Regime analysis response"""
    success: bool = Field(..., description="Analysis success status")
    current_regime: Optional[str] = Field(default=None, description="Current market regime")
    regime_probabilities: Optional[Dict[str, float]] = Field(default=None, description="Regime probabilities")
    transition_matrix: Optional[List[List[float]]] = Field(default=None, description="Regime transition matrix")
    regime_characteristics: Optional[Dict] = Field(default=None, description="Characteristics of each regime")
    execution_time: float = Field(..., description="Execution time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class ForecastAccuracy(BaseRequestModel):
    """Forecast accuracy metrics"""
    mae: float = Field(..., description="Mean Absolute Error")
    mse: float = Field(..., description="Mean Squared Error")
    rmse: float = Field(..., description="Root Mean Squared Error")
    mape: float = Field(..., description="Mean Absolute Percentage Error")
    directional_accuracy: float = Field(..., description="Directional accuracy percentage")

class ScenarioResult(BaseRequestModel):
    """Scenario analysis result"""
    scenario_name: str = Field(..., description="Name of the scenario")
    probability: float = Field(..., description="Scenario probability")
    expected_return: float = Field(..., description="Expected return under scenario")
    risk_metrics: Dict[str, float] = Field(..., description="Risk metrics for scenario")
    portfolio_impact: Dict[str, float] = Field(..., description="Impact on portfolio holdings")

# =============================================================================
# EXAMPLE REQUESTS FOR TESTING
# =============================================================================

FORECASTING_EXAMPLE_REQUESTS = {
    'forecasting_request': {
        "symbols": ["AAPL", "GOOGL"],
        "forecast_horizon": 30,
        "period": "1year",
        "model_type": "arima",
        "use_real_data": True
    },
    'return_forecast': {
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "forecast_horizon": 60,
        "model_type": "auto_arima",
        "confidence_level": 0.95,
        "period": "2years"
    },
    'volatility_forecast': {
        "symbols": ["AAPL", "TSLA"],
        "forecast_horizon": 30,
        "confidence_level": 0.99,
        "period": "1year"
    },
    'regime_analysis': {
        "symbols": ["SPY", "QQQ", "IWM"],
        "regime_method": "hmm",
        "num_regimes": 3,
        "period": "3years"
    },
    'scenario_analysis': {
        "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN"],
        "scenarios": ["bull_market", "bear_market", "recession", "recovery"],
        "num_simulations": 5000,
        "confidence_level": 0.95,
        "period": "2years"
    },
    'comprehensive_forecast': {
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "forecast_horizon": 90,
        "include_returns": True,
        "include_volatility": True,
        "include_regimes": True,
        "include_scenarios": True,
        "period": "2years"
    }
}

# Export all models
__all__ = [
    # Enums
    "ForecastingMethod", "ModelType", "RegimeMethod", "ForecastHorizon",
    
    # Request Models
    "ForecastingRequest", "ComprehensiveForecastRequest", "ReturnForecastRequest",
    "VolatilityForecastRequest", "RegimeAnalysisRequest", "ScenarioAnalysisRequest",
    "FourWayIntegratedAnalysisRequest", "BacktestRequest",
    
    # Response Models
    "ForecastingAnalysisResponse", "ForecastingResponse", "RegimeAnalysisResponse",
    "ForecastAccuracy", "ScenarioResult",
    
    # Example Data
    "FORECASTING_EXAMPLE_REQUESTS"
]