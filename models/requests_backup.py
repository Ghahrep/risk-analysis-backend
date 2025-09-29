# models/requests.py - Complete Request/Response Models for Four-Way Integration
"""
Complete Request/Response Models for Risk Analysis Backend - FINAL VERSION
=========================================================================

COMPREHENSIVE FIXES APPLIED:
1. Added ConfigDict with protected_namespaces=() to ALL models
2. Fixed missing ModelType and IntegrationDepth enums
3. Corrected validation syntax for Pydantic v2
4. Fixed enum value inconsistencies
5. Added ALL missing model classes for complete service integration
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum

# =============================================================================
# PYDANTIC V2 CONFIGURATION - CRITICAL FIX
# =============================================================================

# Global config for all models to prevent namespace conflicts
DEFAULT_CONFIG = ConfigDict(
    extra='ignore',  # Ignore unknown fields
    protected_namespaces=(),  # Disable protected namespace warnings
    use_enum_values=True,  # Use enum values in serialization
    validate_assignment=True  # Validate on assignment
)

# =============================================================================
# ENUMS AND CONSTANTS (Fixed with consistent values)
# =============================================================================
class BaseResponseModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

class AnalysisPeriod(str, Enum):
    """Analysis time periods - CORRECTED VALUES"""
    ONE_MONTH = "1month"
    THREE_MONTHS = "3months"
    SIX_MONTHS = "6months"  # FIXED: added 's'
    ONE_YEAR = "1year"
    TWO_YEARS = "2years"
    THREE_YEARS = "3years"
    FIVE_YEARS = "5years"

class OptimizationMethod(str, Enum):
    """Portfolio optimization methods"""
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    EQUAL_WEIGHT = "equal_weight"
    MAX_RETURN = "max_return"
    RISK_PARITY = "risk_parity"
    MAX_DIVERSIFICATION = "maximize_diversification"

class RiskAnalysisType(str, Enum):
    """Risk analysis types"""
    VAR_ANALYSIS = "var_analysis"
    STRESS_TESTING = "stress_testing"
    CORRELATION_ANALYSIS = "correlation_analysis"
    COMPREHENSIVE = "comprehensive"

class AnalysisDepth(str, Enum):
    """Analysis depth levels"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    DETAILED = "detailed"

class IntegrationDepth(str, Enum):
    """Integration depth for multi-service analysis"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    ADVANCED = "advanced"

class ForecastingMethod(str, Enum):
    """Available forecasting methods"""
    AUTO_ARIMA = "auto_arima"
    SIMPLE_MEAN = "simple_mean"
    REGIME_CONDITIONAL = "regime_conditional"
    MONTE_CARLO = "monte_carlo"

class VolatilityModel(str, Enum):
    """Available volatility models"""
    GARCH = "garch"
    ROLLING_VOLATILITY = "rolling_volatility"
    REGIME_SWITCHING = "regime_switching"

class RegimeMethod(str, Enum):
    """Available regime detection methods"""
    HMM = "hmm"
    VOLATILITY = "volatility_based"
    COMPREHENSIVE = "comprehensive"

class IntegrationLevel(str, Enum):
    """Integration levels for multi-service analysis"""
    SINGLE = "single"
    TWO_WAY = "two_way"
    THREE_WAY = "three_way"
    FOUR_WAY = "four_way"

class ForecastHorizon(int, Enum):
    """Supported forecast horizons in days"""
    ONE_WEEK = 7
    ONE_MONTH = 30
    THREE_MONTHS = 90
    SIX_MONTHS = 180
    ONE_YEAR = 252

class ModelType(str, Enum):
    """Forecasting model types"""
    ARIMA = "arima"
    AUTO_ARIMA = "auto_arima"
    GARCH = "garch"
    SIMPLE_MEAN = "simple_mean"
    MONTE_CARLO = "monte_carlo"

# =============================================================================
# BASE REQUEST CLASSES (Fixed with Pydantic v2 syntax)
# =============================================================================

class BaseFMPRequest(BaseModel):
    """Base class for all requests with enhanced FMP integration support"""
    model_config = DEFAULT_CONFIG
    
    symbols: Optional[List[str]] = Field(None, description="Stock symbols for analysis")
    use_real_data: bool = Field(True, description="Enable FMP real market data integration")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    
    @field_validator('symbols')
    @classmethod
    def validate_symbols(cls, v):
        if v is not None:
            if not v:
                raise ValueError("Symbols list cannot be empty")
            # Clean and validate symbols
            cleaned = []
            for symbol in v:
                clean_symbol = symbol.strip().upper()
                if len(clean_symbol) >= 1 and clean_symbol.replace('.', '').replace('-', '').isalnum():
                    cleaned.append(clean_symbol)
                else:
                    raise ValueError(f"Invalid symbol format: {symbol}")
            
            if not cleaned:
                raise ValueError("No valid symbols provided after cleaning")
            if len(cleaned) > 20:
                raise ValueError("Maximum 20 symbols allowed per request")
            return cleaned
        return v
    
    @field_validator('use_real_data')
    @classmethod
    def validate_real_data_usage(cls, v, info):
        values = info.data if hasattr(info, 'data') else {}
        if v and not values.get('symbols'):
            raise ValueError("Symbols must be provided when use_real_data is True")
        return v

class BaseAnalysisRequest(BaseFMPRequest):
    """Base class for analysis requests with common parameters"""
    analysis_depth: AnalysisDepth = Field(default=AnalysisDepth.COMPREHENSIVE, description="Depth of analysis")

# =============================================================================
# CONVERSATION MODELS (Fixed for behavioral analysis)
# =============================================================================

class ConversationMessage(BaseModel):
    """Individual conversation message for behavioral analysis"""
    model_config = DEFAULT_CONFIG
    
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        allowed_roles = ['user', 'assistant', 'system']
        if v.lower() not in allowed_roles:
            raise ValueError(f"Role must be one of: {allowed_roles}")
        return v.lower()

# =============================================================================
# RISK ANALYSIS MODELS (Fixed validation)
# =============================================================================

class RiskAnalysisRequest(BaseAnalysisRequest):
    """Comprehensive risk analysis request with enhanced validation"""
    model_config = DEFAULT_CONFIG
    
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

# =============================================================================
# PORTFOLIO MANAGEMENT MODELS (Complete set)
# =============================================================================

class PortfolioOptimizationRequest(BaseModel):
    """Request for portfolio optimization - FIXED VALIDATION"""
    model_config = DEFAULT_CONFIG
    
    portfolio_id: Optional[int] = Field(None, description="Portfolio ID (optional)")
    symbols: List[str] = Field(..., description="Stock symbols for optimization")
    period: AnalysisPeriod = Field(AnalysisPeriod.ONE_YEAR, description="Analysis period")
    optimization_method: OptimizationMethod = Field(OptimizationMethod.MAX_SHARPE, description="Optimization method")
    target_return: Optional[float] = Field(None, description="Target return for optimization", ge=0, le=1)
    use_real_data: bool = Field(True, description="Use real FMP market data")
    risk_tolerance: Optional[float] = Field(0.5, description="Risk tolerance level", ge=0, le=1)

class PortfolioAnalysisRequest(BaseModel):
    """Portfolio analysis request"""
    model_config = DEFAULT_CONFIG
    
    portfolio_id: str = Field(..., description="Portfolio identifier")
    holdings: Dict[str, float] = Field(..., description="Portfolio holdings {symbol: weight}")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    benchmark_symbols: List[str] = Field(default=["SPY"], description="Benchmark symbols")
    use_real_data: bool = Field(default=True, description="Use real market data")

class RebalancingRequest(BaseModel):
    """Portfolio rebalancing request"""
    model_config = DEFAULT_CONFIG
    
    portfolio_id: str = Field(..., description="Portfolio identifier")
    current_holdings: Dict[str, float] = Field(..., description="Current portfolio holdings")
    target_allocation: Optional[Dict[str, float]] = Field(default=None, description="Target allocation")
    rebalancing_method: str = Field(default="threshold", description="Rebalancing method")
    threshold: float = Field(default=0.05, description="Rebalancing threshold", ge=0, le=1)
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    use_real_data: bool = Field(default=True, description="Use real market data")

class PortfolioRiskAnalysisRequest(BaseModel):
    """Portfolio risk analysis request"""
    model_config = DEFAULT_CONFIG
    
    portfolio_id: str = Field(..., description="Portfolio identifier")
    holdings: Dict[str, float] = Field(..., description="Portfolio holdings")
    benchmark_symbols: List[str] = Field(default=["SPY"], description="Benchmark symbols")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    confidence_levels: List[float] = Field(default=[0.95, 0.99], description="Risk confidence levels")
    use_real_data: bool = Field(default=True, description="Use real market data")

class FactorAnalysisRequest(BaseModel):
    """Fama-French factor analysis request"""
    model_config = DEFAULT_CONFIG
    
    symbols: List[str] = Field(..., description="Stock symbols for factor analysis")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    model_type: str = Field(default="3factor", description="Factor model type: 3factor or 5factor")
    
    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v):
        if v not in ["3factor", "5factor"]:
            raise ValueError("model_type must be '3factor' or '5factor'")
        return v

class StyleAnalysisRequest(BaseModel):
    """Style analysis request"""
    model_config = DEFAULT_CONFIG
    
    portfolio_returns: List[float] = Field(..., min_length=30, description="Portfolio return series")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    benchmark_symbol: Optional[str] = Field(default=None, description="Benchmark symbol")

class PCAAnalysisRequest(BaseModel):
    """PCA factor analysis request"""
    model_config = DEFAULT_CONFIG
    
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

class RollingFactorRequest(BaseModel):
    """Rolling factor analysis request"""
    model_config = DEFAULT_CONFIG
    
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
# BEHAVIORAL ANALYSIS MODELS (Complete set)
# =============================================================================

class BehavioralAnalysisRequest(BaseModel):
    """Comprehensive behavioral analysis request"""
    model_config = DEFAULT_CONFIG
    
    conversation_messages: List[ConversationMessage] = Field(..., min_length=1, description="Conversation history")
    symbols: Optional[List[str]] = Field(default=None, description="Portfolio symbols for FMP integration")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    use_real_data: bool = Field(default=True, description="Enable FMP integration")
    user_demographics: Optional[Dict] = Field(default=None, description="User demographic information")
    behavioral_focus: Optional[List[str]] = Field(default=None, description="Specific behavioral aspects")

class BiasDetectionRequest(BaseModel):
    """Targeted bias detection request"""
    model_config = DEFAULT_CONFIG
    
    conversation_messages: List[ConversationMessage] = Field(..., min_length=1, description="Conversation history")
    symbols: Optional[List[str]] = Field(default=None, description="Portfolio symbols for enhanced detection")
    bias_types: Optional[List[str]] = Field(default=None, description="Specific bias types to analyze")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    use_real_data: bool = Field(default=True, description="Enable FMP integration")
    sensitivity_level: str = Field(default="standard", description="Detection sensitivity")
    
    @field_validator('bias_types')
    @classmethod
    def validate_bias_types(cls, v):
        if v is not None:
            valid_types = [
                'loss_aversion', 'overconfidence', 'herding_fomo', 'anchoring',
                'confirmation', 'disposition_effect', 'mental_accounting', 'recency_bias'
            ]
            for bias_type in v:
                if bias_type not in valid_types:
                    raise ValueError(f'Bias type must be one of {valid_types}')
        return v

class SentimentAnalysisRequest(BaseModel):
    """Market sentiment analysis request with FMP context"""
    model_config = DEFAULT_CONFIG
    
    conversation_messages: List[ConversationMessage] = Field(..., min_length=1, description="Conversation history")
    symbols: Optional[List[str]] = Field(default=None, description="Portfolio symbols for market context")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    use_real_data: bool = Field(default=True, description="Enable FMP market context")
    time_window: Optional[int] = Field(None, ge=1, le=100, description="Number of recent messages")
    sentiment_model: str = Field(default="behavioral_finance", description="Sentiment analysis model")

class BehavioralProfileRequest(BaseModel):
    """Behavioral profile assessment request"""
    model_config = DEFAULT_CONFIG
    
    conversation_messages: List[ConversationMessage] = Field(..., min_length=1, description="Conversation history")
    profile_depth: str = Field(default="standard", description="Profiling depth")
    assessment_type: str = Field(default="comprehensive", description="Type of assessment")
    symbols: Optional[List[str]] = Field(default=None, description="Portfolio symbols")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    use_real_data: bool = Field(default=True, description="Enable FMP integration")

class PortfolioContextRequest(BaseModel):
    """Portfolio context analysis request"""
    model_config = DEFAULT_CONFIG
    
    conversation_messages: List[ConversationMessage] = Field(..., min_length=1, description="Conversation history")
    portfolio_holdings: Optional[Dict[str, float]] = Field(default=None, description="Portfolio holdings")
    symbols: Optional[List[str]] = Field(default=None, description="Portfolio symbols")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    use_real_data: bool = Field(default=True, description="Enable FMP integration")

class RiskAttributionRequest(BaseResponseModel):
    """Request model for portfolio risk attribution analysis"""
    symbols: List[str] = Field(..., description="List of stock symbols", example=["AAPL", "GOOGL", "MSFT"])
    weights: List[float] = Field(..., description="Portfolio weights (must sum to 1.0)", example=[0.33, 0.33, 0.34])
    factor_model: str = Field(default="fama_french_3", description="Factor model to use", example="fama_french_3")
    period: str = Field(default="1year", description="Analysis period", example="1year")
    use_real_data: bool = Field(default=True, description="Use real market data if available")

class PerformanceAttributionRequest(BaseResponseModel):
    """Request model for portfolio performance attribution analysis"""
    symbols: List[str] = Field(..., description="List of stock symbols", example=["AAPL", "GOOGL", "MSFT"])
    weights: List[float] = Field(..., description="Portfolio weights (must sum to 1.0)", example=[0.33, 0.33, 0.34])
    benchmark: str = Field(default="SPY", description="Benchmark symbol", example="SPY")
    factor_model: str = Field(default="fama_french_3", description="Factor model to use", example="fama_french_3")
    period: str = Field(default="1year", description="Analysis period", example="1year")
    use_real_data: bool = Field(default=True, description="Use real market data if available")

class AdvancedAnalyticsRequest(BaseResponseModel):
    """Request model for advanced portfolio analytics"""
    symbols: List[str] = Field(..., description="List of stock symbols", example=["AAPL", "GOOGL", "MSFT"])
    weights: List[float] = Field(..., description="Portfolio weights (must sum to 1.0)", example=[0.33, 0.33, 0.34])
    period: str = Field(default="1year", description="Analysis period", example="1year")
    use_real_data: bool = Field(default=True, description="Use real market data if available")

class CorrelationAnalysisRequest(BaseResponseModel):
    """Request model for correlation analysis"""
    symbols: List[str] = Field(..., description="List of stock symbols", example=["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"])
    period: str = Field(default="1year", description="Analysis period", example="1year")
    use_real_data: bool = Field(default=True, description="Use real market data if available")
# =============================================================================
# FORECASTING ANALYSIS MODELS (Complete set)
# =============================================================================

class ForecastingRequest(BaseModel):
    """Request for financial forecasting analysis"""
    model_config = DEFAULT_CONFIG
    
    symbols: List[str] = Field(..., description="Stock symbols to forecast")
    forecast_horizon: int = Field(30, description="Number of days to forecast", ge=1, le=365)
    period: AnalysisPeriod = Field(AnalysisPeriod.ONE_YEAR, description="Historical data period")
    include_confidence_intervals: bool = Field(True, description="Include confidence intervals")
    use_real_data: bool = Field(True, description="Use real FMP data")
    model_type: ModelType = Field(ModelType.ARIMA, description="Forecasting model type")

class ComprehensiveForecastRequest(BaseAnalysisRequest):
    """Comprehensive forecasting request combining all components"""
    model_config = DEFAULT_CONFIG
    
    forecast_horizon: ForecastHorizon = Field(default=ForecastHorizon.THREE_MONTHS, description="Forecast horizon")
    include_returns: bool = Field(default=True, description="Include return forecasting")
    include_volatility: bool = Field(default=True, description="Include volatility forecasting")
    include_regimes: bool = Field(default=True, description="Include regime analysis")
    include_scenarios: bool = Field(default=True, description="Include scenario analysis")

class FourWayIntegratedAnalysisRequest(BaseAnalysisRequest):
    """Complete four-way integrated analysis request"""
    model_config = DEFAULT_CONFIG
    
    conversation_messages: List[ConversationMessage] = Field(..., min_length=1, description="Conversation history")
    portfolio_request: Optional[Dict] = Field(default=None, description="Portfolio analysis parameters")
    forecast_horizon: ForecastHorizon = Field(default=ForecastHorizon.THREE_MONTHS, description="Forecast horizon")
    integration_depth: AnalysisDepth = Field(default=AnalysisDepth.COMPREHENSIVE, description="Integration depth")

class ReturnForecastRequest(BaseModel):
    """Return forecasting request"""
    model_config = DEFAULT_CONFIG
    
    symbols: List[str] = Field(..., description="Stock symbols to forecast")
    forecast_horizon: int = Field(30, description="Forecast horizon in days", ge=1, le=365)
    period: AnalysisPeriod = Field(AnalysisPeriod.ONE_YEAR, description="Historical data period")
    model_type: ModelType = Field(ModelType.AUTO_ARIMA, description="Forecasting model")
    confidence_level: float = Field(0.95, description="Confidence level", ge=0.5, le=0.999)
    use_real_data: bool = Field(True, description="Use real FMP data")

class VolatilityForecastRequest(BaseModel):
    """Volatility forecasting request"""
    model_config = DEFAULT_CONFIG
    
    symbols: List[str] = Field(..., description="Stock symbols for volatility forecasting")
    forecast_horizon: int = Field(30, description="Forecast horizon in days", ge=1, le=365)
    period: AnalysisPeriod = Field(AnalysisPeriod.ONE_YEAR, description="Historical data period")
    volatility_model: VolatilityModel = Field(VolatilityModel.GARCH, description="Volatility model")
    confidence_level: float = Field(0.95, description="Confidence level", ge=0.5, le=0.999)
    use_real_data: bool = Field(True, description="Use real FMP data")

class PortfolioSummaryRequest(BaseModel):
    """Portfolio summary request"""
    model_config = DEFAULT_CONFIG
    
    portfolio_id: str = Field(..., description="Portfolio identifier")
    symbols: Optional[List[str]] = Field(default=None, description="Portfolio symbols")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    include_performance: bool = Field(default=True, description="Include performance metrics")
    use_real_data: bool = Field(default=True, description="Use real market data")

class ScenarioAnalysisRequest(BaseModel):
    """Scenario analysis request"""
    model_config = DEFAULT_CONFIG
    
    symbols: List[str] = Field(..., description="Stock symbols for scenario analysis")
    scenarios: Optional[List[str]] = Field(default=None, description="Specific scenarios to analyze")
    period: AnalysisPeriod = Field(AnalysisPeriod.ONE_YEAR, description="Historical data period")
    num_simulations: int = Field(1000, description="Number of Monte Carlo simulations", ge=100, le=10000)
    confidence_level: float = Field(0.95, description="Confidence level", ge=0.5, le=0.999)
    use_real_data: bool = Field(True, description="Use real FMP data")

class RegimeAnalysisResponse(BaseModel):
    """Regime analysis response"""
    model_config = DEFAULT_CONFIG
    
    success: bool = Field(..., description="Analysis success status")
    current_regime: Optional[str] = Field(default=None, description="Current market regime")
    regime_probabilities: Optional[Dict[str, float]] = Field(default=None, description="Regime probabilities")
    transition_matrix: Optional[List[List[float]]] = Field(default=None, description="Regime transition matrix")
    regime_characteristics: Optional[Dict] = Field(default=None, description="Characteristics of each regime")
    execution_time: float = Field(..., description="Execution time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class AdvancedAnalyticsErrorResponse(BaseResponseModel):
    """Error response for advanced analytics endpoints"""
    status: str = Field(default="error", description="Response status")
    message: str = Field(..., description="Error message")
    fallback_used: bool = Field(default=False, description="Whether fallback data was used")
    error_code: Optional[str] = Field(None, description="Specific error code")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Error timestamp")
# =============================================================================
# RESPONSE MODELS (Fixed with consistent structure)
# =============================================================================

class BaseAnalysisResponse(BaseModel):
    """Base response model with consistent structure"""
    model_config = DEFAULT_CONFIG
    
    success: bool = Field(..., description="Analysis success status")
    message: str = Field(..., description="Response message")
    data_source: str = Field(..., description="Data source used")
    execution_time: float = Field(..., description="Execution time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class RiskAnalysisResponse(BaseAnalysisResponse):
    """Risk analysis response"""
    risk_metrics: Optional[Dict] = Field(default=None, description="Risk metrics")
    value_at_risk: Optional[Dict] = Field(default=None, description="VaR calculations")
    stress_test_results: Optional[Dict] = Field(default=None, description="Stress test results")
    risk_insights: List[str] = Field(default_factory=list, description="Risk insights")

class BehavioralAnalysisResponse(BaseAnalysisResponse):
    """Behavioral analysis response"""
    bias_count: int = Field(default=0, description="Number of biases detected")
    overall_risk_score: float = Field(default=0.0, description="Overall behavioral risk score")
    detected_biases: List[Dict] = Field(default_factory=list, description="Detected biases")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")

class PortfolioAnalysisResponse(BaseAnalysisResponse):
    """Portfolio analysis response"""
    optimization_result: Optional[Dict] = Field(default=None, description="Optimization results")
    trade_analysis: Optional[Dict] = Field(default=None, description="Trade analysis")
    rebalancing_needed: Optional[bool] = Field(default=None, description="Rebalancing needed flag")

class ForecastingAnalysisResponse(BaseAnalysisResponse):
    """Forecasting analysis response"""
    forecasts: Optional[Dict] = Field(default=None, description="Forecast results")
    regime_analysis: Optional[Dict] = Field(default=None, description="Regime analysis")
    scenario_results: Optional[Dict] = Field(default=None, description="Scenario analysis")

class ForecastingResponse(BaseModel):
    """Response from forecasting analysis"""
    model_config = DEFAULT_CONFIG
    
    success: bool
    data_source: str
    forecast_data: Dict[str, Any]
    confidence_intervals: Optional[Dict[str, Any]] = None
    model_accuracy: Optional[float] = None
    execution_time: float
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Service health response"""
    model_config = DEFAULT_CONFIG
    
    service: str = Field(..., description="Service name")
    status: str = Field(..., description="Health status")
    version: str = Field(default="2.0", description="Service version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    error: Optional[str] = Field(default=None, description="Error message if unhealthy")

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

class RiskAttributionResponse(BaseResponseModel):
    """Complete risk attribution response"""
    status: str = Field(..., description="Response status")
    risk_attribution: RiskAttributionData = Field(..., description="Risk attribution analysis results")
    metadata: RiskAttributionMetadata = Field(..., description="Analysis metadata")

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

class PerformanceAttributionResponse(BaseResponseModel):
    """Complete performance attribution response"""
    status: str = Field(..., description="Response status")
    performance_attribution: PerformanceAttributionData = Field(..., description="Performance attribution results")
    metadata: PerformanceAttributionMetadata = Field(..., description="Analysis metadata")

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

class AdvancedAnalyticsResponse(BaseResponseModel):
    """Complete advanced analytics response"""
    status: str = Field(..., description="Response status")
    advanced_analytics: AdvancedAnalyticsData = Field(..., description="Advanced analytics results")
    metadata: AdvancedAnalyticsMetadata = Field(..., description="Analysis metadata")

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

class CorrelationAnalysisResponse(BaseResponseModel):
    """Complete correlation analysis response"""
    status: str = Field(..., description="Response status")
    correlation_analysis: CorrelationAnalysisData = Field(..., description="Correlation analysis results")
    metadata: CorrelationAnalysisMetadata = Field(..., description="Analysis metadata")

# =============================================================================
# EXAMPLE REQUESTS FOR TESTING (Updated with correct enum values)
# =============================================================================

EXAMPLE_REQUESTS = {
    'risk_analysis': {
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "use_real_data": True,
        "period": "1year",  # Correct enum value
        "confidence_levels": [0.95, 0.99],
        "analysis_depth": "comprehensive",
        "include_stress_testing": True
    },
    'behavioral_analysis': {
        "conversation_messages": [
            {"role": "user", "content": "I'm worried about my tech stocks losing value"},
            {"role": "assistant", "content": "I understand your concern about tech stock volatility"},
            {"role": "user", "content": "Should I sell everything and wait for a crash?"}
        ],
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "use_real_data": True,
        "period": "1year",  # Correct enum value
        "analysis_depth": "comprehensive"
    },
    'portfolio_optimization': {
        "portfolio_id": 1,
        "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN"],
        "optimization_method": "max_sharpe",  # Correct enum value
        "period": "1year",  # Correct enum value
        "use_real_data": True
    },
    'forecasting_request': {
        "symbols": ["AAPL", "GOOGL"],
        "forecast_horizon": 30,
        "period": "1year",  # Correct enum value
        "model_type": "arima",  # Correct enum value
        "use_real_data": True
    }
}

# =============================================================================
# VALIDATION HELPERS (Fixed syntax)
# =============================================================================

def validate_confidence_levels(levels: List[float]) -> List[float]:
    """Validate confidence levels for risk analysis"""
    for level in levels:
        if not 0.5 <= level <= 0.999:
            raise ValueError("Confidence levels must be between 0.5 and 0.999")
    return sorted(levels)

def validate_symbols_for_analysis(symbols: List[str]) -> List[str]:
    """Validate and clean symbols for analysis"""
    if not symbols:
        raise ValueError("Symbols list cannot be empty")
    
    cleaned_symbols = []
    for symbol in symbols:
        clean_symbol = symbol.strip().upper()
        if len(clean_symbol) >= 1 and clean_symbol.replace('.', '').replace('-', '').isalnum():
            cleaned_symbols.append(clean_symbol)
        else:
            raise ValueError(f"Invalid symbol format: {symbol}")
    
    if not cleaned_symbols:
        raise ValueError("No valid symbols provided after cleaning")
    
    return cleaned_symbols

# =============================================================================
# MODULE METADATA
# =============================================================================

__version__ = "2.2.0"
__author__ = "Risk Analysis Backend Team"
__description__ = "Complete request/response models for all four analysis services with full integration support"

# Export all models and utilities
__all__ = [
    # Enums
    "AnalysisPeriod", "OptimizationMethod", "RiskAnalysisType", "AnalysisDepth",
    "ForecastingMethod", "VolatilityModel", "RegimeMethod", "IntegrationLevel",
    "ForecastHorizon", "ModelType", "IntegrationDepth",
    
    # Base Classes
    "BaseFMPRequest", "BaseAnalysisRequest", "ConversationMessage",
    
    # Risk Models
    "RiskAnalysisRequest",
    
    # Portfolio Models
    "PortfolioOptimizationRequest", "PortfolioAnalysisRequest",
    "RebalancingRequest", "PortfolioRiskAnalysisRequest", "PortfolioSummaryRequest",
    
    # Behavioral Models  
    "BehavioralAnalysisRequest", "BiasDetectionRequest", "SentimentAnalysisRequest",
    "BehavioralProfileRequest", "PortfolioContextRequest",
    
    # Forecasting Models
    "ForecastingRequest", "ComprehensiveForecastRequest", "FourWayIntegratedAnalysisRequest",
    "ReturnForecastRequest", "VolatilityForecastRequest", "RegimeAnalysisRequest", "ScenarioAnalysisRequest",
    
    # Response Models
    "BaseAnalysisResponse", "RiskAnalysisResponse", "BehavioralAnalysisResponse",
    "PortfolioAnalysisResponse", "ForecastingAnalysisResponse", "ForecastingResponse",
    "HealthResponse", "RegimeAnalysisResponse",
    
    # Functions
    "validate_confidence_levels", "validate_symbols_for_analysis",
    
    # Constants
    "EXAMPLE_REQUESTS",
     # Request Models
    "RiskAttributionRequest",
    "PerformanceAttributionRequest", 
    "AdvancedAnalyticsRequest",
    "CorrelationAnalysisRequest",
    
    # Response Models
    "RiskAttributionResponse",
    "PerformanceAttributionResponse",
    "AdvancedAnalyticsResponse", 
    "CorrelationAnalysisResponse",
    
    # Error Models
    "AdvancedAnalyticsErrorResponse",
    
    # Component Models (for flexible usage)
    "ConcentrationMetrics",
    "TailRiskMetrics",
    "DiversificationMetrics",
    "RiskAdjustedPerformance",
    "TailRiskMeasures",
    "CorrelationPair",
    "CorrelationCluster"
]