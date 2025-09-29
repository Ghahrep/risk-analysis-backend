# models/__init__.py
"""
Models Package - Split Architecture
===================================

Centralized import aggregator for all model files.
Provides clean imports and handles missing modules gracefully.

Usage:
    from models import RiskAnalysisRequest, BehavioralAnalysisRequest
    from models.advanced_analytics_models import RiskAttributionRequest
    from models import BEHAVIORAL_EXAMPLE_REQUESTS
"""

from typing import Dict, Any

# Version info
__version__ = "2.4.0"
__description__ = "Split models architecture for risk analysis backend"

# =============================================================================
# IMPORT BASE MODELS (Always Available)
# =============================================================================

try:
    from .base_models import (
        BaseRequestModel, BaseResponseModel, BaseFMPRequest, BaseAnalysisRequest, BaseAnalysisResponse,
        AnalysisPeriod, AnalysisDepth, IntegrationLevel,
        ConversationMessage, HealthResponse,
        validate_confidence_levels, validate_symbols_for_analysis
    )
    BASE_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Base models not available: {e}")
    BASE_MODELS_AVAILABLE = False

# =============================================================================
# IMPORT BEHAVIORAL MODELS (Working)
# =============================================================================

try:
    from .behavioral_models import (
        BehavioralAnalysisRequest, BiasDetectionRequest, SentimentAnalysisRequest,
        BehavioralProfileRequest, PortfolioContextRequest,
        BehavioralAnalysisResponse,
        BEHAVIORAL_EXAMPLE_REQUESTS
    )
    BEHAVIORAL_MODELS_AVAILABLE = True
    print("✅ Behavioral models loaded successfully")
except ImportError as e:
    print(f"⚠️ Behavioral models not available: {e}")
    BEHAVIORAL_MODELS_AVAILABLE = False

# =============================================================================
# IMPORT RISK MODELS (Working - functions missing)
# =============================================================================

try:
    from .risk_models import (
        RiskAnalysisType, VolatilityModel,
        RiskAnalysisRequest, VaRAnalysisRequest, StressTestRequest,
        VolatilityForecastRequest, FactorAnalysisRequest, StyleAnalysisRequest,
        PCAAnalysisRequest, RollingFactorRequest,
        RiskAnalysisResponse,
        RISK_EXAMPLE_REQUESTS
    )
    RISK_MODELS_AVAILABLE = True
    print("✅ Risk models loaded successfully")
except ImportError as e:
    print(f"⚠️ Risk models not available: {e}")
    RISK_MODELS_AVAILABLE = False

# =============================================================================
# IMPORT PORTFOLIO MODELS (Working - functions missing)
# =============================================================================

try:
    from .portfolio_models import (
        OptimizationMethod,
        PortfolioOptimizationRequest, PortfolioAnalysisRequest, RebalancingRequest,
        PortfolioRiskAnalysisRequest, PortfolioSummaryRequest, EfficientFrontierRequest,
        PortfolioBacktestRequest, PortfolioScreeningRequest,
        PortfolioAnalysisResponse, OptimizationResult, PerformanceMetrics,
        PORTFOLIO_EXAMPLE_REQUESTS
    )
    PORTFOLIO_MODELS_AVAILABLE = True
    print("✅ Portfolio models loaded successfully")
except ImportError as e:
    print(f"⚠️ Portfolio models not available: {e}")
    PORTFOLIO_MODELS_AVAILABLE = False

# =============================================================================
# IMPORT FORECASTING MODELS (Working - functions missing)
# =============================================================================

try:
    from .forecasting_models import (
        ForecastingMethod, ModelType, RegimeMethod, ForecastHorizon,
        ForecastingRequest, ComprehensiveForecastRequest, ReturnForecastRequest,
        VolatilityForecastRequest, RegimeAnalysisRequest, ScenarioAnalysisRequest,
        FourWayIntegratedAnalysisRequest, BacktestRequest,
        ForecastingAnalysisResponse, ForecastingResponse, RegimeAnalysisResponse,
        ForecastAccuracy, ScenarioResult,
        FORECASTING_EXAMPLE_REQUESTS
    )
    FORECASTING_MODELS_AVAILABLE = True
    print("✅ Forecasting models loaded successfully")
except ImportError as e:
    print(f"⚠️ Forecasting models not available: {e}")
    FORECASTING_MODELS_AVAILABLE = False

# =============================================================================
# IMPORT ADVANCED ANALYTICS MODELS (Previously Broken - Now Fixed)
# =============================================================================

try:
    from .advanced_analytics_models import (
        RiskAttributionRequest, PerformanceAttributionRequest,
        AdvancedAnalyticsRequest, CorrelationAnalysisRequest,
        ConcentrationMetrics, TailRiskMetrics, RiskAttributionData, RiskAttributionMetadata,
        AttributionEffects, RiskAdjustedMetrics, PerformanceAttributionData, PerformanceAttributionMetadata,
        DiversificationMetrics, RiskAdjustedPerformance, TailRiskMeasures, AdvancedAnalyticsData, AdvancedAnalyticsMetadata,
        CorrelationPair, CorrelationCluster, CorrelationAnalysisData, CorrelationAnalysisMetadata,
        RiskAttributionResponse, PerformanceAttributionResponse,
        AdvancedAnalyticsResponse, CorrelationAnalysisResponse,
        AdvancedAnalyticsErrorResponse,
        ADVANCED_ANALYTICS_EXAMPLE_REQUESTS
    )
    ADVANCED_ANALYTICS_MODELS_AVAILABLE = True
    print("✅ Advanced analytics models loaded successfully")
except ImportError as e:
    print(f"❌ Advanced analytics models not available: {e}")
    ADVANCED_ANALYTICS_MODELS_AVAILABLE = False

# =============================================================================
# IMPORT CORRELATION MODELS (New - Enhanced Correlation Capabilities)
# =============================================================================

try:
    from .correlation_models import (
        RollingCorrelationRequest, RegimeCorrelationRequest,
        CorrelationClusteringRequest, CorrelationNetworkRequest,
        ComprehensiveCorrelationRequest,
        RollingCorrelationResponse, RegimeCorrelationResponse,
        CorrelationClusteringResponse, CorrelationNetworkResponse,
        ComprehensiveCorrelationResponse,
        StabilityMetrics, RegimeSensitivity, ClusterQualityMetrics,
        NetworkHealthMetrics, DegreeCentralityMetrics, SystemicImportanceMetrics,
        CORRELATION_EXAMPLE_REQUESTS
    )
    CORRELATION_MODELS_AVAILABLE = True
    print("✅ Correlation models loaded successfully")
except ImportError as e:
    print(f"⚠️ Correlation models not available: {e}")
    CORRELATION_MODELS_AVAILABLE = False

# =============================================================================
# SYSTEM STATUS SUMMARY
# =============================================================================

def get_models_status() -> Dict[str, Any]:
    """Get the current status of all model categories"""
    return {
        "base_models": BASE_MODELS_AVAILABLE,
        "behavioral_models": BEHAVIORAL_MODELS_AVAILABLE,
        "risk_models": RISK_MODELS_AVAILABLE,
        "portfolio_models": PORTFOLIO_MODELS_AVAILABLE,
        "forecasting_models": FORECASTING_MODELS_AVAILABLE,
        "advanced_analytics_models": ADVANCED_ANALYTICS_MODELS_AVAILABLE,
        "correlation_models": CORRELATION_MODELS_AVAILABLE,
        "total_available": sum([
            BASE_MODELS_AVAILABLE,
            BEHAVIORAL_MODELS_AVAILABLE,
            RISK_MODELS_AVAILABLE,
            PORTFOLIO_MODELS_AVAILABLE,
            FORECASTING_MODELS_AVAILABLE,
            ADVANCED_ANALYTICS_MODELS_AVAILABLE,
            CORRELATION_MODELS_AVAILABLE
        ]),
        "total_possible": 7,
        "availability_percentage": sum([
            BASE_MODELS_AVAILABLE,
            BEHAVIORAL_MODELS_AVAILABLE,
            RISK_MODELS_AVAILABLE,
            PORTFOLIO_MODELS_AVAILABLE,
            FORECASTING_MODELS_AVAILABLE,
            ADVANCED_ANALYTICS_MODELS_AVAILABLE,
            CORRELATION_MODELS_AVAILABLE
        ]) / 7 * 100
    }

# Print status summary on import
status = get_models_status()
print("\n" + "=" * 60)
print("MODELS SPLIT ARCHITECTURE STATUS")
print("=" * 60)
print(f"Available Model Categories: {status['total_available']}/{status['total_possible']} ({status['availability_percentage']:.1f}%)")
print(f"Base Models: {'✅' if BASE_MODELS_AVAILABLE else '❌'}")
print(f"Behavioral Models: {'✅' if BEHAVIORAL_MODELS_AVAILABLE else '❌'}")
print(f"Risk Models: {'✅' if RISK_MODELS_AVAILABLE else '❌'}")
print(f"Portfolio Models: {'✅' if PORTFOLIO_MODELS_AVAILABLE else '❌'}")
print(f"Forecasting Models: {'✅' if FORECASTING_MODELS_AVAILABLE else '❌'}")
print(f"Advanced Analytics Models: {'✅' if ADVANCED_ANALYTICS_MODELS_AVAILABLE else '❌'}")
print(f"Correlation Models: {'✅' if CORRELATION_MODELS_AVAILABLE else '❌'}")
print("=" * 60)

# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

# Export status function and availability flags
__all__ = [
    "get_models_status",
    "BASE_MODELS_AVAILABLE", "BEHAVIORAL_MODELS_AVAILABLE", "RISK_MODELS_AVAILABLE",
    "PORTFOLIO_MODELS_AVAILABLE", "FORECASTING_MODELS_AVAILABLE", "ADVANCED_ANALYTICS_MODELS_AVAILABLE",
    "CORRELATION_MODELS_AVAILABLE"
]

# Add available models to __all__ dynamically
if BASE_MODELS_AVAILABLE:
    __all__.extend([
        "BaseRequestModel", "BaseResponseModel", "BaseFMPRequest", "BaseAnalysisRequest", "BaseAnalysisResponse",
        "AnalysisPeriod", "AnalysisDepth", "IntegrationLevel", "ConversationMessage", "HealthResponse"
    ])

if BEHAVIORAL_MODELS_AVAILABLE:
    __all__.extend([
        "BehavioralAnalysisRequest", "BiasDetectionRequest", "SentimentAnalysisRequest",
        "BehavioralProfileRequest", "PortfolioContextRequest", "BehavioralAnalysisResponse",
        "BEHAVIORAL_EXAMPLE_REQUESTS"
    ])

if RISK_MODELS_AVAILABLE:
    __all__.extend([
        "RiskAnalysisType", "VolatilityModel", "RiskAnalysisRequest", "VaRAnalysisRequest",
        "StressTestRequest", "VolatilityForecastRequest", "FactorAnalysisRequest",
        "StyleAnalysisRequest", "PCAAnalysisRequest", "RollingFactorRequest",
        "RiskAnalysisResponse", "RISK_EXAMPLE_REQUESTS"
    ])

if PORTFOLIO_MODELS_AVAILABLE:
    __all__.extend([
        "OptimizationMethod", "PortfolioOptimizationRequest", "PortfolioAnalysisRequest",
        "RebalancingRequest", "PortfolioRiskAnalysisRequest", "PortfolioSummaryRequest",
        "EfficientFrontierRequest", "PortfolioBacktestRequest", "PortfolioScreeningRequest",
        "PortfolioAnalysisResponse", "OptimizationResult", "PerformanceMetrics",
        "PORTFOLIO_EXAMPLE_REQUESTS"
    ])

if FORECASTING_MODELS_AVAILABLE:
    __all__.extend([
        "ForecastingMethod", "ModelType", "RegimeMethod", "ForecastHorizon",
        "ForecastingRequest", "ComprehensiveForecastRequest", "ReturnForecastRequest",
        "VolatilityForecastRequest", "RegimeAnalysisRequest", "ScenarioAnalysisRequest",
        "FourWayIntegratedAnalysisRequest", "BacktestRequest",
        "ForecastingAnalysisResponse", "ForecastingResponse", "RegimeAnalysisResponse",
        "ForecastAccuracy", "ScenarioResult", "FORECASTING_EXAMPLE_REQUESTS"
    ])

if ADVANCED_ANALYTICS_MODELS_AVAILABLE:
    __all__.extend([
        "RiskAttributionRequest", "PerformanceAttributionRequest", "AdvancedAnalyticsRequest",
        "CorrelationAnalysisRequest", "RiskAttributionResponse", "PerformanceAttributionResponse",
        "AdvancedAnalyticsResponse", "CorrelationAnalysisResponse", "AdvancedAnalyticsErrorResponse",
        "ConcentrationMetrics", "TailRiskMetrics", "DiversificationMetrics", "RiskAdjustedPerformance",
        "TailRiskMeasures", "CorrelationPair", "CorrelationCluster", "ADVANCED_ANALYTICS_EXAMPLE_REQUESTS"
    ])

if CORRELATION_MODELS_AVAILABLE:
    __all__.extend([
        "RollingCorrelationRequest", "RegimeCorrelationRequest", "CorrelationClusteringRequest",
        "CorrelationNetworkRequest", "ComprehensiveCorrelationRequest",
        "RollingCorrelationResponse", "RegimeCorrelationResponse", "CorrelationClusteringResponse",
        "CorrelationNetworkResponse", "ComprehensiveCorrelationResponse",
        "StabilityMetrics", "RegimeSensitivity", "ClusterQualityMetrics",
        "NetworkHealthMetrics", "DegreeCentralityMetrics", "SystemicImportanceMetrics",
        "CORRELATION_EXAMPLE_REQUESTS"
    ])

print(f"✅ Models package initialized with {len(__all__)} available exports")