# core/exceptions.py - Complete Exception Classes for Backend Services
"""
Custom Exceptions for Risk Analysis Backend
==========================================

Following Backend Refactoring Handbook exception handling patterns
with proper inheritance hierarchy and error categorization.
"""

from typing import Optional, Dict, Any


class BackendError(Exception):
    """Base exception for all backend service errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }


class ServiceError(BackendError):
    """Base exception for service-level errors"""
    
    def __init__(self, message: str, service: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.service = service
        super().__init__(message, error_code, details)
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["service"] = self.service
        return result


class DataProviderError(BackendError):
    """Base exception for data provider errors"""
    
    def __init__(self, message: str, provider: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.provider = provider
        super().__init__(message, error_code, details)
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["provider"] = self.provider
        return result


# =============================================================================
# RISK ANALYSIS SERVICE EXCEPTIONS
# =============================================================================

class RiskAnalysisError(ServiceError):
    """Risk analysis service specific errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "risk_analysis", error_code, details)


class PortfolioDataError(RiskAnalysisError):
    """Portfolio data validation errors"""
    pass


class VaRCalculationError(RiskAnalysisError):
    """Value at Risk calculation errors"""
    pass


class StressTestingError(RiskAnalysisError):
    """Stress testing calculation errors"""
    pass


# =============================================================================
# BEHAVIORAL ANALYSIS SERVICE EXCEPTIONS
# =============================================================================

class BehavioralAnalysisError(ServiceError):
    """Behavioral analysis service specific errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "behavioral_analysis", error_code, details)


class ConversationAnalysisError(BehavioralAnalysisError):
    """Conversation analysis errors"""
    pass


class BiasDetectionError(BehavioralAnalysisError):
    """Bias detection specific errors"""
    pass


class SentimentAnalysisError(BehavioralAnalysisError):
    """Sentiment analysis specific errors"""
    pass


# =============================================================================
# PORTFOLIO MANAGEMENT SERVICE EXCEPTIONS
# =============================================================================

class PortfolioManagementError(ServiceError):
    """Portfolio management service specific errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "portfolio_management", error_code, details)


class OptimizationError(PortfolioManagementError):
    """Portfolio optimization errors"""
    pass


class RebalancingError(PortfolioManagementError):
    """Portfolio rebalancing errors"""
    pass


class PerformanceAnalysisError(PortfolioManagementError):
    """Performance analysis errors"""
    pass


# =============================================================================
# FORECASTING SERVICE EXCEPTIONS
# =============================================================================

class ForecastingError(ServiceError):
    """Forecasting service specific errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "forecasting", error_code, details)


class ModelFittingError(ForecastingError):
    """Model fitting and training errors"""
    pass


class PredictionError(ForecastingError):
    """Prediction generation errors"""
    pass


class RegimeAnalysisError(ForecastingError):
    """Regime analysis errors"""
    pass


# =============================================================================
# DATA PROVIDER EXCEPTIONS
# =============================================================================

class FMPDataError(DataProviderError):
    """FMP data provider specific errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "fmp", error_code, details)


class APIRateLimitError(DataProviderError):
    """API rate limit exceeded"""
    pass


class DataNotAvailableError(DataProviderError):
    """Requested data not available"""
    pass


class InvalidSymbolError(DataProviderError):
    """Invalid stock symbol provided"""
    pass


# =============================================================================
# CONFIGURATION AND VALIDATION EXCEPTIONS
# =============================================================================

class ConfigurationError(BackendError):
    """Configuration related errors"""
    pass


class ValidationError(BackendError):
    """Input validation errors"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None, details: Optional[Dict[str, Any]] = None):
        self.field = field
        self.value = value
        super().__init__(message, "VALIDATION_ERROR", details)
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.field:
            result["field"] = self.field
        if self.value is not None:
            result["value"] = str(self.value)
        return result


class AuthenticationError(BackendError):
    """Authentication related errors"""
    pass


class AuthorizationError(BackendError):
    """Authorization related errors"""
    pass


# =============================================================================
# INTEGRATION AND DEPENDENCY EXCEPTIONS
# =============================================================================

class ServiceUnavailableError(ServiceError):
    """Service is unavailable or not responding"""
    pass


class DependencyError(BackendError):
    """Dependency injection or initialization errors"""
    pass


class IntegrationError(BackendError):
    """Cross-service integration errors"""
    
    def __init__(self, message: str, services: list, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.services = services
        super().__init__(message, error_code, details)
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["services"] = self.services
        return result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_service_error(service_name: str, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> ServiceError:
    """Factory function to create appropriate service error"""
    service_error_map = {
        "risk_analysis": RiskAnalysisError,
        "behavioral_analysis": BehavioralAnalysisError,
        "portfolio_management": PortfolioManagementError,
        "forecasting": ForecastingError
    }
    
    error_class = service_error_map.get(service_name, ServiceError)
    
    if error_class == ServiceError:
        return ServiceError(message, service_name, error_code, details)
    else:
        return error_class(message, error_code, details)


def create_data_provider_error(provider_name: str, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> DataProviderError:
    """Factory function to create appropriate data provider error"""
    if provider_name.lower() == "fmp":
        return FMPDataError(message, error_code, details)
    else:
        return DataProviderError(message, provider_name, error_code, details)


def handle_service_exception(func):
    """Decorator to handle service exceptions consistently"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ServiceError:
            # Re-raise service errors as-is
            raise
        except Exception as e:
            # Convert generic exceptions to service errors
            service_name = getattr(func, '__self__', {}).get('service_name', 'unknown')
            raise ServiceError(f"Unexpected error in {service_name}: {str(e)}", service_name)
    
    return wrapper


# =============================================================================
# ERROR CODES CONSTANTS
# =============================================================================

class ErrorCodes:
    """Standard error codes for consistent error handling"""
    
    # General errors
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    
    # Service errors
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    SERVICE_TIMEOUT = "SERVICE_TIMEOUT"
    SERVICE_INITIALIZATION_FAILED = "SERVICE_INITIALIZATION_FAILED"
    
    # Data provider errors
    DATA_PROVIDER_UNAVAILABLE = "DATA_PROVIDER_UNAVAILABLE"
    API_RATE_LIMIT_EXCEEDED = "API_RATE_LIMIT_EXCEEDED"
    DATA_NOT_FOUND = "DATA_NOT_FOUND"
    INVALID_SYMBOL = "INVALID_SYMBOL"
    
    # Risk analysis errors
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    CALCULATION_ERROR = "CALCULATION_ERROR"
    INVALID_PARAMETERS = "INVALID_PARAMETERS"
    
    # Portfolio errors
    OPTIMIZATION_FAILED = "OPTIMIZATION_FAILED"
    INVALID_WEIGHTS = "INVALID_WEIGHTS"
    BENCHMARK_NOT_FOUND = "BENCHMARK_NOT_FOUND"
    
    # Behavioral errors
    CONVERSATION_ANALYSIS_FAILED = "CONVERSATION_ANALYSIS_FAILED"
    BIAS_DETECTION_FAILED = "BIAS_DETECTION_FAILED"
    INSUFFICIENT_CONVERSATION_DATA = "INSUFFICIENT_CONVERSATION_DATA"
    
    # Forecasting errors
    MODEL_FITTING_FAILED = "MODEL_FITTING_FAILED"
    PREDICTION_FAILED = "PREDICTION_FAILED"
    INSUFFICIENT_HISTORICAL_DATA = "INSUFFICIENT_HISTORICAL_DATA"


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Base exceptions
    "BackendError",
    "ServiceError", 
    "DataProviderError",
    
    # Service-specific exceptions
    "RiskAnalysisError",
    "BehavioralAnalysisError",
    "PortfolioManagementError",
    "ForecastingError",
    
    # Risk analysis exceptions
    "PortfolioDataError",
    "VaRCalculationError", 
    "StressTestingError",
    
    # Behavioral analysis exceptions
    "ConversationAnalysisError",
    "BiasDetectionError",
    "SentimentAnalysisError",
    
    # Portfolio management exceptions
    "OptimizationError",
    "RebalancingError",
    "PerformanceAnalysisError",
    
    # Forecasting exceptions
    "ModelFittingError",
    "PredictionError",
    "RegimeAnalysisError",
    
    # Data provider exceptions
    "FMPDataError",
    "APIRateLimitError",
    "DataNotAvailableError",
    "InvalidSymbolError",
    
    # General exceptions
    "ConfigurationError",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "ServiceUnavailableError",
    "DependencyError",
    "IntegrationError",
    
    # Utilities
    "create_service_error",
    "create_data_provider_error",
    "handle_service_exception",
    "ErrorCodes"
]