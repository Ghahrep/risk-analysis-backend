# models/base_models.py
"""
Base Models and Shared Components
=================================

Contains shared base classes, enums, and common validation logic
used across all model files in the backend.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

# =============================================================================
# PYDANTIC V2 CONFIGURATION
# =============================================================================

# Global config for all models to prevent namespace conflicts
DEFAULT_CONFIG = ConfigDict(
    extra='ignore',
    protected_namespaces=(),
    use_enum_values=True,
    validate_assignment=True
)

class BaseRequestModel(BaseModel):
    """Base class for all request models"""
    model_config = DEFAULT_CONFIG

class BaseResponseModel(BaseModel):
    """Base class for all response models"""
    model_config = DEFAULT_CONFIG

# =============================================================================
# SHARED ENUMS
# =============================================================================

class AnalysisPeriod(str, Enum):
    """Analysis time periods"""
    ONE_MONTH = "1month"
    THREE_MONTHS = "3months"
    SIX_MONTHS = "6months"
    ONE_YEAR = "1year"
    TWO_YEARS = "2years"
    THREE_YEARS = "3years"
    FIVE_YEARS = "5years"

class AnalysisDepth(str, Enum):
    """Analysis depth levels"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    DETAILED = "detailed"

class IntegrationLevel(str, Enum):
    """Integration levels for multi-service analysis"""
    SINGLE = "single"
    TWO_WAY = "two_way"
    THREE_WAY = "three_way"
    FOUR_WAY = "four_way"

# =============================================================================
# SHARED BASE CLASSES
# =============================================================================

class BaseFMPRequest(BaseRequestModel):
    """Base class for requests with FMP integration support"""
    symbols: Optional[List[str]] = Field(None, description="Stock symbols for analysis")
    use_real_data: bool = Field(True, description="Enable FMP real market data integration")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    
    @field_validator('symbols')
    @classmethod
    def validate_symbols(cls, v):
        if v is not None:
            if not v:
                raise ValueError("Symbols list cannot be empty")
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

class BaseAnalysisRequest(BaseFMPRequest):
    """Base class for analysis requests with common parameters"""
    analysis_depth: AnalysisDepth = Field(default=AnalysisDepth.COMPREHENSIVE, description="Depth of analysis")

class BaseAnalysisResponse(BaseResponseModel):
    """Base response model with consistent structure"""
    success: bool = Field(..., description="Analysis success status")
    message: str = Field(..., description="Response message")
    data_source: str = Field(..., description="Data source used")
    execution_time: float = Field(..., description="Execution time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class HealthResponse(BaseResponseModel):
    """Service health response"""
    service: str = Field(..., description="Service name")
    status: str = Field(..., description="Health status")
    version: str = Field(default="2.0", description="Service version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    error: Optional[str] = Field(default=None, description="Error message if unhealthy")

# =============================================================================
# SHARED MESSAGE MODEL
# =============================================================================

class ConversationMessage(BaseRequestModel):
    """Individual conversation message for behavioral analysis"""
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
# SHARED VALIDATION HELPERS
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

# Export all shared components
__all__ = [
    # Base Classes
    "BaseRequestModel", "BaseResponseModel", "BaseFMPRequest", "BaseAnalysisRequest", "BaseAnalysisResponse",
    
    # Enums
    "AnalysisPeriod", "AnalysisDepth", "IntegrationLevel",
    
    # Common Models
    "ConversationMessage", "HealthResponse",
    
    # Validation Functions
    "validate_confidence_levels", "validate_symbols_for_analysis"
]