# models/behavioral_models.py
"""
Behavioral Analysis Models
==========================

Models for behavioral analysis service including bias detection,
sentiment analysis, and behavioral profiling with FMP integration.
"""

from typing import Dict, List, Optional
from pydantic import Field, field_validator
from .base_models import (
    BaseAnalysisRequest, BaseAnalysisResponse, ConversationMessage, 
    AnalysisPeriod, AnalysisDepth
)

# =============================================================================
# REQUEST MODELS - Behavioral Analysis
# =============================================================================

class BehavioralAnalysisRequest(BaseAnalysisRequest):
    """Comprehensive behavioral analysis request"""
    conversation_messages: List[ConversationMessage] = Field(..., min_length=1, description="Conversation history")
    symbols: Optional[List[str]] = Field(default=None, description="Portfolio symbols for FMP integration")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    use_real_data: bool = Field(default=True, description="Enable FMP integration")
    user_demographics: Optional[Dict] = Field(default=None, description="User demographic information")
    behavioral_focus: Optional[List[str]] = Field(default=None, description="Specific behavioral aspects")

class BiasDetectionRequest(BaseAnalysisRequest):
    """Targeted bias detection request"""
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

class SentimentAnalysisRequest(BaseAnalysisRequest):
    """Market sentiment analysis request with FMP context"""
    conversation_messages: List[ConversationMessage] = Field(..., min_length=1, description="Conversation history")
    symbols: Optional[List[str]] = Field(default=None, description="Portfolio symbols for market context")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    use_real_data: bool = Field(default=True, description="Enable FMP market context")
    time_window: Optional[int] = Field(None, ge=1, le=100, description="Number of recent messages")
    sentiment_model: str = Field(default="behavioral_finance", description="Sentiment analysis model")

class BehavioralProfileRequest(BaseAnalysisRequest):
    """Behavioral profile assessment request"""
    conversation_messages: List[ConversationMessage] = Field(..., min_length=1, description="Conversation history")
    profile_depth: str = Field(default="standard", description="Profiling depth")
    assessment_type: str = Field(default="comprehensive", description="Type of assessment")
    symbols: Optional[List[str]] = Field(default=None, description="Portfolio symbols")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    use_real_data: bool = Field(default=True, description="Enable FMP integration")

class PortfolioContextRequest(BaseAnalysisRequest):
    """Portfolio context analysis request"""
    conversation_messages: List[ConversationMessage] = Field(..., min_length=1, description="Conversation history")
    portfolio_holdings: Optional[Dict[str, float]] = Field(default=None, description="Portfolio holdings")
    symbols: Optional[List[str]] = Field(default=None, description="Portfolio symbols")
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR, description="Analysis period")
    use_real_data: bool = Field(default=True, description="Enable FMP integration")

# =============================================================================
# RESPONSE MODELS - Behavioral Analysis
# =============================================================================

class BehavioralAnalysisResponse(BaseAnalysisResponse):
    """Behavioral analysis response"""
    bias_count: int = Field(default=0, description="Number of biases detected")
    overall_risk_score: float = Field(default=0.0, description="Overall behavioral risk score")
    detected_biases: List[Dict] = Field(default_factory=list, description="Detected biases")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")

# =============================================================================
# EXAMPLE REQUESTS FOR TESTING
# =============================================================================

BEHAVIORAL_EXAMPLE_REQUESTS = {
    'behavioral_analysis': {
        "conversation_messages": [
            {"role": "user", "content": "I'm worried about my tech stocks losing value"},
            {"role": "assistant", "content": "I understand your concern about tech stock volatility"},
            {"role": "user", "content": "Should I sell everything and wait for a crash?"}
        ],
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "use_real_data": True,
        "period": "1year",
        "analysis_depth": "comprehensive"
    },
    'bias_detection': {
        "conversation_messages": [
            {"role": "user", "content": "I never sell my losing stocks because I know they'll come back"},
            {"role": "user", "content": "But I always sell my winners quickly to lock in profits"}
        ],
        "bias_types": ["disposition_effect", "loss_aversion"],
        "symbols": ["AAPL", "TSLA"],
        "sensitivity_level": "high"
    },
    'sentiment_analysis': {
        "conversation_messages": [
            {"role": "user", "content": "The market is definitely going to crash soon"},
            {"role": "user", "content": "Everyone is talking about it on social media"}
        ],
        "symbols": ["SPY", "QQQ"],
        "sentiment_model": "behavioral_finance",
        "time_window": 10
    }
}

# Export all models
__all__ = [
    # Request Models
    "BehavioralAnalysisRequest", "BiasDetectionRequest", "SentimentAnalysisRequest",
    "BehavioralProfileRequest", "PortfolioContextRequest",
    
    # Response Models
    "BehavioralAnalysisResponse",
    
    # Example Data
    "BEHAVIORAL_EXAMPLE_REQUESTS"
]