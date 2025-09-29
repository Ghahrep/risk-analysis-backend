# models/responses.py
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime

class RiskAnalysisResponse(BaseModel):
    success: bool
    portfolio_id: str
    analysis_type: str
    data: Dict[str, Any]
    execution_time_seconds: float
    analysis_timestamp: datetime
    api_version: str = "1.0.0"

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    error_code: Optional[str] = None
    timestamp: datetime