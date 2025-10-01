"""
API Client for interacting with Risk Analysis Backend
Handles all API calls to minimal_api.py and behavioral_complete_api.py
"""

import requests
import streamlit as st
from typing import List, Dict, Any, Optional
import logging
import time
from utils.request_logger import request_logger

logger = logging.getLogger(__name__)

class RiskAnalysisAPIClient:
    """Client for minimal_api.py (Port 8001)"""
    
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
    
    def _get(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Make GET request with error handling"""
        try:
            response = requests.get(
                f"{self.base_url}{endpoint}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {str(e)}")
            return None
    
    # In utils/api_client.py, update the _post method around line 46-52:

    def _post(self, endpoint: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make POST request with error handling"""
        request_logger.log_request(endpoint, data)
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json=data,
                timeout=30
            )
            duration = time.time() - start_time
            
            response.raise_for_status()
            
            # FIX: Wrap json() in try-except
            try:
                result = response.json()
            except ValueError as e:
                request_logger.log_response(endpoint, False, duration, f"Invalid JSON: {str(e)}")
                logger.error(f"Invalid JSON response: {str(e)}")
                return None
            
            # Log success
            request_logger.log_response(endpoint, True, duration)
            return result
            
        except requests.exceptions.RequestException as e:
            duration = time.time() - start_time
            request_logger.log_response(endpoint, False, duration, str(e))
            logger.error(f"POST request failed: {str(e)}")
            return None
    
    # Health & Status
    def health_check(self) -> Optional[Dict[str, Any]]:
        """Check API health status"""
        return self._get("/health")
    
    def get_capabilities(self) -> Optional[Dict[str, Any]]:
        """Get API capabilities"""
        return self._get("/capabilities")
    
    # Portfolio Optimization
    def optimize_portfolio(
        self, 
        symbols: List[str], 
        method: str = "max_sharpe",
        period: str = "1year"
    ) -> Optional[Dict[str, Any]]:
        """Optimize portfolio allocation"""
        return self._post("/optimize", {
            "symbols": symbols,
            "method": method,
            "period": period,
            "use_real_data": True
        })
    
    # Risk Analysis
    def analyze_risk(
        self,
        symbols: List[str],
        weights: List[float],
        period: str = "1year"
    ) -> Optional[Dict[str, Any]]:
        """Comprehensive risk analysis"""
        return self._post("/analyze", {
            "symbols": symbols,
            "weights": weights,
            "period": period,
            "use_real_data": True
        })
    
    def calculate_var(
        self,
        symbols: List[str],
        weights: List[float],
        confidence_level: float = 0.95
    ) -> Optional[Dict[str, Any]]:
        """Calculate Value at Risk"""
        return self._post("/var", {
            "symbols": symbols,
            "weights": weights,
            "confidence_level": confidence_level,
            "use_real_data": True
        })
    
    def stress_test(
        self,
        symbols: List[str],
        weights: List[float]
    ) -> Optional[Dict[str, Any]]:
        """Run stress test scenarios"""
        return self._post("/stress", {
            "symbols": symbols,
            "weights": weights,
            "use_real_data": True
        })
    
    # Advanced Analytics
    def risk_attribution(
        self,
        symbols: List[str],
        weights: List[float],
        period: str = "1year"
    ) -> Optional[Dict[str, Any]]:
        """Portfolio risk attribution analysis"""
        return self._post("/risk-attribution", {
            "symbols": symbols,
            "weights": weights,
            "period": period,
            "use_real_data": True
        })
    
    def performance_attribution(
        self,
        symbols: List[str],
        weights: List[float],
        benchmark: str = "SPY",
        period: str = "1year"
    ) -> Optional[Dict[str, Any]]:
        """Performance attribution vs benchmark"""
        return self._post("/performance-attribution", {
            "symbols": symbols,
            "weights": weights,
            "benchmark": benchmark,
            "period": period,
            "use_real_data": True
        })
    
    def advanced_analytics(
        self,
        symbols: List[str],
        weights: List[float],
        period: str = "1year"
    ) -> Optional[Dict[str, Any]]:
        """Comprehensive advanced portfolio metrics"""
        return self._post("/advanced-analytics", {
            "symbols": symbols,
            "weights": weights,
            "period": period,
            "use_real_data": True
        })
    
    # Correlation Analysis
    def correlation_analysis(
        self,
        symbols: List[str],
        period: str = "1year"
    ) -> Optional[Dict[str, Any]]:
        """Basic correlation analysis"""
        return self._post("/correlation-analysis", {
            "symbols": symbols,
            "period": period,
            "use_real_data": True
        })
    
    def rolling_correlations(
        self,
        symbols: List[str],
        window_size: int = 30,
        period: str = "1year"
    ) -> Optional[Dict[str, Any]]:
        """Time-varying correlation analysis"""
        return self._post("/rolling-correlations", {
            "symbols": symbols,
            "window_size": window_size,
            "period": period,
            "use_real_data": True
        })
    
    def regime_correlations(
        self,
        symbols: List[str],
        regime_method: str = "volatility",
        period: str = "1year"
    ) -> Optional[Dict[str, Any]]:
        """Regime-conditional correlation analysis"""
        return self._post("/regime-correlations", {
            "symbols": symbols,
            "regime_method": regime_method,
            "period": period,
            "use_real_data": True
        })
    
    def correlation_clustering(
        self,
        symbols: List[str],
        period: str = "1year"
    ) -> Optional[Dict[str, Any]]:
        """Hierarchical clustering of correlations"""
        return self._post("/correlation-clustering", {
            "symbols": symbols,
            "period": period,
            "use_real_data": True
        })
    
    def correlation_network(
        self,
        symbols: List[str],
        period: str = "1year"
    ) -> Optional[Dict[str, Any]]:
        """Network topology analysis"""
        return self._post("/correlation-network", {
            "symbols": symbols,
            "period": period,
            "use_real_data": True
        })
    
    def comprehensive_correlation(
        self,
        symbols: List[str],
        period: str = "1year"
    ) -> Optional[Dict[str, Any]]:
        """Integrated correlation analysis"""
        return self._post("/comprehensive-correlation", {
            "symbols": symbols,
            "period": period,
            "use_real_data": True
        })
    
    # Forecasting & Regime
    def forecast_returns(
        self,
        symbols: List[str],
        period: str = "1year"
    ) -> Optional[Dict[str, Any]]:
        """Return forecasting"""
        return self._post("/forecast", {
            "symbols": symbols,
            "period": period,
            "use_real_data": True
        })
    
    def regime_analysis_hmm(
        self,
        symbol: str,
        period: str = "1year"
    ) -> Optional[Dict[str, Any]]:
        """HMM regime detection"""
        return self._post("/regime-hmm", {
            "symbol": symbol,
            "period": period,
            "use_real_data": True
        })
    
    def regime_analysis_volatility(
        self,
        symbol: str,
        period: str = "1year"
    ) -> Optional[Dict[str, Any]]:
        """Volatility-based regime analysis"""
        return self._post("/regime-volatility", {
            "symbol": symbol,
            "period": period,
            "use_real_data": True
        })
    
    def forecast_volatility_garch(
        self,
        symbols: List[str],
        forecast_horizon: int = 30,
        period: str = "1year"
    ) -> Optional[Dict[str, Any]]:
        """Forecast volatility using GARCH models"""
        response_data = self._post("/volatility", {
            "symbols": symbols,
            "forecast_horizon": forecast_horizon,
            "period": period,
            "use_real_data": True
        })
        
        if not response_data:
            return None
        
        # Transform API response to match Streamlit page expectations
        forecast = response_data.get("forecast", {})
        
        return {
            "success": response_data.get("success", False),
            "volatility_forecast": {
                "current_volatility": forecast.get("current_volatility", 0),
                "forecast_mean": forecast.get("forecast_volatility", 0),
                "trend": forecast.get("volatility_trend", "stable").lower(),
                "confidence_bands": forecast.get("confidence_bands", {}),
                "model_aic": forecast.get("model_aic")
            },
            "data_source": response_data.get("data_source", "Unknown"),
            "symbols": response_data.get("symbols", symbols),
            "period": response_data.get("period", period)
        }


class BehavioralAPIClient:
    """Client for behavioral_complete_api.py (Port 8003)"""
    
    def __init__(self, base_url: str = "http://localhost:8003"):
        self.base_url = base_url
        self.timeout = 20
    
    def _post(self, endpoint: str, data: dict) -> Optional[Dict[str, Any]]:
        """Make POST request with error handling"""
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Behavioral API Error: {str(e)}")
            return None
    
    def _get(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Make GET request"""
        try:
            response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Behavioral API Error: {str(e)}")
            return None
    
    def health_check(self) -> Optional[Dict[str, Any]]:
        """Check behavioral API health"""
        return self._get("/health")
    
    def analyze_biases(
        self,
        conversation_messages: List[Dict[str, str]],
        symbols: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Detect cognitive biases in conversation"""
        data = {"conversation_messages": conversation_messages}
        if symbols:
            data["symbols"] = symbols
        return self._post("/analyze-biases", data)
    
    def analyze_sentiment(
        self,
        conversation_messages: List[Dict[str, str]],
        symbols: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Analyze market sentiment"""
        data = {"conversation_messages": conversation_messages}
        if symbols:
            data["symbols"] = symbols
        return self._post("/analyze-sentiment", data)
    
    def assess_profile(
        self,
        conversation_messages: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        """Assess behavioral profile and risk tolerance"""
        return self._post("/assess-profile", {
            "conversation_messages": conversation_messages
        })
    
    def comprehensive_behavioral_analysis(
        self,
        conversation_messages: List[Dict[str, str]],
        symbols: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Run all behavioral analyses"""
        data = {"conversation_messages": conversation_messages}
        if symbols:
            data["symbols"] = symbols
        return self._post("/comprehensive-behavioral", data)


# Singleton instances for reuse
@st.cache_resource
def get_risk_api_client():
    """Get cached risk analysis API client"""
    return RiskAnalysisAPIClient()

@st.cache_resource
def get_behavioral_api_client():
    """Get cached behavioral API client"""
    return BehavioralAPIClient()