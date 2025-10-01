import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import time

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd

import multiprocessing
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
def load_env_file(env_path: str = '.env'):
    """Load environment variables from .env file"""
    if not os.path.exists(env_path):
        logger.warning(f".env file not found at {env_path}")
        return
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value

# Load environment variables
load_env_file()

# ✅ SECURE: No hardcoded API key fallback
FMP_API_KEY = os.getenv('FMP_API_KEY')
if not FMP_API_KEY:
    raise ValueError(
        "FMP_API_KEY environment variable not set.\n"
        "Create a .env file in the project root with:\n"
        "FMP_API_KEY=your_api_key_here"
    )

# Mask key for logging
masked_key = f"{FMP_API_KEY[:10]}...{FMP_API_KEY[-6:]}" if len(FMP_API_KEY) > 16 else "***"
logger.info(f"FMP API Key loaded: {masked_key}")

# Initialize FastAPI
app = FastAPI(
    title="Risk Analysis & Portfolio Management API",
    description="Minimal API with real FMP market data integration for risk analysis and portfolio optimization",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Import risk tools
try:
    from tools.risk_tools_standalone import (
        calculate_comprehensive_risk,
        estimate_var_cvar,
        perform_stress_testing,
        forecast_volatility_garch
    )
    RISK_TOOLS_AVAILABLE = True
    logger.info("Risk analysis tools imported successfully")
except ImportError as e:
    logger.warning(f"Risk analysis tools not available: {e}")
    RISK_TOOLS_AVAILABLE = False

# Import portfolio tools
try:
    from tools.standalone_fmp_portfolio_tools import (
        optimize_portfolio,
        calculate_portfolio_risk
    )
    PORTFOLIO_TOOLS_AVAILABLE = True
    logger.info("Portfolio tools imported successfully")
except ImportError as e:
    logger.warning(f"Portfolio tools not available: {e}")
    PORTFOLIO_TOOLS_AVAILABLE = False

# Import forecasting tools (if available)
try:
    from tools.forecasting_tools import (
        forecast_portfolio_returns,
        forecast_volatility_with_regimes,
        forecast_regime_transitions,
        generate_scenario_forecasts
    )
    FORECASTING_TOOLS_AVAILABLE = True
    logger.info("Forecasting tools imported successfully")
except ImportError as e:
    logger.warning(f"Forecasting tools not available: {e}")
    FORECASTING_TOOLS_AVAILABLE = False

# Import regime analysis tools (if available)
try:
    from tools.regime_analysis_tools import (
        detect_hmm_regimes,
        detect_volatility_regimes,
        comprehensive_regime_analysis,
        get_regime_tools_integration_status
    )
    REGIME_TOOLS_AVAILABLE = True
    logger.info("Regime analysis tools imported successfully")
except ImportError as e:
    logger.warning(f"Regime analysis tools not available: {e}")
    REGIME_TOOLS_AVAILABLE = False

# ==================== HEALTH CHECK ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "risk-analysis-api",
        "version": "2.0.0",
        "fmp_connected": bool(FMP_API_KEY),
        "available_tools": {
            "risk_analysis": RISK_TOOLS_AVAILABLE,
            "portfolio_optimization": PORTFOLIO_TOOLS_AVAILABLE,
            "forecasting": FORECASTING_TOOLS_AVAILABLE,
            "regime_analysis": REGIME_TOOLS_AVAILABLE
        }
    }

# ==================== RISK ANALYSIS ENDPOINTS ====================

@app.post("/analyze")
async def comprehensive_risk_analysis(request: dict):
    """
    Comprehensive risk analysis for a portfolio
    
    Request format:
    {
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "weights": [0.4, 0.3, 0.3],  # Optional, defaults to equal weight
        "period": "1year",  # Optional: "1month", "3months", "6months", "1year", "2years"
        "use_real_data": true
    }
    """
    if not RISK_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Risk analysis tools not available")
    
    try:
        # Extract and validate parameters
        symbols = request.get("symbols", [])
        weights = request.get("weights")
        period = request.get("period", "1year")
        use_real_data = request.get("use_real_data", True)
        
        if not symbols:
            raise HTTPException(status_code=400, detail="Symbols list is required")
        
        # Default to equal weights if not provided
        if weights is None:
            weights = [1.0 / len(symbols)] * len(symbols)
        
        # Validate weights
        if len(weights) != len(symbols):
            raise HTTPException(status_code=400, detail="Weights must match symbols length")
        
        if not np.isclose(sum(weights), 1.0):
            raise HTTPException(status_code=400, detail="Weights must sum to 1.0")
        
        logger.info(f"Analyzing portfolio: {symbols} with weights: {weights}")
        
        # Call risk analysis tool
        result = calculate_comprehensive_risk(
            symbols=symbols,
            weights=weights,
            period=period,
            use_real_data=use_real_data,
            fmp_api_key=FMP_API_KEY
        )
        
        return {
            "status": "success",
            "data_source": "FMP_Real" if use_real_data else "Synthetic",
            "portfolio": {
                "symbols": symbols,
                "weights": weights
            },
            "metrics": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Comprehensive analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/var")
async def value_at_risk_analysis(request: dict):
    """
    Calculate Value at Risk (VaR) and Conditional VaR
    
    Request format:
    {
        "symbols": ["AAPL", "GOOGL"],
        "weights": [0.5, 0.5],
        "confidence_level": 0.95,  # Optional, default 0.95
        "period": "1year",
        "use_real_data": true
    }
    """
    if not RISK_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Risk analysis tools not available")
    
    try:
        symbols = request.get("symbols", [])
        weights = request.get("weights")
        confidence_level = request.get("confidence_level", 0.95)
        period = request.get("period", "1year")
        use_real_data = request.get("use_real_data", True)
        
        if not symbols:
            raise HTTPException(status_code=400, detail="Symbols required")
        
        if weights is None:
            weights = [1.0 / len(symbols)] * len(symbols)
        
        result = estimate_var_cvar(
            symbols=symbols,
            weights=weights,
            confidence_level=confidence_level,
            period=period,
            use_real_data=use_real_data,
            fmp_api_key=FMP_API_KEY
        )
        
        return {
            "status": "success",
            "data_source": "FMP_Real" if use_real_data else "Synthetic",
            "var_analysis": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"VaR analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stress-test")
async def stress_testing(request: dict):
    """
    Perform stress testing on portfolio
    
    Request format:
    {
        "symbols": ["AAPL", "GOOGL"],
        "weights": [0.5, 0.5],
        "scenarios": ["market_crash", "volatility_spike"],  # Optional
        "period": "1year",
        "use_real_data": true
    }
    """
    if not RISK_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Risk analysis tools not available")
    
    try:
        symbols = request.get("symbols", [])
        weights = request.get("weights")
        scenarios = request.get("scenarios")
        period = request.get("period", "1year")
        use_real_data = request.get("use_real_data", True)
        
        if not symbols:
            raise HTTPException(status_code=400, detail="Symbols required")
        
        if weights is None:
            weights = [1.0 / len(symbols)] * len(symbols)
        
        result = perform_stress_testing(
            symbols=symbols,
            weights=weights,
            scenarios=scenarios,
            period=period,
            use_real_data=use_real_data,
            fmp_api_key=FMP_API_KEY
        )
        
        return {
            "status": "success",
            "data_source": "FMP_Real" if use_real_data else "Synthetic",
            "stress_test_results": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Stress testing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/volatility-forecast")
async def volatility_forecasting(request: dict):
    """
    Forecast volatility using GARCH model
    
    Request format:
    {
        "symbols": ["AAPL"],
        "forecast_horizon": 30,  # Days
        "period": "1year",
        "use_real_data": true
    }
    """
    if not RISK_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Risk analysis tools not available")
    
    try:
        symbols = request.get("symbols", [])
        forecast_horizon = request.get("forecast_horizon", 30)
        period = request.get("period", "1year")
        use_real_data = request.get("use_real_data", True)
        
        if not symbols:
            raise HTTPException(status_code=400, detail="Symbols required")
        
        result = forecast_volatility_garch(
            symbols=symbols,
            forecast_horizon=forecast_horizon,
            period=period,
            use_real_data=use_real_data,
            fmp_api_key=FMP_API_KEY
        )
        
        return {
            "status": "success",
            "data_source": "FMP_Real" if use_real_data else "Synthetic",
            "volatility_forecast": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Volatility forecasting failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== PORTFOLIO OPTIMIZATION ENDPOINTS ====================

@app.post("/optimize")
async def portfolio_optimization(request: dict):
    """
    Optimize portfolio allocation
    
    Request format:
    {
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "method": "max_sharpe",  # "max_sharpe", "min_variance", "equal_weight"
        "period": "1year",
        "use_real_data": true
    }
    """
    if not PORTFOLIO_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Portfolio optimization tools not available")
    
    try:
        symbols = request.get("symbols", [])
        method = request.get("method", "max_sharpe")
        period = request.get("period", "1year")
        use_real_data = request.get("use_real_data", True)
        
        if not symbols:
            raise HTTPException(status_code=400, detail="Symbols required")
        
        result = optimize_portfolio(
            symbols=symbols,
            method=method,
            period=period,
            use_real_data=use_real_data,
            fmp_api_key=FMP_API_KEY
        )
        
        return {
            "status": "success",
            "data_source": "FMP_Real" if use_real_data else "Synthetic",
            "optimization": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/portfolio-risk")
async def portfolio_risk_analysis(request: dict):
    """
    Calculate portfolio risk metrics
    
    Request format:
    {
        "symbols": ["AAPL", "GOOGL"],
        "weights": [0.5, 0.5],
        "period": "1year",
        "use_real_data": true
    }
    """
    if not PORTFOLIO_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Portfolio tools not available")
    
    try:
        symbols = request.get("symbols", [])
        weights = request.get("weights")
        period = request.get("period", "1year")
        use_real_data = request.get("use_real_data", True)
        
        if not symbols:
            raise HTTPException(status_code=400, detail="Symbols required")
        
        if weights is None:
            weights = [1.0 / len(symbols)] * len(symbols)
        
        result = calculate_portfolio_risk(
            symbols=symbols,
            weights=weights,
            period=period,
            use_real_data=use_real_data,
            fmp_api_key=FMP_API_KEY
        )
        
        return {
            "status": "success",
            "data_source": "FMP_Real" if use_real_data else "Synthetic",
            "risk_metrics": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Portfolio risk analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== SERVER STARTUP ====================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting Risk Analysis & Portfolio Management API")
    logger.info("=" * 60)
    logger.info(f"FMP API Key: {masked_key}")
    logger.info(f"Risk Tools Available: {RISK_TOOLS_AVAILABLE}")
    logger.info(f"Portfolio Tools Available: {PORTFOLIO_TOOLS_AVAILABLE}")
    logger.info(f"Forecasting Tools Available: {FORECASTING_TOOLS_AVAILABLE}")
    logger.info(f"Regime Tools Available: {REGIME_TOOLS_AVAILABLE}")
    logger.info("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)