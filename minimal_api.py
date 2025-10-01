import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import time
import json
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

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    return obj

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
# Import risk tools - THESE ARE ASYNC
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

# Import forecasting tools - THESE ARE SYNC
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

# Import regime tools - THESE ARE SYNC  
try:
    from tools.regime_tools_standalone import (
        detect_hmm_regimes,
        detect_volatility_regimes,
        comprehensive_regime_analysis
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
            "portfolio_optimization": PORTFOLIO_TOOLS_AVAILABLE,  # Add this
            "forecasting": FORECASTING_TOOLS_AVAILABLE,
            "regime_analysis": REGIME_TOOLS_AVAILABLE
        }
    }

# ==================== RISK ANALYSIS ENDPOINTS (ASYNC) ====================

@app.post("/analyze")
async def comprehensive_risk_analysis(request: dict):
    """Comprehensive risk analysis - uses ASYNC tools"""
    if not RISK_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Risk analysis tools not available")
    
    try:
        symbols = request.get("symbols", [])
        weights = request.get("weights")
        period = request.get("period", "1year")
        use_real_data = request.get("use_real_data", True)
        
        if not symbols:
            raise HTTPException(status_code=400, detail="Symbols required")
        
        if weights and isinstance(weights, list):
            weights = {sym: wgt for sym, wgt in zip(symbols, weights)}
        
        # ✅ ASYNC function - use await
        result = await calculate_comprehensive_risk(
            symbols=symbols,
            weights=weights,
            period=period,
            use_real_data=use_real_data
        )
        
        return {
            "status": "success",
            "data_source": result.data_source if hasattr(result, 'data_source') else "Unknown",
            "metrics": result.__dict__ if hasattr(result, '__dict__') else result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/var")
async def value_at_risk_analysis(request: dict):
    """VaR analysis - uses ASYNC tools"""
    if not RISK_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Risk tools not available")
    
    try:
        symbols = request.get("symbols", [])
        weights = request.get("weights")
        confidence_levels = request.get("confidence_levels", [0.95, 0.99])
        period = request.get("period", "1year")
        use_real_data = request.get("use_real_data", True)
        
        if not symbols:
            raise HTTPException(status_code=400, detail="Symbols required")
        
        if weights and isinstance(weights, list):
            weights = {sym: wgt for sym, wgt in zip(symbols, weights)}
        
        # ✅ ASYNC function - use await
        result = await estimate_var_cvar(
            symbols=symbols,
            weights=weights,
            confidence_levels=confidence_levels,
            period=period,
            use_real_data=use_real_data
        )
        
        return {
            "status": "success",
            "var_analysis": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"VaR analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== FORECASTING ENDPOINTS (SYNC) ====================

@app.post("/forecast/returns")
async def forecast_returns(request: dict):
    """Forecast portfolio returns - uses SYNC tools"""
    if not FORECASTING_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Forecasting tools not available")
    
    try:
        symbols = request.get("symbols", [])
        period = request.get("period", "1year")
        forecast_horizon = request.get("forecast_horizon", 21)
        use_real_data = request.get("use_real_data", True)
        
        if not symbols:
            raise HTTPException(status_code=400, detail="Symbols required")
        
        result = forecast_portfolio_returns(
            returns=None,
            symbols=symbols,
            use_real_data=use_real_data,
            period=period,
            forecast_horizon=forecast_horizon
        )
        
        # ✅ Convert numpy types before returning
        result_clean = convert_numpy_types(result)
        
        return {
            "status": "success",
            "forecast": result_clean,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Forecast failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== REGIME ANALYSIS ENDPOINTS (SYNC) ====================

@app.post("/regime/hmm")
async def hmm_regime_detection(request: dict):
    """HMM regime detection - uses SYNC tools"""
    if not REGIME_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Regime tools not available")
    
    try:
        symbols = request.get("symbols", [])
        period = request.get("period", "2years")
        n_regimes = request.get("n_regimes", 2)
        use_real_data = request.get("use_real_data", True)
        
        if not symbols:
            raise HTTPException(status_code=400, detail="Symbols required")
        
        result = detect_hmm_regimes(
            symbols=symbols,
            use_real_data=use_real_data,
            period=period,
            n_regimes=n_regimes
        )
        
        # ✅ Convert numpy types before returning
        result_clean = convert_numpy_types(result)
        
        return {
            "status": "success",
            "regime_analysis": result_clean,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Regime detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

# ==================== PORTFOLIO OPTIMIZATION ENDPOINTS (ASYNC) ====================

@app.post("/optimize")
async def portfolio_optimization(request: dict):
    """Optimize portfolio allocation - uses ASYNC tools"""
    if not PORTFOLIO_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Portfolio optimization tools not available")
    
    try:
        symbols = request.get("symbols", [])
        method = request.get("method", "max_sharpe")
        period = request.get("period", "1year")
        risk_free_rate = request.get("risk_free_rate", 0.02)
        
        if not symbols:
            raise HTTPException(status_code=400, detail="Symbols required")
        
        # ✅ ASYNC function - use await
        result = await optimize_portfolio(
            symbols=symbols,
            method=method,
            period=period,
            risk_free_rate=risk_free_rate
        )
        
        return {
            "status": "success",
            "optimization": result.__dict__ if hasattr(result, '__dict__') else result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/portfolio-risk")
async def portfolio_risk_analysis(request: dict):
    """Calculate portfolio risk metrics - uses ASYNC tools"""
    if not PORTFOLIO_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Portfolio tools not available")
    
    try:
        holdings = request.get("holdings", {})
        symbols = request.get("symbols")
        period = request.get("period", "1year")
        risk_free_rate = request.get("risk_free_rate", 0.02)
        
        if not holdings:
            raise HTTPException(status_code=400, detail="Holdings required")
        
        # ✅ ASYNC function - use await
        result = await calculate_portfolio_risk(
            holdings=holdings,
            symbols=symbols,
            period=period,
            risk_free_rate=risk_free_rate
        )
        
        return {
            "status": "success",
            "risk_metrics": result.__dict__ if hasattr(result, '__dict__') else result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Portfolio risk analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    


@app.post("/stress-test")
async def stress_testing(request: dict):
    """Perform stress testing on portfolio - uses ASYNC tools"""
    if not RISK_TOOLS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Risk analysis tools not available")
    
    try:
        symbols = request.get("symbols", [])
        weights = request.get("weights")
        stress_scenarios = request.get("stress_scenarios")
        period = request.get("period", "1year")
        use_real_data = request.get("use_real_data", True)
        
        if not symbols:
            raise HTTPException(status_code=400, detail="Symbols required")
        
        if weights and isinstance(weights, list):
            weights = {sym: wgt for sym, wgt in zip(symbols, weights)}
        
        # ✅ ASYNC function - use await
        result = await perform_stress_testing(
            symbols=symbols,
            weights=weights,
            stress_scenarios=stress_scenarios,
            period=period,
            use_real_data=use_real_data
        )
        
        # Convert numpy types
        result_clean = convert_numpy_types(result.__dict__ if hasattr(result, '__dict__') else result)
        
        return {
            "status": "success",
            "stress_test_results": result_clean,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Stress testing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== SERVER STARTUP ====================



if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting Risk Analysis & Portfolio Management API")
    logger.info("=" * 60)
    logger.info(f"FMP API Key: {masked_key}")
    logger.info(f"Risk Tools Available: {RISK_TOOLS_AVAILABLE}")
    logger.info(f"Portfolio Tools Available: {PORTFOLIO_TOOLS_AVAILABLE}")  # Add this
    logger.info(f"Forecasting Tools Available: {FORECASTING_TOOLS_AVAILABLE}")
    logger.info(f"Regime Tools Available: {REGIME_TOOLS_AVAILABLE}")
    logger.info("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)