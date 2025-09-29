# tools/forecasting_tools.py - Enhanced with FMP Integration
"""
Time Series Forecasting Tools - Integrated Architecture with FMP Data
====================================================================

ARIMA-based return forecasting with regime conditioning and market structure analysis.
Enhanced with Financial Modeling Prep integration for real market data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings
import logging
from dataclasses import dataclass

warnings.filterwarnings('ignore', category=RuntimeWarning)
logger = logging.getLogger(__name__)

# FMP Data Integration
try:
    from data.utils.market_data_manager import MarketDataManager
    HAS_FMP_INTEGRATION = True
except ImportError:
    HAS_FMP_INTEGRATION = False
    logger.warning("FMP integration not available. Using synthetic data only.")

# Optional dependencies with graceful fallbacks
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logger.warning("statsmodels not available - ARIMA forecasting disabled")

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    logger.warning("arch not available - advanced GARCH disabled")

try:
    from .risk_tools_standalone import fit_garch_forecast, calculate_risk_metrics
    from .regime_tools_standalone import detect_hmm_regimes, detect_volatility_regimes
    HAS_INTEGRATED_TOOLS = True
except ImportError:
    try:
        from risk_tools import fit_garch_forecast, calculate_risk_metrics
        from tools.regime_tools_standalone import detect_hmm_regimes, detect_volatility_regimes
        HAS_INTEGRATED_TOOLS = True
    except ImportError:
        HAS_INTEGRATED_TOOLS = False
        logger.warning("Integrated tools not available - reduced functionality")

# Add helper function to get data manager
def _get_data_manager():
    """Get FMP data manager instance"""
    if not HAS_FMP_INTEGRATION:
        return None
    return MarketDataManager()

# ============================================================================
# CORE FORECASTING FUNCTIONS - ENHANCED WITH FMP INTEGRATION
# ============================================================================

def _safe_date_format(date_obj):
    """Safely format date object to string"""
    if hasattr(date_obj, 'strftime'):
        return date_obj.strftime('%Y-%m-%d')
    else:
        return str(date_obj)

def forecast_portfolio_returns(
    returns: pd.Series = None,
    symbols: Optional[List[str]] = None,        # NEW: For FMP integration
    use_real_data: bool = False,               # NEW: Enable FMP data
    period: str = "1year",                     # NEW: Data period
    forecast_horizon: int = 21,
    model_type: str = "auto_arima",
    confidence_levels: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95],
    include_regime_conditioning: bool = True,
) -> Dict[str, Any]:
    """
    ENHANCED: Comprehensive return forecasting with regime conditioning and FMP integration
    
    Parameters:
    -----------
    returns : pd.Series, optional
        Historical return series (backward compatibility)
    symbols : List[str], optional
        List of symbols for FMP data retrieval (NEW)
    use_real_data : bool, default=False
        Use real market data via FMP (NEW)
    period : str, default="1year"
        Data period for real data ('1month', '3months', '1year', '2years', '5years') (NEW)
    forecast_horizon : int, default=21
        Number of periods to forecast (business days)
    model_type : str, default="auto_arima"
        Forecasting model: 'auto_arima', 'arima_garch', 'simple_mean', 'regime_conditional'
    confidence_levels : List[float]
        Confidence levels for prediction intervals
    include_regime_conditioning : bool, default=True
        Whether to include regime-conditional analysis
        
    Returns:
    --------
    Dict[str, Any]
        Comprehensive forecasting results with data source tracking
    """
    try:
        data_source = "Unknown"
        
        # Handle FMP real data integration
        if use_real_data and symbols and HAS_FMP_INTEGRATION:
            data_manager = _get_data_manager()
            if data_manager:
                real_returns_data = data_manager.get_portfolio_returns(symbols, period=period)
                if real_returns_data is not None:
                    # Create equal-weighted portfolio returns or use single asset
                    if len(symbols) == 1:
                        returns = real_returns_data[symbols[0]]
                    else:
                        returns = real_returns_data.mean(axis=1)
                    data_source = "Financial Modeling Prep"
                    logger.info(f"Using FMP data for forecasting: {symbols} over {period}")
                else:
                    returns = _generate_synthetic_returns_single(symbols[0] if symbols else "SPY")
                    data_source = "Synthetic (FMP failed)"
                    logger.warning("FMP data retrieval failed, using synthetic data")
            else:
                returns = _generate_synthetic_returns_single(symbols[0] if symbols else "SPY")
                data_source = "Synthetic (FMP unavailable)"
        elif returns is not None:
            data_source = "Provided"
        else:
            # Fallback to synthetic data
            symbol = symbols[0] if symbols else "SPY"
            returns = _generate_synthetic_returns_single(symbol)
            data_source = "Generated Synthetic"
        
        if returns is None or returns.empty or len(returns) < 30:
            return _empty_forecast_response("Insufficient data for forecasting", data_source)
        
        forecast_results = {
            'success': True,
            'forecast_type': 'portfolio_returns',
            'model_used': model_type,
            'forecast_horizon': forecast_horizon,
            'data_source': data_source,  # NEW: Track data source
            'data_period': period if use_real_data else None,  # NEW: Track period used
            'symbols_analyzed': symbols if symbols else None,  # NEW: Track symbols
            'data_summary': {
                'observations': len(returns.dropna()),
                'start_date': _safe_date_format(returns.index[0]),
                'end_date': _safe_date_format(returns.index[-1]),
                'mean_return': float(returns.mean()),
                'volatility': float(returns.std()),
                'annualized_return': float(returns.mean() * 252),
                'annualized_volatility': float(returns.std() * np.sqrt(252))
            },
            'forecasts': {},
            'confidence_intervals': {},
            'forecast_accuracy': {},
            'regime_analysis': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate base forecast using specified model
        base_forecast = _generate_base_forecast(returns, forecast_horizon, model_type)
        if not base_forecast.get('success', False):
            return _error_forecast_response(base_forecast.get('error', 'Base forecasting failed'), data_source)
        
        # Store forecasts as dictionary to allow regime conditioning
        forecast_results['forecasts'] = {
            'base_forecast': base_forecast['forecasts']
        }
        forecast_results['confidence_intervals'] = base_forecast['confidence_intervals']
        forecast_results['forecast_accuracy'] = base_forecast['model_diagnostics']

        # Add regime conditioning if requested and available
        if include_regime_conditioning and HAS_INTEGRATED_TOOLS:
            regime_forecast = _generate_regime_conditional_forecast(returns, forecast_horizon, confidence_levels)
            if regime_forecast.get('success', False):
                forecast_results['regime_analysis'] = regime_forecast['regime_analysis']
                forecast_results['forecasts']['regime_conditional'] = regime_forecast['forecasts']
        
        return forecast_results
        
    except Exception as e:
        logger.error(f"Portfolio return forecasting failed: {e}")
        return _error_forecast_response(str(e), data_source)

def forecast_volatility_with_regimes(
    returns: pd.Series = None,
    symbols: Optional[List[str]] = None,        # NEW: For FMP integration
    use_real_data: bool = False,               # NEW: Enable FMP data
    period: str = "1year",                     # NEW: Data period
    forecast_horizon: int = 21,
    volatility_model: str = "garch",
    include_regime_switching: bool = True
) -> Dict[str, Any]:
    """
    ENHANCED: Volatility forecasting with regime awareness and FMP integration
    
    Parameters:
    -----------
    returns : pd.Series, optional
        Historical return series (backward compatibility)
    symbols : List[str], optional
        List of symbols for FMP data retrieval (NEW)
    use_real_data : bool, default=False
        Use real market data via FMP (NEW)
    period : str, default="1year"
        Data period for real data (NEW)
    forecast_horizon : int, default=21
        Forecast horizon in days
    volatility_model : str, default="garch"
        Volatility model: 'garch', 'rolling', 'regime_switching'
    include_regime_switching : bool, default=True
        Include regime-switching volatility model
        
    Returns:
    --------
    Dict[str, Any]
        Volatility forecasting results with data source tracking
    """
    try:
        data_source = "Unknown"
        
        # Handle FMP real data integration
        if use_real_data and symbols and HAS_FMP_INTEGRATION:
            data_manager = _get_data_manager()
            if data_manager:
                real_returns_data = data_manager.get_portfolio_returns(symbols, period=period)
                if real_returns_data is not None:
                    # Create equal-weighted portfolio returns or use single asset
                    if len(symbols) == 1:
                        returns = real_returns_data[symbols[0]]
                    else:
                        returns = real_returns_data.mean(axis=1)
                    data_source = "Financial Modeling Prep"
                    logger.info(f"Using FMP data for volatility forecasting: {symbols}")
                else:
                    returns = _generate_synthetic_returns_single(symbols[0] if symbols else "SPY")
                    data_source = "Synthetic (FMP failed)"
            else:
                returns = _generate_synthetic_returns_single(symbols[0] if symbols else "SPY")
                data_source = "Synthetic (FMP unavailable)"
        elif returns is not None:
            data_source = "Provided"
        else:
            # Fallback to synthetic data
            symbol = symbols[0] if symbols else "SPY"
            returns = _generate_synthetic_returns_single(symbol)
            data_source = "Generated Synthetic"
        
        if returns is None or returns.empty or len(returns) < 50:
            return _empty_forecast_response("Insufficient data for volatility forecasting", data_source)
        
        volatility_results = {
            'success': True,
            'forecast_type': 'volatility',
            'model_used': volatility_model,
            'forecast_horizon': forecast_horizon,
            'data_source': data_source,  # NEW: Track data source
            'data_period': period if use_real_data else None,  # NEW: Track period used
            'symbols_analyzed': symbols if symbols else None,  # NEW: Track symbols
            'volatility_forecasts': {},
            'regime_volatility': {},
            'model_diagnostics': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Base volatility forecast
        if volatility_model == "garch" and HAS_INTEGRATED_TOOLS:
            garch_result = fit_garch_forecast(returns, forecast_horizon)
            if garch_result:
                volatility_results['volatility_forecasts']['garch'] = {
                    'daily_volatilities': _extract_volatility_series(garch_result, forecast_horizon),
                    'mean_volatility': garch_result.get('mean_forecast_daily_vol', 0),
                    'volatility_trend': garch_result.get('volatility_trend', 'Stable'),
                    'model_aic': garch_result.get('model_aic')
                }
        
        # Regime-switching volatility
        if include_regime_switching and HAS_INTEGRATED_TOOLS:
            regime_vol = _generate_regime_switching_volatility(returns, forecast_horizon)
            if regime_vol.get('success', False):
                volatility_results['regime_volatility'] = regime_vol['regime_volatility']
        
        # Fallback: Simple rolling volatility
        if not volatility_results['volatility_forecasts']:
            rolling_vol = _generate_rolling_volatility_forecast(returns, forecast_horizon)
            volatility_results['volatility_forecasts']['rolling'] = rolling_vol
        
        return volatility_results
        
    except Exception as e:
        logger.error(f"Volatility forecasting failed: {e}")
        return _error_forecast_response(str(e), data_source)

def forecast_regime_transitions(
    returns: pd.Series = None,
    symbols: Optional[List[str]] = None,        # NEW: For FMP integration
    use_real_data: bool = False,               # NEW: Enable FMP data
    period: str = "1year",                     # NEW: Data period
    forecast_horizon: int = 21,
    n_regimes: int = 2
) -> Dict[str, Any]:
    """
    ENHANCED: Market regime transition forecasting with FMP integration
    
    Parameters:
    -----------
    returns : pd.Series, optional
        Historical return series (backward compatibility)
    symbols : List[str], optional
        List of symbols for FMP data retrieval (NEW)
    use_real_data : bool, default=False
        Use real market data via FMP (NEW)
    period : str, default="1year"
        Data period for real data (NEW)
    forecast_horizon : int, default=21
        Forecast horizon in days
    n_regimes : int, default=2
        Number of market regimes to model
        
    Returns:
    --------
    Dict[str, Any]
        Regime transition forecasting results with data source tracking
    """
    try:
        data_source = "Unknown"
        
        # Handle FMP real data integration
        if use_real_data and symbols and HAS_FMP_INTEGRATION:
            data_manager = _get_data_manager()
            if data_manager:
                real_returns_data = data_manager.get_portfolio_returns(symbols, period=period)
                if real_returns_data is not None:
                    # Create equal-weighted portfolio returns or use single asset
                    if len(symbols) == 1:
                        returns = real_returns_data[symbols[0]]
                    else:
                        returns = real_returns_data.mean(axis=1)
                    data_source = "Financial Modeling Prep"
                    logger.info(f"Using FMP data for regime analysis: {symbols}")
                else:
                    returns = _generate_synthetic_returns_single(symbols[0] if symbols else "SPY")
                    data_source = "Synthetic (FMP failed)"
            else:
                returns = _generate_synthetic_returns_single(symbols[0] if symbols else "SPY")
                data_source = "Synthetic (FMP unavailable)"
        elif returns is not None:
            data_source = "Provided"
        else:
            # Fallback to synthetic data
            symbol = symbols[0] if symbols else "SPY"
            returns = _generate_synthetic_returns_single(symbol)
            data_source = "Generated Synthetic"
        
        if not HAS_INTEGRATED_TOOLS:
            return _error_forecast_response("Regime tools not available", data_source)
        
        if returns is None or len(returns) < 100:
            return _empty_forecast_response("Insufficient data for regime analysis", data_source)
        
        # Detect current regimes using your existing function
        regime_results = detect_hmm_regimes(returns, n_regimes=n_regimes)
        if not regime_results:
            return _error_forecast_response("Regime detection failed", data_source)
        
        transition_forecasts = {
            'success': True,
            'forecast_type': 'regime_transitions',
            'data_source': data_source,  # NEW: Track data source
            'data_period': period if use_real_data else None,  # NEW: Track period used
            'symbols_analyzed': symbols if symbols else None,  # NEW: Track symbols
            'current_regime': regime_results['current_regime'],
            'regime_characteristics': regime_results['regime_characteristics'],
            'transition_matrix': regime_results['transition_matrix'],
            'regime_forecasts': {},
            'persistence_analysis': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Multi-step regime probabilities
        transition_matrix = np.array(regime_results['transition_matrix'])
        current_regime = regime_results['current_regime']
        
        regime_probabilities = []
        for step in range(1, min(forecast_horizon + 1, 64)):  # Cap at 64 steps for efficiency
            multi_step_matrix = np.linalg.matrix_power(transition_matrix, step)
            step_probabilities = multi_step_matrix[current_regime, :]
            regime_probabilities.append({
                'step': step,
                'regime_probabilities': step_probabilities.tolist(),
                'most_likely_regime': int(np.argmax(step_probabilities)),
                'probability_stay_current': float(step_probabilities[current_regime])
            })
        
        transition_forecasts['regime_forecasts'] = regime_probabilities
        
        # Persistence analysis
        regime_series = regime_results['regime_series']
        persistence_stats = _calculate_regime_persistence(regime_series)
        transition_forecasts['persistence_analysis'] = persistence_stats
        
        return transition_forecasts
        
    except Exception as e:
        logger.error(f"Regime transition forecasting failed: {e}")
        return _error_forecast_response(str(e), data_source)

def generate_scenario_forecasts(
    returns: pd.Series = None,
    symbols: Optional[List[str]] = None,        # NEW: For FMP integration
    use_real_data: bool = False,               # NEW: Enable FMP data
    period: str = "1year",                     # NEW: Data period
    forecast_horizon: int = 21,
    scenarios: Optional[Dict[str, float]] = None,
    monte_carlo_paths: int = 1000
) -> Dict[str, Any]:
    """
    ENHANCED: Generate scenario-based forecasts for stress testing with FMP integration
    
    Parameters:
    -----------
    returns : pd.Series, optional
        Historical return series (backward compatibility)
    symbols : List[str], optional
        List of symbols for FMP data retrieval (NEW)
    use_real_data : bool, default=False
        Use real market data via FMP (NEW)
    period : str, default="1year"
        Data period for real data (NEW)
    forecast_horizon : int, default=21
        Forecast horizon in days
    scenarios : Dict[str, float], optional
        Custom shock scenarios {name: shock_magnitude}
    monte_carlo_paths : int, default=1000
        Number of Monte Carlo simulation paths
        
    Returns:
    --------
    Dict[str, Any]
        Scenario-based forecasting results with data source tracking
    """
    try:
        data_source = "Unknown"
        
        # Handle FMP real data integration
        if use_real_data and symbols and HAS_FMP_INTEGRATION:
            data_manager = _get_data_manager()
            if data_manager:
                real_returns_data = data_manager.get_portfolio_returns(symbols, period=period)
                if real_returns_data is not None:
                    # Create equal-weighted portfolio returns or use single asset
                    if len(symbols) == 1:
                        returns = real_returns_data[symbols[0]]
                    else:
                        returns = real_returns_data.mean(axis=1)
                    data_source = "Financial Modeling Prep"
                    logger.info(f"Using FMP data for scenario analysis: {symbols}")
                else:
                    returns = _generate_synthetic_returns_single(symbols[0] if symbols else "SPY")
                    data_source = "Synthetic (FMP failed)"
            else:
                returns = _generate_synthetic_returns_single(symbols[0] if symbols else "SPY")
                data_source = "Synthetic (FMP unavailable)"
        elif returns is not None:
            data_source = "Provided"
        else:
            # Fallback to synthetic data
            symbol = symbols[0] if symbols else "SPY"
            returns = _generate_synthetic_returns_single(symbol)
            data_source = "Generated Synthetic"
        
        if returns is None or returns.empty or len(returns) < 30:
            return _empty_forecast_response("Insufficient data for scenario analysis", data_source)
        
        if scenarios is None:
            scenarios = {
                'base_case': 0.0,
                'mild_stress': -0.05,
                'moderate_stress': -0.10,
                'severe_stress': -0.20,
                'tail_risk': -0.30
            }
        
        scenario_results = {
            'success': True,
            'forecast_type': 'scenario_analysis',
            'forecast_horizon': forecast_horizon,
            'monte_carlo_paths': monte_carlo_paths,
            'data_source': data_source,  # NEW: Track data source
            'data_period': period if use_real_data else None,  # NEW: Track period used
            'symbols_analyzed': symbols if symbols else None,  # NEW: Track symbols
            'scenarios': {},
            'monte_carlo_analysis': {},
            'risk_metrics': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate base forecast for each scenario
        base_mean = returns.mean()
        base_vol = returns.std()
        
        for scenario_name, shock_magnitude in scenarios.items():
            # Adjust mean return based on scenario
            adjusted_mean = base_mean + (shock_magnitude / forecast_horizon)
            
            # Generate scenario-specific forecasts
            scenario_forecast = _generate_scenario_specific_forecast(
                adjusted_mean, base_vol, forecast_horizon, monte_carlo_paths
            )
            
            scenario_results['scenarios'][scenario_name] = {
                'shock_magnitude': shock_magnitude,
                'adjusted_daily_return': adjusted_mean,
                'expected_cumulative_return': adjusted_mean * forecast_horizon,
                'forecast_paths': scenario_forecast['paths_summary'],
                'confidence_intervals': scenario_forecast['confidence_intervals'],
                'risk_metrics': scenario_forecast['risk_metrics']
            }
        
        # Overall Monte Carlo analysis
        scenario_results['monte_carlo_analysis'] = _generate_monte_carlo_summary(
            base_mean, base_vol, forecast_horizon, monte_carlo_paths
        )
        
        return scenario_results
        
    except Exception as e:
        logger.error(f"Scenario forecasting failed: {e}")
        return _error_forecast_response(str(e), data_source)

def generate_comprehensive_forecast(
    returns: pd.Series = None,
    symbols: Optional[List[str]] = None,        # NEW: For FMP integration
    use_real_data: bool = False,               # NEW: Enable FMP data
    period: str = "1year",                     # NEW: Data period
    forecast_horizon: int = 21,
    include_volatility: bool = True,
    include_regimes: bool = True,
    include_scenarios: bool = True,
) -> Dict[str, Any]:
    """
    ENHANCED: Convenience function for comprehensive forecasting analysis with FMP integration
    """
    try:
        data_source = "Unknown"
        
        # Handle FMP real data integration
        if use_real_data and symbols and HAS_FMP_INTEGRATION:
            data_manager = _get_data_manager()
            if data_manager:
                real_returns_data = data_manager.get_portfolio_returns(symbols, period=period)
                if real_returns_data is not None:
                    # Create equal-weighted portfolio returns or use single asset
                    if len(symbols) == 1:
                        returns = real_returns_data[symbols[0]]
                    else:
                        returns = real_returns_data.mean(axis=1)
                    data_source = "Financial Modeling Prep"
                    logger.info(f"Using FMP data for comprehensive forecasting: {symbols}")
                else:
                    returns = _generate_synthetic_returns_single(symbols[0] if symbols else "SPY")
                    data_source = "Synthetic (FMP failed)"
            else:
                returns = _generate_synthetic_returns_single(symbols[0] if symbols else "SPY")
                data_source = "Synthetic (FMP unavailable)"
        elif returns is not None:
            data_source = "Provided"
        else:
            # Fallback to synthetic data
            symbol = symbols[0] if symbols else "SPY"
            returns = _generate_synthetic_returns_single(symbol)
            data_source = "Generated Synthetic"
        
        comprehensive_results = {
            'success': True,
            'forecast_horizon': forecast_horizon,
            'data_source': data_source,  # NEW: Track data source
            'data_period': period if use_real_data else None,  # NEW: Track period used
            'symbols_analyzed': symbols if symbols else None,  # NEW: Track symbols
            'return_forecast': {},
            'volatility_forecast': {},
            'regime_analysis': {},
            'scenario_analysis': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Return forecasting
        return_forecast = forecast_portfolio_returns(
            returns=returns, symbols=symbols, use_real_data=False, period=period, 
            forecast_horizon=forecast_horizon
        )
        if return_forecast.get('success'):
            comprehensive_results['return_forecast'] = return_forecast
        
        # Volatility forecasting
        if include_volatility:
            vol_forecast = forecast_volatility_with_regimes(
                returns=returns, symbols=symbols, use_real_data=False, period=period,
                forecast_horizon=forecast_horizon
            )
            if vol_forecast.get('success'):
                comprehensive_results['volatility_forecast'] = vol_forecast
        
        # Regime analysis
        if include_regimes:
            regime_forecast = forecast_regime_transitions(
                returns=returns, symbols=symbols, use_real_data=False, period=period,
                forecast_horizon=forecast_horizon
            )
            if regime_forecast.get('success'):
                comprehensive_results['regime_analysis'] = regime_forecast
        
        # Scenario analysis
        if include_scenarios:
            scenario_forecast = generate_scenario_forecasts(
                returns=returns, symbols=symbols, use_real_data=False, period=period,
                forecast_horizon=forecast_horizon
            )
            if scenario_forecast.get('success'):
                comprehensive_results['scenario_analysis'] = scenario_forecast
        
        return comprehensive_results
        
    except Exception as e:
        logger.error(f"Comprehensive forecasting failed: {e}")
        return _error_forecast_response(str(e), data_source)

# ============================================================================
# HELPER FUNCTIONS - ENHANCED FOR FMP INTEGRATION
# ============================================================================

def _generate_synthetic_returns_single(symbol: str, days: int = 252) -> pd.Series:
    """Generate synthetic returns for a single symbol"""
    np.random.seed(42)  # Reproducible results
    
    dates = pd.date_range(end='today', periods=days, freq='D')
    
    # Asset-specific characteristics
    if symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']:
        mean_return = 0.0008
        volatility = 0.025
    elif symbol in ['TLT', 'AGG', 'BND']:
        mean_return = 0.0003
        volatility = 0.008
    else:
        mean_return = 0.0006
        volatility = 0.018
    
    returns = np.random.normal(mean_return, volatility, days)
    return pd.Series(returns, index=dates, name=f'{symbol}_returns')

def get_forecasting_tools_integration_status() -> Dict[str, Any]:
    """
    Get integration status for forecasting tools with FMP
    
    Returns:
        Dictionary with integration status and available functions
    """
    return {
        "fmp_integration_available": HAS_FMP_INTEGRATION,
        "integrated_functions": [
            "forecast_portfolio_returns",
            "forecast_volatility_with_regimes", 
            "forecast_regime_transitions",
            "generate_scenario_forecasts",
            "generate_comprehensive_forecast"
        ],
        "integration_parameters": {
            "use_real_data": "bool - Enable FMP data integration",
            "symbols": "List[str] - Symbols for real data retrieval",
            "period": "str - Data period (1month, 3months, 1year, 2years, 5years)"
        },
        "supported_data_sources": [
            "Financial Modeling Prep (Real)",
            "Synthetic (Fallback)",
            "Provided (User data)"
        ],
        "dependencies": {
            "statsmodels": HAS_STATSMODELS,
            "arch": HAS_ARCH,
            "integrated_tools": HAS_INTEGRATED_TOOLS
        }
    }

# ============================================================================
# EXISTING HELPER FUNCTIONS - UNCHANGED FOR BACKWARD COMPATIBILITY
# ============================================================================

def _generate_base_forecast(returns: pd.Series, horizon: int, model_type: str) -> Dict[str, Any]:
    """Generate base forecast using specified model"""
    try:
        if model_type == "auto_arima" and HAS_STATSMODELS:
            return _generate_auto_arima_forecast(returns, horizon)
        elif model_type == "arima_garch" and HAS_STATSMODELS and HAS_ARCH:
            return _generate_arima_garch_forecast(returns, horizon)
        elif model_type == "simple_mean":
            return _generate_simple_mean_forecast(returns, horizon)
        else:
            # Fallback to simple mean
            return _generate_simple_mean_forecast(returns, horizon)
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

def _generate_auto_arima_forecast(returns: pd.Series, horizon: int) -> Dict[str, Any]:
    """Auto ARIMA forecasting with model selection"""
    try:
        returns_clean = returns.dropna()
        
        # Stationarity check
        adf_result = adfuller(returns_clean)
        is_stationary = adf_result[1] < 0.05
        
        # Simple ARIMA model selection
        best_model = None
        best_aic = np.inf
        
        # Try different ARIMA specifications
        arima_orders = [(1, 0, 1), (2, 0, 1), (1, 0, 2)] if is_stationary else [(1, 1, 1), (2, 1, 1), (1, 1, 2)]
        
        for order in arima_orders:
            try:
                model = ARIMA(returns_clean, order=order)
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_model = fitted
            except:
                continue
        
        if best_model is None:
            return {'success': False, 'error': 'ARIMA model fitting failed'}
        
        # Generate forecast with enhanced error handling
        try:
            forecast_result = best_model.forecast(steps=horizon, return_conf_int=True)
            
            # Handle different return formats from statsmodels
            if isinstance(forecast_result, tuple):
                forecast_values = forecast_result[0]
                conf_intervals = forecast_result[1] if len(forecast_result) > 1 else None
            else:
                forecast_values = forecast_result
                conf_intervals = None
                
        except (IndexError, ValueError) as e:
            # Fallback to simple forecast without confidence intervals
            forecast_result = best_model.forecast(steps=horizon)
            forecast_values = forecast_result
            conf_intervals = None
        
        # Ensure forecast_values is array-like
        if hasattr(forecast_values, 'values'):
            forecast_values = forecast_values.values
        elif not hasattr(forecast_values, '__len__'):
            forecast_values = [forecast_values] * horizon
        
        # Convert to list if it's not already
        if not isinstance(forecast_values, list):
            forecast_values = forecast_values.tolist()
        
        # Generate confidence intervals safely
        if conf_intervals is not None and hasattr(conf_intervals, 'shape') and len(conf_intervals.shape) == 2:
            try:
                confidence_intervals = {
                    'lower_95': conf_intervals[:, 0].tolist(),
                    'upper_95': conf_intervals[:, 1].tolist()
                }
            except (IndexError, ValueError):
                # Fallback confidence intervals
                forecast_std = np.std(forecast_values) if len(forecast_values) > 1 else 0.02
                confidence_intervals = {
                    'lower_95': [float(val - 1.96 * forecast_std) for val in forecast_values],
                    'upper_95': [float(val + 1.96 * forecast_std) for val in forecast_values]
                }
        else:
            # Generate simple confidence intervals based on forecast values
            forecast_std = np.std(forecast_values) if len(forecast_values) > 1 else 0.02
            confidence_intervals = {
                'lower_95': [float(val - 1.96 * forecast_std) for val in forecast_values],
                'upper_95': [float(val + 1.96 * forecast_std) for val in forecast_values]
            }
        
        return {
            'success': True,
            'forecasts': forecast_values,
            'confidence_intervals': confidence_intervals,
            'model_diagnostics': {
                'model_order': best_model.model.order,
                'aic': float(best_model.aic),
                'bic': float(best_model.bic),
                'is_stationary': is_stationary
            }
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def _generate_arima_garch_forecast(returns: pd.Series, horizon: int) -> Dict[str, Any]:
    """ARIMA-GARCH hybrid forecasting"""
    try:
        returns_clean = returns.dropna()
        
        # Fit ARIMA for mean
        arima_model = ARIMA(returns_clean, order=(1, 0, 1))
        arima_fitted = arima_model.fit()
        
        # Fit GARCH on residuals
        residuals = arima_fitted.resid
        garch_model = arch_model(residuals * 100, vol='Garch', p=1, q=1)
        garch_fitted = garch_model.fit(disp='off')
        
        # Generate forecasts
        mean_forecast = arima_fitted.forecast(steps=horizon)
        vol_forecast = garch_fitted.forecast(horizon=horizon)
        
        # Extract volatility forecast
        vol_values = np.sqrt(vol_forecast.variance.iloc[0].values) / 100
        
        # Generate confidence intervals
        confidence_intervals = {}
        for i, (mean_val, vol_val) in enumerate(zip(mean_forecast, vol_values)):
            confidence_intervals[f'day_{i+1}'] = {
                'mean': float(mean_val),
                'volatility': float(vol_val),
                'lower_95': float(mean_val - 1.96 * vol_val),
                'upper_95': float(mean_val + 1.96 * vol_val)
            }
        
        return {
            'success': True,
            'forecasts': mean_forecast.tolist(),
            'volatility_forecasts': vol_values.tolist(),
            'confidence_intervals': confidence_intervals,
            'model_diagnostics': {
                'arima_aic': arima_fitted.aic,
                'garch_aic': garch_fitted.aic,
                'combined_model': 'ARIMA(1,0,1)-GARCH(1,1)'
            }
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def _generate_simple_mean_forecast(returns: pd.Series, horizon: int) -> Dict[str, Any]:
    """Simple mean reversion forecast - fallback method"""
    try:
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generate constant forecast
        forecast_values = [mean_return] * horizon
        
        # Simple confidence intervals
        confidence_intervals = {
            'lower_95': [mean_return - 1.96 * std_return] * horizon,
            'upper_95': [mean_return + 1.96 * std_return] * horizon
        }
        
        return {
            'success': True,
            'forecasts': forecast_values,
            'confidence_intervals': confidence_intervals,
            'model_diagnostics': {
                'model_type': 'simple_mean',
                'historical_mean': mean_return,
                'historical_std': std_return
            }
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def _generate_regime_conditional_forecast(returns: pd.Series, horizon: int, confidence_levels: List[float]) -> Dict[str, Any]:
    """Generate regime-conditional forecasts using your regime tools"""
    try:
        # Detect regimes using your existing function
        regime_results = detect_hmm_regimes(returns, n_regimes=2)
        
        # Check if regime_results is valid
        if not regime_results:
            return {'success': False, 'error': 'Regime detection failed - no results returned'}
        
        # Check if regime detection was successful
        if not regime_results.get('success', False):
            return {'success': False, 'error': 'Regime detection failed - regime analysis unsuccessful'}
        
        # Safely extract regime information with error checking
        current_regime = regime_results.get('current_regime')
        regime_chars = regime_results.get('regime_characteristics')
        transition_matrix_data = regime_results.get('transition_matrix')
        
        # Validate all required components exist
        if current_regime is None:
            return {'success': False, 'error': 'No current regime identified'}
        
        if not regime_chars:
            return {'success': False, 'error': 'No regime characteristics available'}
        
        if not transition_matrix_data:
            return {'success': False, 'error': 'No transition matrix available'}
        
        # Convert transition matrix safely
        try:
            transition_matrix = np.array(transition_matrix_data)
            if transition_matrix.size == 0:
                return {'success': False, 'error': 'Empty transition matrix'}
        except Exception as e:
            return {'success': False, 'error': f'Invalid transition matrix: {str(e)}'}
        
        # Generate regime-conditional forecasts
        regime_forecasts = {}
        
        for step in range(1, horizon + 1):
            try:
                # Calculate regime probabilities for this step
                multi_step_matrix = np.linalg.matrix_power(transition_matrix, step)
                regime_probs = multi_step_matrix[current_regime, :]
                
                # Weighted forecast based on regime probabilities
                expected_return = 0.0
                expected_vol = 0.0
                
                for regime, prob in enumerate(regime_probs):
                    if regime in regime_chars:
                        expected_return += prob * regime_chars[regime].get('mean_return', 0.0)
                        expected_vol += prob * regime_chars[regime].get('volatility', 0.02)
                
                regime_forecasts[f'step_{step}'] = {
                    'expected_return': float(expected_return),
                    'expected_volatility': float(expected_vol),
                    'regime_probabilities': [float(p) for p in regime_probs]
                }
                
            except Exception as e:
                # If any step fails, use simple fallback
                regime_forecasts[f'step_{step}'] = {
                    'expected_return': 0.001,  # Default return
                    'expected_volatility': 0.02,  # Default volatility
                    'regime_probabilities': [0.5, 0.5]  # Default equal probabilities
                }
        
        # Extract forecasts as list
        forecast_values = [regime_forecasts[f'step_{i+1}']['expected_return'] for i in range(horizon)]
        
        return {
            'success': True,
            'forecasts': forecast_values,
            'regime_analysis': {
                'current_regime': current_regime,
                'regime_characteristics': regime_chars,
                'step_by_step_analysis': regime_forecasts
            }
        }
        
    except Exception as e:
        return {'success': False, 'error': f'Regime conditional forecasting failed: {str(e)}'}

def _generate_regime_switching_volatility(returns: pd.Series, horizon: int) -> Dict[str, Any]:
    """Generate regime-switching volatility forecast"""
    try:
        vol_regimes = detect_volatility_regimes(returns)
        if vol_regimes.get('error'):
            return {'success': False, 'error': vol_regimes['error']}
        
        current_regime = vol_regimes['current_regime']
        regime_stats = vol_regimes['regime_statistics']
        
        # Simple persistence assumption
        vol_forecasts = []
        if current_regime in regime_stats:
            regime_vol = regime_stats[current_regime]['avg_volatility']
            vol_forecasts = [regime_vol / np.sqrt(252)] * horizon  # Convert to daily
        else:
            vol_forecasts = [returns.std()] * horizon
        
        return {
            'success': True,
            'regime_volatility': {
                'current_regime': current_regime,
                'forecasted_volatilities': vol_forecasts,
                'regime_characteristics': regime_stats
            }
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def _generate_rolling_volatility_forecast(returns: pd.Series, horizon: int) -> Dict[str, float]:
    """Generate simple rolling volatility forecast"""
    try:
        # Use last 30-day volatility as forecast
        recent_vol = returns.tail(30).std()
        
        return {
            'daily_volatilities': [recent_vol] * horizon,
            'mean_volatility': recent_vol,
            'volatility_trend': 'Constant',
            'model_type': 'rolling_30d'
        }
        
    except Exception as e:
        return {
            'daily_volatilities': [0.02] * horizon,
            'mean_volatility': 0.02,
            'error': str(e)
        }

def _generate_scenario_specific_forecast(mean: float, vol: float, horizon: int, n_paths: int) -> Dict[str, Any]:
    """Generate Monte Carlo paths for specific scenario"""
    try:
        # Generate random paths
        random_paths = np.random.normal(mean, vol, (n_paths, horizon))
        cumulative_returns = np.cumsum(random_paths, axis=1)
        final_returns = cumulative_returns[:, -1]
        
        # Path statistics
        paths_summary = {
            'mean_final_return': float(np.mean(final_returns)),
            'median_final_return': float(np.median(final_returns)),
            'std_final_return': float(np.std(final_returns)),
            'min_final_return': float(np.min(final_returns)),
            'max_final_return': float(np.max(final_returns))
        }
        
        # Confidence intervals
        confidence_intervals = {}
        percentiles = [5, 25, 50, 75, 95]
        for p in percentiles:
            confidence_intervals[f'p{p}'] = float(np.percentile(final_returns, p))
        
        # Risk metrics
        risk_metrics = {
            'probability_loss': float(np.mean(final_returns < 0)),
            'var_5': float(np.percentile(final_returns, 5)),
            'cvar_5': float(np.mean(final_returns[final_returns <= np.percentile(final_returns, 5)]))
        }
        
        return {
            'paths_summary': paths_summary,
            'confidence_intervals': confidence_intervals,
            'risk_metrics': risk_metrics
        }
        
    except Exception as e:
        return {'error': str(e)}

def _generate_monte_carlo_summary(mean: float, vol: float, horizon: int, n_paths: int) -> Dict[str, Any]:
    """Generate overall Monte Carlo analysis summary"""
    try:
        # Generate paths
        paths = np.random.normal(mean, vol, (n_paths, horizon))
        cumulative_paths = np.cumsum(paths, axis=1)
        
        # Summary statistics
        final_values = cumulative_paths[:, -1]
        
        return {
            'simulation_params': {
                'n_paths': n_paths,
                'horizon_days': horizon,
                'daily_mean': mean,
                'daily_volatility': vol
            },
            'results': {
                'expected_return': float(np.mean(final_values)),
                'return_volatility': float(np.std(final_values)),
                'probability_positive': float(np.mean(final_values > 0)),
                'worst_case_5th_percentile': float(np.percentile(final_values, 5)),
                'best_case_95th_percentile': float(np.percentile(final_values, 95))
            }
        }
        
    except Exception as e:
        return {'error': str(e)}

def _calculate_regime_persistence(regime_series: pd.Series) -> Dict[str, Any]:
    """Calculate regime persistence statistics"""
    try:
        # Find regime transitions
        transitions = regime_series != regime_series.shift(1)
        transition_points = transitions[transitions].index
        
        # Calculate durations for each regime episode
        regime_durations = {}
        
        if len(transition_points) > 1:
            for i in range(len(transition_points) - 1):
                start_idx = transition_points[i]
                end_idx = transition_points[i + 1]
                regime_value = regime_series.loc[start_idx]
                duration = (end_idx - start_idx).days if hasattr(end_idx - start_idx, 'days') else int(end_idx - start_idx)
                
                if regime_value not in regime_durations:
                    regime_durations[regime_value] = []
                regime_durations[regime_value].append(duration)
        
        # Calculate statistics
        persistence_stats = {}
        for regime, durations in regime_durations.items():
            if durations:
                persistence_stats[f'regime_{regime}'] = {
                    'avg_duration': float(np.mean(durations)),
                    'median_duration': float(np.median(durations)),
                    'max_duration': int(np.max(durations)),
                    'min_duration': int(np.min(durations)),
                    'episodes': len(durations)
                }
        
        return {
            'regime_persistence': persistence_stats,
            'total_transitions': len(transition_points) - 1 if len(transition_points) > 1 else 0,
            'analysis_period': len(regime_series)
        }
        
    except Exception as e:
        return {'error': str(e)}

def _extract_volatility_series(garch_result: Dict, horizon: int) -> List[float]:
    """Extract volatility series from GARCH result"""
    try:
        if 'forecast_series' in garch_result:
            series = garch_result['forecast_series']
            if hasattr(series, 'values'):
                return series.values[:horizon].tolist()
            elif isinstance(series, list):
                return series[:horizon]
        
        # Fallback
        mean_vol = garch_result.get('mean_forecast_daily_vol', 0.02)
        return [mean_vol] * horizon
        
    except Exception:
        return [0.02] * horizon

# ============================================================================
# ERROR HANDLING FUNCTIONS - ENHANCED WITH DATA SOURCE TRACKING
# ============================================================================

def _empty_forecast_response(message: str, data_source: str = "Unknown") -> Dict[str, Any]:
    """Empty forecast response for insufficient data"""
    return {
        'success': False,
        'error': message,
        'forecast_type': 'empty',
        'data_source': data_source,  # NEW: Track data source
        'forecasts': [],
        'confidence_intervals': {},
        'timestamp': datetime.now().isoformat()
    }

def _error_forecast_response(error: str, data_source: str = "Unknown") -> Dict[str, Any]:
    """Error response for forecasting failures"""
    return {
        'success': False,
        'error': error,
        'forecast_type': 'error',
        'data_source': data_source,  # NEW: Track data source
        'forecasts': [],
        'confidence_intervals': {},
        'timestamp': datetime.now().isoformat()
    }