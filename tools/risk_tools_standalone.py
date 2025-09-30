"""
Standalone Risk Analysis Tools with FMP Integration
==================================================

This module provides comprehensive risk analysis tools with real market data integration.
Created as standalone version to avoid circular imports and integrate FMP data.

Key Features:
- Real market data from Financial Modeling Prep (FMP)
- Comprehensive risk metrics (VaR, CVaR, Maximum Drawdown, Beta, etc.)
- Stress testing and scenario analysis
- GARCH volatility modeling
- Risk attribution analysis
- Automatic fallback to synthetic data if FMP unavailable
"""

import asyncio
import importlib.util
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
import logging
from arch import arch_model
from scipy import stats
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# FMP Data Integration (Direct Import to Avoid Circular Dependencies)
# ============================================================================

def import_fmp_tools():
    """Import FMP integration with detailed debugging"""
    try:
        import os
        import sys
        
        # Get the project root directory (one level up from tools)
        current_file = os.path.abspath(__file__)
        tools_dir = os.path.dirname(current_file)
        project_root = os.path.dirname(tools_dir)
        fmp_path = os.path.join(project_root, "data", "providers", "fmp_integration.py")
        
        print(f"DEBUG: Current file: {current_file}")
        print(f"DEBUG: Tools directory: {tools_dir}")
        print(f"DEBUG: Project root: {project_root}")
        print(f"DEBUG: Looking for FMP at: {fmp_path}")
        print(f"DEBUG: FMP file exists: {os.path.exists(fmp_path)}")
        
        # Check if file exists
        if not os.path.exists(fmp_path):
            print(f"DEBUG: FMP integration file not found!")
            # Try to find it elsewhere
            for root, dirs, files in os.walk(project_root):
                if "fmp_integration.py" in files:
                    print(f"DEBUG: Found fmp_integration.py at: {os.path.join(root, 'fmp_integration.py')}")
            return None, None
        
        # Direct file import with correct path
        spec = importlib.util.spec_from_file_location("fmp_integration", fmp_path)
        fmp_module = importlib.util.module_from_spec(spec)
        
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        print("DEBUG: About to execute FMP module...")
        spec.loader.exec_module(fmp_module)
        print("DEBUG: FMP module loaded successfully")
        
        return fmp_module.PortfolioDataManager, fmp_module.FMPDataProvider
    except Exception as e:
        print(f"DEBUG: Error importing FMP: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ADD THIS LINE - Execute the import and assign to module-level variables
PortfolioDataManager, FMPDataProvider = import_fmp_tools()

def get_data_manager():
    """Get data manager with FMP integration"""
    if PortfolioDataManager and FMPDataProvider:
        try:
            fmp_provider = FMPDataProvider()
            return PortfolioDataManager(fmp_provider)
        except ValueError:
            logger.warning("FMP API key not available, using fallback only")
            return PortfolioDataManager()
    else:
        logger.warning("FMP integration not available")
        return None

# ============================================================================
# Data Classes for Standardized Responses
# ============================================================================

@dataclass
class RiskMetricsResult:
    """Standardized response for risk metrics calculations"""
    success: bool
    portfolio_var_95: Optional[float] = None
    portfolio_var_99: Optional[float] = None
    portfolio_cvar_95: Optional[float] = None
    portfolio_cvar_99: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    beta: Optional[float] = None
    annualized_volatility: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    data_source: str = "Synthetic"
    error: Optional[str] = None
    analysis_period: Optional[str] = None
    symbols_analyzed: Optional[List[str]] = None

@dataclass
class StressTestResult:
    """Standardized response for stress testing"""
    success: bool
    stress_scenarios: Optional[Dict[str, Any]] = None
    worst_case_scenario: Optional[str] = None
    resilience_score: Optional[float] = None
    monte_carlo_results: Optional[Dict[str, Any]] = None
    historical_stress_results: Optional[Dict[str, Any]] = None
    data_source: str = "Synthetic"
    error: Optional[str] = None

@dataclass
class GARCHForecastResult:
    """Standardized response for GARCH forecasting"""
    success: bool
    forecast_horizon: Optional[int] = None
    current_volatility: Optional[float] = None
    forecast_volatility: Optional[float] = None
    volatility_trend: Optional[str] = None
    confidence_bands: Optional[Dict[str, float]] = None
    model_aic: Optional[float] = None
    data_source: str = "Synthetic"
    error: Optional[str] = None

# ============================================================================
# Core Risk Analysis Functions with FMP Integration
# ============================================================================

async def calculate_comprehensive_risk(
    symbols: List[str],
    weights: Optional[Dict[str, float]] = None,
    period: str = "1year",
    confidence_levels: List[float] = [0.95, 0.99],
    use_real_data: bool = True
) -> RiskMetricsResult:
    try:
        if use_real_data:
            data_manager = get_data_manager()
            if data_manager:
                # Get real market data
                returns_data, data_source = await data_manager.get_returns_data(symbols, period)
                
                # ADD DEBUG LOGGING HERE
                logger.info(f"Returns data shape: {returns_data.shape if returns_data is not None else 'None'}")
                logger.info(f"Returns data columns: {returns_data.columns.tolist() if returns_data is not None else 'None'}")
                logger.info(f"Returns data head:\n{returns_data.head() if returns_data is not None else 'None'}")
                
                if returns_data is not None:
                    # Calculate portfolio returns
                    if weights:
                        # ... weight validation code ...
                        portfolio_returns = pd.Series(0, index=returns_data.index)
                        for symbol in symbols:
                            if symbol in returns_data.columns and symbol in weights:
                                portfolio_returns += returns_data[symbol] * weights[symbol]
                    else:
                        # Equal weighted portfolio
                        portfolio_returns = returns_data.mean(axis=1)
                    
                    # ADD DEBUG HERE
                    logger.info(f"Portfolio returns shape: {portfolio_returns.shape}")
                    logger.info(f"Portfolio returns mean: {portfolio_returns.mean():.6f}")
                    logger.info(f"Portfolio returns head: {portfolio_returns.head().tolist()}")
                    
                    # Calculate risk metrics
                    risk_result = _calculate_risk_metrics_from_returns(
                        portfolio_returns, confidence_levels
                    )
                    
                    return RiskMetricsResult(
                        success=True,
                        portfolio_var_95=risk_result.get('var_95'),
                        portfolio_var_99=risk_result.get('var_99'),
                        portfolio_cvar_95=risk_result.get('cvar_95'),
                        portfolio_cvar_99=risk_result.get('cvar_99'),
                        max_drawdown_pct=risk_result.get('max_drawdown_pct'),
                        sharpe_ratio=risk_result.get('sharpe_ratio'),
                        sortino_ratio=risk_result.get('sortino_ratio'),
                        annualized_volatility=risk_result.get('annualized_volatility'),
                        skewness=risk_result.get('skewness'),
                        kurtosis=risk_result.get('kurtosis'),
                        data_source=data_source,
                        analysis_period=period,
                        symbols_analyzed=symbols
                    )
        
        # Fallback to synthetic data
        synthetic_returns = _generate_synthetic_portfolio_returns(symbols, period, weights)
        risk_result = _calculate_risk_metrics_from_returns(synthetic_returns, confidence_levels)
        
        return RiskMetricsResult(
            success=True,
            portfolio_var_95=risk_result.get('var_95'),
            portfolio_var_99=risk_result.get('var_99'),
            portfolio_cvar_95=risk_result.get('cvar_95'),
            portfolio_cvar_99=risk_result.get('cvar_99'),
            max_drawdown_pct=risk_result.get('max_drawdown_pct'),
            sharpe_ratio=risk_result.get('sharpe_ratio'),
            sortino_ratio=risk_result.get('sortino_ratio'),
            annualized_volatility=risk_result.get('annualized_volatility'),
            skewness=risk_result.get('skewness'),
            kurtosis=risk_result.get('kurtosis'),
            data_source="Synthetic (FMP unavailable)",
            analysis_period=period,
            symbols_analyzed=symbols
        )
        
    except Exception as e:
        return RiskMetricsResult(
            success=False,
            error=f"Risk calculation failed: {str(e)}"
        )

async def estimate_var_cvar(
    symbols: List[str],
    weights: Optional[Dict[str, float]] = None,
    confidence_levels: List[float] = [0.95, 0.99],
    period: str = "1year",
    use_real_data: bool = True
) -> Dict[str, Any]:
    """
    Estimate Value at Risk (VaR) and Conditional VaR with real data
    
    Parameters:
    -----------
    symbols : List[str]
        Portfolio symbols
    weights : Optional[Dict[str, float]]
        Portfolio weights
    confidence_levels : List[float]
        Confidence levels for estimation
    period : str
        Data period for analysis
    use_real_data : bool
        Whether to use real FMP data
        
    Returns:
    --------
    Dict[str, Any] : VaR/CVaR estimates with data source tracking
    """
    try:
        # Get comprehensive risk analysis
        risk_result = await calculate_comprehensive_risk(
            symbols, weights, period, confidence_levels, use_real_data
        )
        
        if not risk_result.success:
            return {
                'success': False,
                'error': risk_result.error
            }
        
        # Format VaR/CVaR results
        var_cvar_results = {}
        for conf in confidence_levels:
            conf_key = f"{int(conf*100)}%"
            
            if conf == 0.95:
                var_val = risk_result.portfolio_var_95
                cvar_val = risk_result.portfolio_cvar_95
            elif conf == 0.99:
                var_val = risk_result.portfolio_var_99
                cvar_val = risk_result.portfolio_cvar_99
            else:
                # Calculate for other confidence levels if needed
                var_val = None
                cvar_val = None
            
            var_cvar_results[conf_key] = {
                'var': var_val,
                'cvar': cvar_val,
                'var_pct': var_val * 100 if var_val else None,
                'cvar_pct': cvar_val * 100 if cvar_val else None
            }
        
        return {
            'success': True,
            'var_cvar_estimates': var_cvar_results,
            'data_source': risk_result.data_source,
            'analysis_period': period,
            'symbols': symbols,
            'portfolio_weights': weights or {symbol: 1/len(symbols) for symbol in symbols}
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"VaR/CVaR estimation failed: {str(e)}"
        }

async def perform_stress_testing(
    symbols: List[str],
    weights: Optional[Dict[str, float]] = None,
    stress_scenarios: Optional[Dict[str, float]] = None,
    period: str = "1year",
    use_real_data: bool = True,
    monte_carlo_simulations: int = 1000
) -> StressTestResult:
    """
    Perform comprehensive stress testing with real market data
    
    Parameters:
    -----------
    symbols : List[str]
        Portfolio symbols
    weights : Optional[Dict[str, float]]
        Portfolio weights
    stress_scenarios : Optional[Dict[str, float]]
        Custom stress scenarios (name -> shock magnitude)
    period : str
        Analysis period
    use_real_data : bool
        Whether to use real market data
    monte_carlo_simulations : int
        Number of Monte Carlo simulations
        
    Returns:
    --------
    StressTestResult : Comprehensive stress testing results
    """
    try:
        # Default stress scenarios
        if stress_scenarios is None:
            stress_scenarios = {
                "Market Crash 2008": -0.37,
                "COVID Crash 2020": -0.34,
                "Flash Crash 2010": -0.09,
                "Severe Correction": -0.20,
                "Mild Correction": -0.10
            }
        
        stress_results = {}
        
        if use_real_data:
            data_manager = get_data_manager()
            if data_manager:
                returns_data, data_source = await data_manager.get_returns_data(symbols, period)
                
                if returns_data is not None:
                    # Calculate portfolio returns
                    if weights:
                        portfolio_returns = pd.Series(0, index=returns_data.index)
                        for symbol in symbols:
                            if symbol in returns_data.columns and symbol in weights:
                                portfolio_returns += returns_data[symbol] * weights[symbol]
                    else:
                        portfolio_returns = returns_data.mean(axis=1)
                    
                    # Run stress scenarios
                    for scenario_name, shock_magnitude in stress_scenarios.items():
                        stress_results[scenario_name] = _apply_stress_scenario(
                            portfolio_returns, shock_magnitude
                        )
                    
                    # Monte Carlo stress testing
                    monte_carlo_results = _run_monte_carlo_stress(
                        portfolio_returns, monte_carlo_simulations
                    )
                    
                    # Determine worst case scenario
                    worst_scenario = min(
                        stress_results.keys(), 
                        key=lambda x: stress_results[x]['total_loss_pct']
                    )
                    
                    # Calculate resilience score (0-100)
                    avg_loss = np.mean([result['total_loss_pct'] for result in stress_results.values()])
                    resilience_score = max(0, 100 + avg_loss * 2)  # Convert negative loss to positive score
                    
                    return StressTestResult(
                        success=True,
                        stress_scenarios=stress_results,
                        worst_case_scenario=worst_scenario,
                        resilience_score=resilience_score,
                        monte_carlo_results=monte_carlo_results,
                        data_source=data_source
                    )
        
        # Fallback to synthetic data
        synthetic_returns = _generate_synthetic_portfolio_returns(symbols, period, weights)
        
        for scenario_name, shock_magnitude in stress_scenarios.items():
            stress_results[scenario_name] = _apply_stress_scenario(
                synthetic_returns, shock_magnitude
            )
        
        monte_carlo_results = _run_monte_carlo_stress(
            synthetic_returns, monte_carlo_simulations
        )
        
        worst_scenario = min(
            stress_results.keys(), 
            key=lambda x: stress_results[x]['total_loss_pct']
        )
        
        avg_loss = np.mean([result['total_loss_pct'] for result in stress_results.values()])
        resilience_score = max(0, 100 + avg_loss * 2)
        
        return StressTestResult(
            success=True,
            stress_scenarios=stress_results,
            worst_case_scenario=worst_scenario,
            resilience_score=resilience_score,
            monte_carlo_results=monte_carlo_results,
            data_source="Synthetic (FMP unavailable)"
        )
        
    except Exception as e:
        return StressTestResult(
            success=False,
            error=f"Stress testing failed: {str(e)}"
        )

async def forecast_volatility_garch(
    symbols: List[str],
    weights: Optional[Dict[str, float]] = None,
    forecast_horizon: int = 30,
    period: str = "1year",
    use_real_data: bool = True
) -> GARCHForecastResult:
    """
    Forecast volatility using GARCH models with real market data
    
    Parameters:
    -----------
    symbols : List[str]
        Portfolio symbols
    weights : Optional[Dict[str, float]]
        Portfolio weights
    forecast_horizon : int
        Number of days to forecast
    period : str
        Historical data period
    use_real_data : bool
        Whether to use real market data
        
    Returns:
    --------
    GARCHForecastResult : GARCH volatility forecast
    """
    try:
        portfolio_returns = None
        data_source = "Synthetic"
        
        if use_real_data:
            data_manager = get_data_manager()
            if data_manager:
                returns_data, data_source = await data_manager.get_returns_data(symbols, period)
                
                if returns_data is not None:
                    # Calculate portfolio returns
                    if weights:
                        portfolio_returns = pd.Series(0, index=returns_data.index)
                        for symbol in symbols:
                            if symbol in returns_data.columns and symbol in weights:
                                portfolio_returns += returns_data[symbol] * weights[symbol]
                    else:
                        portfolio_returns = returns_data.mean(axis=1)
        
        if portfolio_returns is None:
            # Fallback to synthetic data
            portfolio_returns = _generate_synthetic_portfolio_returns(symbols, period, weights)
            data_source = "Synthetic (FMP unavailable)"
        
        # GARCH modeling
        if len(portfolio_returns) < 100:
            return GARCHForecastResult(
                success=False,
                error="Insufficient data for GARCH modeling (need at least 100 observations)"
            )
        
        # Fit GARCH model
        returns_pct = portfolio_returns.dropna() * 100
        
        try:
            model = arch_model(returns_pct, vol='Garch', p=1, q=1, dist='t')
            fitted_model = model.fit(disp='off', show_warning=False)
        except Exception as e:
            return GARCHForecastResult(
                success=False,
                error=f"GARCH model fitting failed: {str(e)}"
            )
        
        # Generate forecast
        forecast = fitted_model.forecast(horizon=forecast_horizon, reindex=False)
        future_vol_values = np.sqrt(forecast.variance.iloc[0].values) / 100
        
        current_vol = np.sqrt(fitted_model.conditional_volatility.iloc[-1]) / 100
        mean_forecast_vol = np.mean(future_vol_values)
        
        # Determine trend
        if mean_forecast_vol > current_vol * 1.05:
            trend = "Increasing"
        elif mean_forecast_vol < current_vol * 0.95:
            trend = "Decreasing"
        else:
            trend = "Stable"
        
        # Confidence bands
        forecast_std = np.std(future_vol_values)
        confidence_bands = {
            "lower_95": mean_forecast_vol - 1.96 * forecast_std,
            "upper_95": mean_forecast_vol + 1.96 * forecast_std,
            "lower_68": mean_forecast_vol - forecast_std,
            "upper_68": mean_forecast_vol + forecast_std
        }
        
        return GARCHForecastResult(
            success=True,
            forecast_horizon=forecast_horizon,
            current_volatility=current_vol,
            forecast_volatility=mean_forecast_vol,
            volatility_trend=trend,
            confidence_bands=confidence_bands,
            model_aic=fitted_model.aic,
            data_source=data_source
        )
        
    except Exception as e:
        return GARCHForecastResult(
            success=False,
            error=f"GARCH forecasting failed: {str(e)}"
        )

# ============================================================================
# Helper Functions
# ============================================================================

def _calculate_risk_metrics_from_returns(
    returns: pd.Series,
    confidence_levels: List[float] = [0.95, 0.99]
) -> Dict[str, Any]:
    """Calculate risk metrics from return series"""
    if returns.empty:
        logger.error("Empty returns series received!")
        return {}
    
    # ADD THESE DEBUG LINES
    logger.info(f"Calculating metrics from {len(returns)} return observations")
    logger.info(f"Returns mean: {returns.mean():.6f}, std: {returns.std():.6f}")
    logger.info(f"Returns min: {returns.min():.6f}, max: {returns.max():.6f}")
    logger.info(f"First 5 returns: {returns.head().tolist()}")
    logger.info(f"Returns dtype: {returns.dtype}")
    
    try:
        # Basic statistics
        mean_return = returns.mean()
        std_return = returns.std()
        
        # ADD DEBUG HERE TOO
        logger.info(f"Annualizing: mean_return={mean_return:.6f}, std_return={std_return:.6f}")
        
        annualized_return = mean_return * 252
        annualized_vol = std_return * np.sqrt(252)
        
        logger.info(f"Annualized: return={annualized_return:.6f}, vol={annualized_vol:.6f}")
        
        
        # Risk-adjusted ratios
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Downside metrics
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_std if downside_std > 0 else 0
        
        # Distribution metrics
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # VaR and CVaR
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else var_99
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'annualized_volatility': annualized_vol,
            'annualized_return': annualized_return,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
        
    except Exception as e:
        logger.error(f"Risk metrics calculation failed: {e}")
        return {}

def _generate_synthetic_portfolio_returns(
    symbols: List[str],
    period: str = "1year",
    weights: Optional[Dict[str, float]] = None,
    seed: int = 42
) -> pd.Series:
    """Generate synthetic portfolio returns for fallback"""
    np.random.seed(seed)
    
    # Period mapping
    period_mapping = {
        "1month": 21,
        "3months": 63,
        "6months": 126,
        "1year": 252,
        "2years": 504,
        "5years": 1260
    }
    
    n_days = period_mapping.get(period, 252)
    
    # Generate correlated returns for multiple assets
    n_assets = len(symbols)
    
    # Create correlation matrix with realistic correlations
    correlation_matrix = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            correlation_matrix[i, j] = correlation_matrix[j, i] = np.random.uniform(0.1, 0.7)
    
    # Generate multivariate normal returns
    mean_returns = np.random.uniform(0.0003, 0.001, n_assets)  # Daily returns
    volatilities = np.random.uniform(0.01, 0.03, n_assets)    # Daily volatilities
    
    # Create covariance matrix
    cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
    
    # Generate returns
    asset_returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
    
    # Create portfolio returns
    if weights:
        weight_vector = np.array([weights.get(symbol, 0) for symbol in symbols])
        if np.sum(weight_vector) == 0:  # If no weights match symbols
            weight_vector = np.ones(n_assets) / n_assets
        else:
            weight_vector = weight_vector / np.sum(weight_vector)  # Normalize
    else:
        weight_vector = np.ones(n_assets) / n_assets
    
    portfolio_returns = np.dot(asset_returns, weight_vector)
    
    # Create date index
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=n_days)
    date_index = pd.date_range(start=start_date, end=end_date, freq='D')[:n_days]
    
    return pd.Series(portfolio_returns, index=date_index)

def _apply_stress_scenario(returns: pd.Series, shock_magnitude: float) -> Dict[str, Any]:
    """Apply stress scenario to return series"""
    try:
        # Apply shock to the return series
        stressed_returns = returns.copy()
        stressed_returns.iloc[-1] = shock_magnitude
        
        # Calculate metrics
        total_return = (1 + stressed_returns).prod() - 1
        volatility = stressed_returns.std() * np.sqrt(252)
        max_daily_loss = stressed_returns.min()
        
        # Drawdown analysis
        cumulative = (1 + stressed_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'shock_magnitude_pct': shock_magnitude * 100,
            'total_return_pct': total_return * 100,
            'total_loss_pct': min(0, total_return * 100),
            'max_daily_loss_pct': max_daily_loss * 100,
            'annualized_volatility_pct': volatility * 100,
            'max_drawdown_pct': max_drawdown * 100,
            'negative_days': (stressed_returns < 0).sum(),
            'recovery_estimate_days': max(1, abs(shock_magnitude) * 252)
        }
        
    except Exception as e:
        return {'error': f'Stress scenario calculation failed: {str(e)}'}

def _run_monte_carlo_stress(returns: pd.Series, n_simulations: int = 1000) -> Dict[str, Any]:
    """Run Monte Carlo stress testing"""
    try:
        if len(returns) < 30:
            return {'error': 'Insufficient data for Monte Carlo simulation'}
        
        # Fit distribution parameters
        mu = returns.mean()
        sigma = returns.std()
        
        # Test for normality
        _, p_value = stats.jarque_bera(returns.dropna())
        use_normal = p_value > 0.05
        
        # Generate scenarios
        if use_normal:
            scenarios = np.random.normal(mu, sigma, (n_simulations, 30))  # 30-day scenarios
        else:
            # Use t-distribution for fat tails
            df = 6  # Conservative estimate
            scenarios = stats.t.rvs(df, loc=mu, scale=sigma, size=(n_simulations, 30))
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + scenarios, axis=1) - 1
        final_returns = cumulative_returns[:, -1]
        
        # Risk metrics
        var_95 = np.percentile(final_returns, 5)
        var_99 = np.percentile(final_returns, 1)
        expected_shortfall_95 = np.mean(final_returns[final_returns <= var_95])
        expected_shortfall_99 = np.mean(final_returns[final_returns <= var_99])
        
        # Probability estimates
        prob_loss_10 = np.mean(final_returns < -0.10) * 100
        prob_loss_20 = np.mean(final_returns < -0.20) * 100
        
        return {
            'simulations': n_simulations,
            'distribution_used': 'normal' if use_normal else 't-distribution',
            'var_95_30day': var_95,
            'var_99_30day': var_99,
            'expected_shortfall_95': expected_shortfall_95,
            'expected_shortfall_99': expected_shortfall_99,
            'probability_loss_10pct': prob_loss_10,
            'probability_loss_20pct': prob_loss_20,
            'worst_case_30day': np.min(final_returns),
            'best_case_30day': np.max(final_returns),
            'expected_return_30day': np.mean(final_returns)
        }
        
    except Exception as e:
        return {'error': f'Monte Carlo simulation failed: {str(e)}'}

# ============================================================================
# Testing Functions
# ============================================================================

async def test_risk_tools_integration():
    """Test FMP integration for risk tools"""
    print("=" * 50)
    print("RISK TOOLS INTEGRATION TEST")
    print("=" * 50)
    
    test_symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    try:
        # Test 1: Comprehensive Risk Analysis
        print("\n1. Testing Comprehensive Risk Analysis...")
        risk_result = await calculate_comprehensive_risk(
            symbols=test_symbols,
            period='6months',
            use_real_data=True
        )
        
        if risk_result.success:
            print(f"✅ Risk analysis successful with {risk_result.data_source}")
            print(f"   Sharpe Ratio: {risk_result.sharpe_ratio:.3f}")
            print(f"   Max Drawdown: {risk_result.max_drawdown_pct:.2f}%")
        else:
            print(f"❌ Risk analysis failed: {risk_result.error}")
        
        # Test 2: VaR/CVaR Estimation
        print("\n2. Testing VaR/CVaR Estimation...")
        var_result = await estimate_var_cvar(
            symbols=test_symbols,
            use_real_data=True
        )
        
        if var_result.get('success'):
            print(f"✅ VaR/CVaR estimation successful")
            print(f"   Data Source: {var_result['data_source']}")
            var_95 = var_result['var_cvar_estimates']['95%']['var_pct']
            print(f"   VaR 95%: {var_95:.2f}%" if var_95 else "   VaR 95%: N/A")
        else:
            print(f"❌ VaR/CVaR estimation failed: {var_result.get('error')}")
        
        # Test 3: Stress Testing
        print("\n3. Testing Stress Testing...")
        stress_result = await perform_stress_testing(
            symbols=test_symbols,
            use_real_data=True,
            monte_carlo_simulations=100  # Reduced for faster testing
        )
        
        if stress_result.success:
            print(f"✅ Stress testing successful")
            print(f"   Data Source: {stress_result.data_source}")
            print(f"   Worst Case: {stress_result.worst_case_scenario}")
            print(f"   Resilience Score: {stress_result.resilience_score:.1f}")
        else:
            print(f"❌ Stress testing failed: {stress_result.error}")
        
        # Test 4: GARCH Volatility Forecasting
        print("\n4. Testing GARCH Volatility Forecasting...")
        garch_result = await forecast_volatility_garch(
            symbols=test_symbols,
            forecast_horizon=21,
            use_real_data=True
        )
        
        if garch_result.success:
            print(f"✅ GARCH forecasting successful")
            print(f"   Data Source: {garch_result.data_source}")
            print(f"   Current Vol: {garch_result.current_volatility:.4f}")
            print(f"   Forecast Vol: {garch_result.forecast_volatility:.4f}")
            print(f"   Trend: {garch_result.volatility_trend}")
        else:
            print(f"❌ GARCH forecasting failed: {garch_result.error}")
        
        print("\n" + "=" * 50)
        print("RISK TOOLS INTEGRATION TEST COMPLETE")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("Risk Tools Standalone Module")
    print("Testing FMP integration...")
    
    # Run integration test
    asyncio.run(test_risk_tools_integration())