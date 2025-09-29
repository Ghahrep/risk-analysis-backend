# tools/regime_tools_standalone.py
"""
Regime Analysis Tools - Enhanced with FMP Integration
===================================================

Standalone regime analysis tools with Financial Modeling Prep (FMP) integration.
Following the successful pattern from risk, behavioral, portfolio, and forecasting tools.

Key Features:
- Hidden Markov Model (HMM) regime detection
- Volatility-based regime identification
- Transition probability analysis
- FMP real market data integration with automatic fallbacks
- Comprehensive error handling and logging
- Type-safe dataclass responses
- No circular import dependencies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import warnings

# Suppress common warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

# Import data providers following your successful pattern
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "portfolio_data_manager", 
        "data/providers/fmp_integration.py"
    )
    portfolio_data_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(portfolio_data_module)
    PortfolioDataManager = portfolio_data_module.PortfolioDataManager
    HAS_FMP_INTEGRATION = True
    logger.info("FMP integration available for regime tools")
except Exception as e:
    logger.warning(f"FMP integration not available: {e}")
    HAS_FMP_INTEGRATION = False

# Try to import regime analysis libraries
try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMMLEARN = True
except ImportError:
    logger.warning("hmmlearn not available - using simplified regime detection")
    HAS_HMMLEARN = False

try:
    import scipy.stats as stats
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    logger.warning("Scipy not available - using numpy for statistics")
    HAS_SCIPY = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    logger.warning("Scikit-learn not available - using basic clustering")
    HAS_SKLEARN = False

# Response dataclasses following your successful pattern
@dataclass
class RegimeAnalysisResult:
    """Standardized regime analysis result structure"""
    success: bool
    regime_series: Optional[pd.Series] = None
    regime_stats: Optional[Dict[str, Any]] = None
    transition_matrix: Optional[np.ndarray] = None
    regime_probabilities: Optional[Dict[str, float]] = None
    n_regimes: Optional[int] = None
    method: Optional[str] = None
    data_source: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class HMMRegimeResult:
    """HMM-specific regime analysis result"""
    success: bool
    regime_series: Optional[pd.Series] = None
    model_params: Optional[Dict[str, Any]] = None
    transition_matrix: Optional[np.ndarray] = None
    emission_params: Optional[Dict[str, Any]] = None
    log_likelihood: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    n_regimes: Optional[int] = None
    convergence_info: Optional[Dict[str, Any]] = None
    data_source: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class VolatilityRegimeResult:
    """Volatility regime analysis result"""
    success: bool
    regime_series: Optional[pd.Series] = None
    volatility_series: Optional[pd.Series] = None
    regime_thresholds: Optional[Dict[str, float]] = None
    regime_characteristics: Optional[Dict[str, Dict[str, float]]] = None
    transition_analysis: Optional[Dict[str, Any]] = None
    persistence_metrics: Optional[Dict[str, float]] = None
    data_source: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class ComprehensiveRegimeResult:
    """Comprehensive regime analysis combining multiple methods"""
    success: bool
    hmm_analysis: Optional[Dict[str, Any]] = None
    volatility_analysis: Optional[Dict[str, Any]] = None
    transition_analysis: Optional[Dict[str, Any]] = None
    regime_consensus: Optional[Dict[str, Any]] = None
    cross_method_validation: Optional[Dict[str, Any]] = None
    unified_regime_series: Optional[pd.Series] = None
    data_source: Optional[str] = None
    analysis_quality: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

# Data provider helper functions (reuse from forecasting tools)
def get_regime_returns_data(
    symbols: Optional[List[str]] = None,
    use_real_data: bool = True,
    period: str = "2years"  # Longer default for regime analysis
) -> Tuple[Optional[pd.Series], str]:
    """
    Get returns data for regime analysis with FMP integration
    Returns: (returns_series, data_source)
    """
    try:
        if not use_real_data or not HAS_FMP_INTEGRATION:
            return _generate_synthetic_regime_returns(symbols), "Synthetic Data Provider"
        
        if not symbols:
            symbols = ['AAPL', 'GOOGL', 'MSFT']  # Default portfolio
        
        # Use FMP integration following your successful pattern
        data_manager = PortfolioDataManager()
        portfolio_data = data_manager.get_returns_data(symbols, period)
        
        if portfolio_data is None:
            logger.warning("FMP data unavailable, using synthetic data")
            return _generate_synthetic_regime_returns(symbols), "FMP Fallback - Synthetic Data"
        
        # Extract returns series from portfolio data
        if isinstance(portfolio_data, dict) and 'returns' in portfolio_data:
            returns = portfolio_data['returns']
            if isinstance(returns, pd.Series):
                return returns, "Financial Modeling Prep (FMP)"
            elif isinstance(returns, pd.DataFrame):
                # Use first column or average if multiple assets
                if len(returns.columns) == 1:
                    return returns.iloc[:, 0], "Financial Modeling Prep (FMP)"
                else:
                    # Equal-weighted portfolio returns
                    portfolio_returns = returns.mean(axis=1)
                    portfolio_returns.name = 'portfolio_returns'
                    return portfolio_returns, "Financial Modeling Prep (FMP) - Portfolio"
        
        # Fallback to synthetic if FMP data format unexpected
        logger.warning("Unexpected FMP data format, using synthetic fallback")
        return _generate_synthetic_regime_returns(symbols), "FMP Format Fallback - Synthetic Data"
        
    except Exception as e:
        logger.error(f"Error getting regime returns data: {e}")
        return _generate_synthetic_regime_returns(symbols), f"Error Fallback - Synthetic Data: {str(e)}"

def _generate_synthetic_regime_returns(symbols: Optional[List[str]] = None) -> pd.Series:
    """Generate synthetic returns with regime characteristics for testing/fallback"""
    try:
        n_days = 500  # ~2 years of trading days
        np.random.seed(42)  # Reproducible for testing
        
        # Generate multi-regime market returns with clear regime switches
        returns = []
        regime_labels = []
        
        # Define regime characteristics
        regimes = {
            'low_vol': {'mean': 0.0005, 'vol': 0.008, 'prob_stay': 0.95},
            'high_vol': {'mean': -0.001, 'vol': 0.025, 'prob_stay': 0.88},
            'crisis': {'mean': -0.005, 'vol': 0.045, 'prob_stay': 0.80}
        }
        
        current_regime = 'low_vol'
        
        for i in range(n_days):
            # Regime switching logic
            if i > 0:  # Don't switch on first day
                if current_regime == 'low_vol':
                    if np.random.random() > regimes[current_regime]['prob_stay']:
                        current_regime = np.random.choice(['high_vol'], p=[1.0])
                elif current_regime == 'high_vol':
                    if np.random.random() > regimes[current_regime]['prob_stay']:
                        current_regime = np.random.choice(['low_vol', 'crisis'], p=[0.7, 0.3])
                elif current_regime == 'crisis':
                    if np.random.random() > regimes[current_regime]['prob_stay']:
                        current_regime = np.random.choice(['high_vol', 'low_vol'], p=[0.8, 0.2])
            
            # Generate return for current regime
            regime_params = regimes[current_regime]
            daily_return = np.random.normal(regime_params['mean'], regime_params['vol'])
            
            returns.append(daily_return)
            regime_labels.append(current_regime)
        
        # Create date index
        end_date = datetime.now()
        date_range = pd.bdate_range(end=end_date, periods=n_days, freq='D')
        
        returns_series = pd.Series(returns, index=date_range, name='synthetic_regime_returns')
        
        # Store regime labels for validation (in real scenario, these would be unknown)
        returns_series.true_regimes = pd.Series(regime_labels, index=date_range)
        
        return returns_series
        
    except Exception as e:
        logger.error(f"Error generating synthetic regime returns: {e}")
        # Ultra-simple fallback
        simple_returns = np.random.normal(0.001, 0.02, 200)
        return pd.Series(simple_returns, name='fallback_regime_returns')

# Core regime analysis functions
def detect_hmm_regimes(
    returns: Optional[pd.Series] = None,
    symbols: Optional[List[str]] = None,
    use_real_data: bool = True,
    period: str = "2years",
    n_regimes: int = 2,
    max_iter: int = 1000,
    random_state: int = 42,
    covariance_type: str = "full"
) -> Dict[str, Any]:
    """
    Enhanced HMM regime detection with FMP integration
    """
    try:
        # Get returns data
        if returns is None:
            returns, data_source = get_regime_returns_data(symbols, use_real_data, period)
        else:
            data_source = "Provided Returns Data"
        
        if returns is None or len(returns) < 100:
            return {
                'success': False,
                'error': 'Insufficient returns data for HMM regime analysis (need 100+ observations)',
                'data_source': data_source
            }
        
        returns_clean = returns.dropna()
        logger.info(f"HMM regime detection with {len(returns_clean)} observations from {data_source}")
        
        # Prepare data for HMM (reshape for multivariate input)
        returns_array = returns_clean.values.reshape(-1, 1)
        
        if HAS_HMMLEARN:
            hmm_result = _fit_hmm_model(returns_array, n_regimes, max_iter, random_state, covariance_type)
        else:
            # Simplified regime detection without hmmlearn
            hmm_result = _fit_simple_hmm(returns_clean, n_regimes)
        
        if not hmm_result['success']:
            return {
                'success': False,
                'error': hmm_result.get('error', 'HMM fitting failed'),
                'data_source': data_source
            }
        
        # Create regime series with proper index
        regime_series = pd.Series(
            hmm_result['regime_sequence'], 
            index=returns_clean.index, 
            name='hmm_regimes'
        )
        
        # Calculate regime statistics
        regime_stats = _calculate_regime_statistics(returns_clean, regime_series, n_regimes)
        
        # Calculate transition matrix
        transition_matrix = _calculate_transition_matrix(regime_series, n_regimes)
        
        # Calculate performance metrics
        performance_metrics = _calculate_hmm_performance_metrics(
            returns_clean, regime_series, hmm_result
        )
        
        result = HMMRegimeResult(
            success=True,
            regime_series=regime_series,
            model_params=hmm_result.get('model_params', {}),
            transition_matrix=transition_matrix,
            emission_params=hmm_result.get('emission_params', {}),
            log_likelihood=hmm_result.get('log_likelihood'),
            aic=hmm_result.get('aic'),
            bic=hmm_result.get('bic'),
            n_regimes=n_regimes,
            convergence_info=hmm_result.get('convergence_info', {}),
            data_source=data_source
        )
        
        return asdict(result)
        
    except Exception as e:
        logger.error(f"HMM regime detection failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'data_source': locals().get('data_source', 'Unknown')
        }

def detect_volatility_regimes(
    returns: Optional[pd.Series] = None,
    symbols: Optional[List[str]] = None,
    use_real_data: bool = True,
    period: str = "2years",
    window: int = 30,
    threshold_low: float = 0.15,
    threshold_high: float = 0.25
) -> Dict[str, Any]:
    """
    Enhanced volatility-based regime detection with FMP integration
    """
    try:
        # Get returns data
        if returns is None:
            returns, data_source = get_regime_returns_data(symbols, use_real_data, period)
        else:
            data_source = "Provided Returns Data"
        
        if returns is None or len(returns) < window + 50:
            return {
                'success': False,
                'error': f'Insufficient returns data for volatility regime analysis (need {window + 50}+ observations)',
                'data_source': data_source
            }
        
        returns_clean = returns.dropna()
        logger.info(f"Volatility regime detection with {len(returns_clean)} observations from {data_source}")
        
        # Calculate rolling volatility
        rolling_vol = returns_clean.rolling(window=window).std()
        vol_series = rolling_vol.dropna()
        
        if len(vol_series) < 50:
            return {
                'success': False,
                'error': 'Insufficient volatility data after rolling calculation',
                'data_source': data_source
            }
        
        # Annualize volatility for more intuitive thresholds
        vol_annualized = vol_series * np.sqrt(252)
        
        # Create regime series with same index as vol_annualized
        regime_series = pd.Series(0, index=vol_annualized.index, name='volatility_regimes')
        
        # Assign regimes safely
        for i in range(len(vol_annualized)):
            vol_val = vol_annualized.iloc[i]
            if vol_val <= threshold_low:
                regime_series.iloc[i] = 0
            elif vol_val <= threshold_high:
                regime_series.iloc[i] = 1
            else:
                regime_series.iloc[i] = 2
        
        # Calculate regime characteristics using aligned indices
        regime_characteristics = {}
        for regime in range(3):
            regime_mask = regime_series == regime
            if regime_mask.sum() > 0:
                # Use returns that align with volatility index
                aligned_returns = returns_clean.loc[vol_annualized.index]  # KEY FIX
                regime_returns = aligned_returns[regime_mask]
                regime_vol = vol_annualized[regime_mask]
                
                regime_characteristics[f'regime_{regime}'] = {
                    'mean_return': float(regime_returns.mean()) if len(regime_returns) > 0 else 0.0,
                    'volatility': float(regime_vol.mean()) if len(regime_vol) > 0 else 0.0,
                    'return_std': float(regime_returns.std()) if len(regime_returns) > 0 else 0.0,
                    'skewness': float(_calculate_skewness(regime_returns)) if len(regime_returns) > 0 else 0.0,
                    'kurtosis': float(_calculate_kurtosis(regime_returns)) if len(regime_returns) > 0 else 0.0,
                    'observations': int(regime_mask.sum()),
                    'proportion': float(regime_mask.mean()),
                    'min_volatility': float(regime_vol.min()) if len(regime_vol) > 0 else 0.0,
                    'max_volatility': float(regime_vol.max()) if len(regime_vol) > 0 else 0.0
                }
        
        # Calculate transition analysis and persistence metrics
        transition_analysis = _analyze_volatility_regime_transitions(regime_series, vol_annualized)
        persistence_metrics = _calculate_regime_persistence(regime_series)
        
        result = VolatilityRegimeResult(
            success=True,
            regime_series=regime_series,
            volatility_series=vol_annualized,
            regime_thresholds={
                'low_threshold': threshold_low,
                'high_threshold': threshold_high,
                'actual_low_mean': regime_characteristics.get('regime_0', {}).get('volatility', 0),
                'actual_high_mean': regime_characteristics.get('regime_2', {}).get('volatility', 0)
            },
            regime_characteristics=regime_characteristics,
            transition_analysis=transition_analysis,
            persistence_metrics=persistence_metrics,
            data_source=data_source
        )
        
        return asdict(result)
        
    except Exception as e:
        logger.error(f"Volatility regime detection failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'data_source': locals().get('data_source', 'Unknown')
        }

def comprehensive_regime_analysis(
    returns: Optional[pd.Series] = None,
    symbols: Optional[List[str]] = None,
    use_real_data: bool = True,
    period: str = "2years",
    include_hmm: bool = True,
    include_volatility: bool = True,
    include_transitions: bool = True,
    include_returns_analysis: bool = True,
    include_shift_detection: bool = False,
    hmm_config: Optional[Dict[str, Any]] = None,
    volatility_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Enhanced comprehensive regime analysis combining multiple methodologies
    """
    try:
        # Get returns data once for all analyses
        if returns is None:
            returns, data_source = get_regime_returns_data(symbols, use_real_data, period)
        else:
            data_source = "Provided Returns Data"
        
        if returns is None or len(returns) < 200:
            return {
                'success': False,
                'error': 'Insufficient returns data for comprehensive regime analysis (need 200+ observations)',
                'data_source': data_source
            }
        
        logger.info(f"Comprehensive regime analysis with data from {data_source}")
        
        comprehensive_results = {
            'success': True,
            'data_source': data_source,
            'analysis_components': {},
            'cross_validation': {},
            'unified_insights': {}
        }
        
        # 1. HMM Analysis
        if include_hmm:
            hmm_config = hmm_config or {}
            hmm_result = detect_hmm_regimes(
                returns=returns,
                use_real_data=False,  # Already have data
                n_regimes=hmm_config.get('n_regimes', 2),
                max_iter=hmm_config.get('max_iter', 1000),
                random_state=hmm_config.get('random_state', 42),
                covariance_type=hmm_config.get('covariance_type', 'full')
            )
            comprehensive_results['analysis_components']['hmm_analysis'] = hmm_result
        
        # 2. Volatility-based Analysis
        if include_volatility:
            volatility_config = volatility_config or {}
            vol_result = detect_volatility_regimes(
                returns=returns,
                use_real_data=False,  # Already have data
                window=volatility_config.get('window', 30),
                threshold_low=volatility_config.get('threshold_low', 0.15),
                threshold_high=volatility_config.get('threshold_high', 0.25)
            )
            comprehensive_results['analysis_components']['volatility_analysis'] = vol_result
        
        # 3. Transition Analysis
        if include_transitions:
            transition_results = {}
            
            # HMM transitions
            if include_hmm and comprehensive_results['analysis_components'].get('hmm_analysis', {}).get('success'):
                hmm_regimes = comprehensive_results['analysis_components']['hmm_analysis'].get('regime_series')
                if hmm_regimes is not None:
                    # Convert list back to pandas Series for transition analysis
                    hmm_series = pd.Series(hmm_regimes, name='hmm_regimes')
                    hmm_transitions = _analyze_regime_transition_dynamics(hmm_series, 'HMM')
                    transition_results['hmm_transitions'] = hmm_transitions
            
            # Volatility transitions
            if include_volatility and comprehensive_results['analysis_components'].get('volatility_analysis', {}).get('success'):
                vol_regimes = comprehensive_results['analysis_components']['volatility_analysis'].get('regime_series')
                if vol_regimes is not None:
                    # Convert list back to pandas Series for transition analysis
                    vol_series = pd.Series(vol_regimes, name='volatility_regimes')
                    vol_transitions = _analyze_regime_transition_dynamics(vol_series, 'Volatility')
                    transition_results['volatility_transitions'] = vol_transitions
            
            comprehensive_results['analysis_components']['transition_analysis'] = transition_results
        
        # 4. Returns Analysis by Regime
        if include_returns_analysis:
            returns_analysis = _analyze_regime_conditional_returns(
                returns, comprehensive_results['analysis_components']
            )
            comprehensive_results['analysis_components']['returns_analysis'] = returns_analysis
        
        # 5. Cross-method validation
        cross_validation = _validate_regime_methods(
            comprehensive_results['analysis_components']
        )
        comprehensive_results['cross_validation'] = cross_validation
        
        # 6. Unified regime consensus (if multiple methods successful)
        if len([comp for comp in comprehensive_results['analysis_components'].values() 
                if isinstance(comp, dict) and comp.get('success', False)]) >= 2:
            unified_regime = _create_unified_regime_series(
                comprehensive_results['analysis_components']
            )
            comprehensive_results['unified_insights']['unified_regime'] = unified_regime
        
        # 7. Analysis quality assessment
        quality_assessment = _assess_regime_analysis_quality(
            comprehensive_results, returns
        )
        comprehensive_results['analysis_quality'] = quality_assessment
        
        result = ComprehensiveRegimeResult(
            success=True,
            hmm_analysis=comprehensive_results['analysis_components'].get('hmm_analysis'),
            volatility_analysis=comprehensive_results['analysis_components'].get('volatility_analysis'),
            transition_analysis=comprehensive_results['analysis_components'].get('transition_analysis'),
            regime_consensus=comprehensive_results.get('cross_validation', {}),
            cross_method_validation=comprehensive_results.get('cross_validation', {}),
            unified_regime_series=comprehensive_results.get('unified_insights', {}).get('unified_regime'),
            data_source=data_source,
            analysis_quality=comprehensive_results.get('analysis_quality', {})
        )
        
        return asdict(result)
        
    except Exception as e:
        logger.error(f"Comprehensive regime analysis failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'data_source': locals().get('data_source', 'Unknown')
        }

# Helper functions for HMM analysis
def _fit_hmm_model(returns_array: np.ndarray, n_regimes: int, max_iter: int, random_state: int, covariance_type: str) -> Dict[str, Any]:
    """Fit HMM model using hmmlearn"""
    try:
        # Initialize HMM model
        model = GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_iter=max_iter,
            random_state=random_state,
            verbose=False
        )
        
        # Fit the model
        model.fit(returns_array)
        
        # Get regime sequence
        regime_sequence = model.predict(returns_array)
        
        # Calculate model metrics
        log_likelihood = model.score(returns_array)
        n_params = model._get_n_fit_scalars_per_param()
        n_observations = len(returns_array)
        
        aic = -2 * log_likelihood + 2 * sum(n_params.values())
        bic = -2 * log_likelihood + np.log(n_observations) * sum(n_params.values())
        
        # Extract model parameters
        model_params = {
            'transition_matrix': model.transmat_.tolist(),
            'start_probabilities': model.startprob_.tolist(),
            'n_iter': model.monitor_.iter,
            'converged': model.monitor_.converged
        }
        
        emission_params = {
            'means': model.means_.flatten().tolist(),
            'covariances': model.covars_.flatten().tolist() if hasattr(model.covars_, 'flatten') else model.covars_.tolist()
        }
        
        convergence_info = {
            'converged': model.monitor_.converged,
            'n_iterations': model.monitor_.iter,
            'final_log_likelihood': log_likelihood
        }
        
        return {
            'success': True,
            'regime_sequence': regime_sequence.tolist(),
            'model_params': model_params,
            'emission_params': emission_params,
            'log_likelihood': float(log_likelihood),
            'aic': float(aic),
            'bic': float(bic),
            'convergence_info': convergence_info
        }
        
    except Exception as e:
        logger.error(f"HMM model fitting failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def _fit_simple_hmm(returns: pd.Series, n_regimes: int) -> Dict[str, Any]:
    """Simple regime detection fallback when hmmlearn is not available"""
    try:
        # Use rolling volatility for regime detection
        rolling_vol = returns.rolling(window=30).std()
        vol_values = rolling_vol.dropna()
        
        if n_regimes == 2:
            # Simple two-regime model
            vol_median = vol_values.median()
            regime_sequence = (rolling_vol > vol_median).astype(int)
        else:
            # Multi-regime using quantiles
            quantiles = np.linspace(0, 1, n_regimes + 1)[1:-1]
            thresholds = vol_values.quantile(quantiles).values
            
            regime_sequence = pd.Series(0, index=rolling_vol.index)
            for i, threshold in enumerate(thresholds):
                regime_sequence[rolling_vol > threshold] = i + 1
        
        regime_sequence = regime_sequence.fillna(method='ffill').fillna(0)
        
        return {
            'success': True,
            'regime_sequence': regime_sequence.values.tolist(),
            'model_params': {'method': 'simple_volatility_based'},
            'emission_params': {'note': 'Simplified regime detection'},
            'log_likelihood': None,
            'aic': None,
            'bic': None,
            'convergence_info': {'method': 'deterministic'}
        }
        
    except Exception as e:
        logger.error(f"Simple HMM fitting failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def _calculate_regime_statistics(returns: pd.Series, regime_series: pd.Series, n_regimes: int) -> Dict[str, Any]:
    """Calculate statistics for each regime"""
    regime_stats = {}
    
    try:
        for regime in range(n_regimes):
            regime_mask = regime_series == regime
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) > 0:
                regime_stats[f'regime_{regime}'] = {
                    'mean_return': float(regime_returns.mean()),
                    'volatility': float(regime_returns.std()),
                    'skewness': float(_calculate_skewness(regime_returns)),
                    'kurtosis': float(_calculate_kurtosis(regime_returns)),
                    'min_return': float(regime_returns.min()),
                    'max_return': float(regime_returns.max()),
                    'observations': int(regime_mask.sum()),
                    'proportion': float(regime_mask.mean()),
                    'annualized_return': float(regime_returns.mean() * 252),
                    'annualized_volatility': float(regime_returns.std() * np.sqrt(252))
                }
            else:
                regime_stats[f'regime_{regime}'] = {
                    'observations': 0,
                    'proportion': 0.0,
                    'note': 'No observations in this regime'
                }
        
        return regime_stats
        
    except Exception as e:
        logger.error(f"Regime statistics calculation failed: {e}")
        return {}

def _calculate_transition_matrix(regime_series: pd.Series, n_regimes: int) -> np.ndarray:
    """Calculate transition probability matrix"""
    try:
        regime_values = regime_series.dropna()
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        # Count transitions
        for i in range(len(regime_values) - 1):
            current_regime = int(regime_values.iloc[i])
            next_regime = int(regime_values.iloc[i + 1])
            
            if 0 <= current_regime < n_regimes and 0 <= next_regime < n_regimes:
                transition_matrix[current_regime, next_regime] += 1
        
        # Normalize to probabilities
        for i in range(n_regimes):
            row_sum = transition_matrix[i, :].sum()
            if row_sum > 0:
                transition_matrix[i, :] /= row_sum
        
        return transition_matrix
        
    except Exception as e:
        logger.error(f"Transition matrix calculation failed: {e}")
        return np.zeros((n_regimes, n_regimes))

def _calculate_hmm_performance_metrics(returns: pd.Series, regime_series: pd.Series, hmm_result: Dict) -> Dict[str, Any]:
    """Calculate performance metrics for HMM model"""
    try:
        metrics = {
            'data_observations': len(returns),
            'regime_observations': len(regime_series),
            'unique_regimes': len(regime_series.unique()),
            'model_type': 'HMM'
        }
        
        if hmm_result.get('log_likelihood') is not None:
            metrics['log_likelihood'] = hmm_result['log_likelihood']
        
        if hmm_result.get('aic') is not None:
            metrics['aic'] = hmm_result['aic']
            
        if hmm_result.get('bic') is not None:
            metrics['bic'] = hmm_result['bic']
        
        # Calculate regime stability
        regime_changes = (regime_series != regime_series.shift(1)).sum()
        metrics['regime_changes'] = int(regime_changes)
        metrics['regime_stability'] = float(1 - regime_changes / len(regime_series))
        
        return metrics
        
    except Exception as e:
        logger.error(f"HMM performance metrics calculation failed: {e}")
        return {
            'error': str(e),
            'data_observations': len(returns) if returns is not None else 0
        }

# Additional helper functions
def _calculate_skewness(returns: pd.Series) -> float:
    """Calculate skewness with fallback"""
    try:
        if HAS_SCIPY:
            return stats.skew(returns)
        else:
            # Manual calculation
            mean_ret = returns.mean()
            std_ret = returns.std()
            if std_ret == 0:
                return 0.0
            return ((returns - mean_ret) ** 3).mean() / (std_ret ** 3)
    except:
        return 0.0

def _calculate_kurtosis(returns: pd.Series) -> float:
    """Calculate kurtosis with fallback"""
    try:
        if HAS_SCIPY:
            return stats.kurtosis(returns)
        else:
            # Manual calculation
            mean_ret = returns.mean()
            std_ret = returns.std()
            if std_ret == 0:
                return 0.0
            return ((returns - mean_ret) ** 4).mean() / (std_ret ** 4) - 3
    except:
        return 0.0

def _analyze_volatility_regime_transitions(regime_series: pd.Series, volatility_series: pd.Series) -> Dict[str, Any]:
    """Analyze transitions between volatility regimes"""
    try:
        # Calculate transition frequencies
        transitions = {}
        regime_values = regime_series.dropna()
        
        for i in range(len(regime_values) - 1):
            current = int(regime_values.iloc[i])
            next_regime = int(regime_values.iloc[i + 1])
            transition_key = f'{current}_to_{next_regime}'
            
            if transition_key not in transitions:
                transitions[transition_key] = 0
            transitions[transition_key] += 1
        
        # Calculate transition probabilities
        total_transitions = sum(transitions.values())
        transition_probs = {
            key: count / total_transitions for key, count in transitions.items()
        } if total_transitions > 0 else {}
        
        # Calculate average duration in each regime
        regime_durations = {}
        current_regime = regime_values.iloc[0]
        current_duration = 1
        
        for i in range(1, len(regime_values)):
            if regime_values.iloc[i] == current_regime:
                current_duration += 1
            else:
                if current_regime not in regime_durations:
                    regime_durations[current_regime] = []
                regime_durations[current_regime].append(current_duration)
                current_regime = regime_values.iloc[i]
                current_duration = 1
        
        # Add final duration
        if current_regime not in regime_durations:
            regime_durations[current_regime] = []
        regime_durations[current_regime].append(current_duration)
        
        # Calculate average durations
        avg_durations = {
            f'regime_{regime}': float(np.mean(durations)) 
            for regime, durations in regime_durations.items()
        }
        
        return {
            'transition_counts': transitions,
            'transition_probabilities': transition_probs,
            'average_durations': avg_durations,
            'total_transitions': total_transitions
        }
        
    except Exception as e:
        logger.error(f"Volatility regime transition analysis failed: {e}")
        return {}

def _calculate_regime_persistence(regime_series: pd.Series) -> Dict[str, float]:
    """Calculate persistence metrics for regimes"""
    try:
        persistence_metrics = {}
        
        # Overall persistence
        no_change = (regime_series == regime_series.shift(1)).sum()
        overall_persistence = float(no_change / (len(regime_series) - 1))
        persistence_metrics['overall_persistence'] = overall_persistence
        
        # Persistence by regime
        for regime in regime_series.unique():
            if pd.isna(regime):
                continue
                
            regime_mask = regime_series == regime
            regime_indices = regime_series[regime_mask].index
            
            # Calculate persistence for this regime
            if len(regime_indices) > 1:
                consecutive_count = 0
                total_periods = 0
                
                for i in range(len(regime_indices) - 1):
                    current_idx = regime_indices[i]
                    next_idx = regime_indices[i + 1]
                    
                    # Check if consecutive
                    current_pos = regime_series.index.get_loc(current_idx)
                    next_pos = regime_series.index.get_loc(next_idx)
                    
                    if next_pos == current_pos + 1:
                        consecutive_count += 1
                    total_periods += 1
                
                regime_persistence = consecutive_count / total_periods if total_periods > 0 else 0
                persistence_metrics[f'regime_{int(regime)}_persistence'] = float(regime_persistence)
        
        return persistence_metrics
        
    except Exception as e:
        logger.error(f"Regime persistence calculation failed: {e}")
        return {}

def _analyze_regime_transition_dynamics(regime_series: pd.Series, method_name: str) -> Dict[str, Any]:
    """Analyze transition dynamics for any regime series"""
    try:
        transition_dynamics = {
            'method': method_name,
            'total_regimes': len(regime_series.unique()),
            'total_observations': len(regime_series),
            'transitions': {}
        }
        
        # Calculate transition matrix
        n_regimes = len(regime_series.unique())
        transition_matrix = _calculate_transition_matrix(regime_series, n_regimes)
        transition_dynamics['transition_matrix'] = transition_matrix.tolist()
        
        # Calculate stability metrics
        regime_changes = (regime_series != regime_series.shift(1)).sum()
        stability = 1 - (regime_changes / len(regime_series))
        transition_dynamics['stability'] = float(stability)
        transition_dynamics['total_changes'] = int(regime_changes)
        
        # Calculate expected duration in each regime
        expected_durations = {}
        for i in range(n_regimes):
            persistence_prob = transition_matrix[i, i]
            if persistence_prob < 1.0:
                expected_duration = 1 / (1 - persistence_prob)
                expected_durations[f'regime_{i}'] = float(expected_duration)
            else:
                expected_durations[f'regime_{i}'] = float('inf')
        
        transition_dynamics['expected_durations'] = expected_durations
        
        return transition_dynamics
        
    except Exception as e:
        logger.error(f"Regime transition dynamics analysis failed: {e}")
        return {'error': str(e)}

def _analyze_regime_conditional_returns(returns: pd.Series, analysis_components: Dict) -> Dict[str, Any]:
    """Analyze returns conditional on different regime identifications"""
    try:
        returns_analysis = {}
        
        # HMM conditional returns
        if 'hmm_analysis' in analysis_components:
            hmm_data = analysis_components['hmm_analysis']
            if hmm_data.get('success') and hmm_data.get('regime_series'):
                hmm_regimes = pd.Series(hmm_data['regime_series'])
                hmm_returns_stats = {}
                
                for regime in hmm_regimes.unique():
                    regime_mask = hmm_regimes == regime
                    if regime_mask.sum() > 0:
                        regime_returns = returns[regime_mask]
                        hmm_returns_stats[f'regime_{int(regime)}'] = {
                            'mean': float(regime_returns.mean()),
                            'std': float(regime_returns.std()),
                            'sharpe': float(regime_returns.mean() / regime_returns.std()) if regime_returns.std() > 0 else 0,
                            'observations': int(regime_mask.sum())
                        }
                
                returns_analysis['hmm_conditional'] = hmm_returns_stats
        
        # Volatility conditional returns
        if 'volatility_analysis' in analysis_components:
            vol_data = analysis_components['volatility_analysis']
            if vol_data.get('success') and vol_data.get('regime_series'):
                vol_regimes = pd.Series(vol_data['regime_series'])
                vol_returns_stats = {}
                
                for regime in vol_regimes.unique():
                    regime_mask = vol_regimes == regime
                    if regime_mask.sum() > 0:
                        regime_returns = returns[regime_mask]
                        vol_returns_stats[f'regime_{int(regime)}'] = {
                            'mean': float(regime_returns.mean()),
                            'std': float(regime_returns.std()),
                            'sharpe': float(regime_returns.mean() / regime_returns.std()) if regime_returns.std() > 0 else 0,
                            'observations': int(regime_mask.sum())
                        }
                
                returns_analysis['volatility_conditional'] = vol_returns_stats
        
        return returns_analysis
        
    except Exception as e:
        logger.error(f"Regime conditional returns analysis failed: {e}")
        return {}

def _validate_regime_methods(analysis_components: Dict) -> Dict[str, Any]:
    """Cross-validate different regime detection methods"""
    try:
        validation_results = {}
        
        # Check if both HMM and volatility analyses succeeded
        hmm_success = analysis_components.get('hmm_analysis', {}).get('success', False)
        vol_success = analysis_components.get('volatility_analysis', {}).get('success', False)
        
        if hmm_success and vol_success:
            hmm_regimes = pd.Series(analysis_components['hmm_analysis']['regime_series'])
            vol_regimes = pd.Series(analysis_components['volatility_analysis']['regime_series'])
            
            # Align series lengths
            min_length = min(len(hmm_regimes), len(vol_regimes))
            hmm_aligned = hmm_regimes[:min_length]
            vol_aligned = vol_regimes[:min_length]
            
            # Calculate agreement
            if len(hmm_aligned) > 0:
                # Simple agreement rate
                agreement = (hmm_aligned == vol_aligned).mean()
                validation_results['method_agreement'] = float(agreement)
                
                # Adjusted Rand Index if available
                if HAS_SKLEARN:
                    from sklearn.metrics import adjusted_rand_score
                    ari = adjusted_rand_score(hmm_aligned, vol_aligned)
                    validation_results['adjusted_rand_index'] = float(ari)
                
                validation_results['comparison_observations'] = len(hmm_aligned)
                validation_results['methods_compared'] = ['HMM', 'Volatility']
            else:
                validation_results['error'] = 'No observations for comparison'
        
        # Quality scores for individual methods
        individual_scores = {}
        
        if hmm_success:
            hmm_data = analysis_components['hmm_analysis']
            hmm_score = 0
            
            # Score based on convergence
            if hmm_data.get('convergence_info', {}).get('converged'):
                hmm_score += 3
            
            # Score based on likelihood
            if hmm_data.get('log_likelihood') is not None:
                hmm_score += 2
            
            # Score based on regime balance
            if hmm_data.get('regime_stats'):
                regime_proportions = [
                    stats.get('proportion', 0) 
                    for stats in hmm_data['regime_stats'].values()
                    if isinstance(stats, dict)
                ]
                if regime_proportions and max(regime_proportions) < 0.95:  # Not too imbalanced
                    hmm_score += 2
            
            individual_scores['hmm_quality_score'] = hmm_score
        
        if vol_success:
            vol_data = analysis_components['volatility_analysis']
            vol_score = 0
            
            # Score based on regime characteristics
            if vol_data.get('regime_characteristics'):
                vol_score += 3
            
            # Score based on persistence
            if vol_data.get('persistence_metrics'):
                persistence = vol_data['persistence_metrics'].get('overall_persistence', 0)
                if 0.7 <= persistence <= 0.95:  # Good persistence range
                    vol_score += 2
            
            # Score based on transition analysis
            if vol_data.get('transition_analysis'):
                vol_score += 2
            
            individual_scores['volatility_quality_score'] = vol_score
        
        validation_results['individual_quality'] = individual_scores
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Regime method validation failed: {e}")
        return {}

def _create_unified_regime_series(analysis_components: Dict) -> Optional[pd.Series]:
    """Create unified regime series from multiple methods"""
    try:
        # This is a simplified approach - in practice, you might use more sophisticated ensemble methods
        hmm_data = analysis_components.get('hmm_analysis', {})
        vol_data = analysis_components.get('volatility_analysis', {})
        
        if not (hmm_data.get('success') and vol_data.get('success')):
            return None
        
        hmm_regimes = pd.Series(hmm_data['regime_series'])
        vol_regimes = pd.Series(vol_data['regime_series'])
        
        # Align series
        min_length = min(len(hmm_regimes), len(vol_regimes))
        
        # Simple majority vote approach (could be enhanced)
        unified_regimes = []
        for i in range(min_length):
            hmm_val = hmm_regimes.iloc[i]
            vol_val = vol_regimes.iloc[i]
            
            # Simple rule: if both indicate high volatility regime, mark as 1, else 0
            # This is oversimplified but demonstrates the concept
            if (hmm_val == 1 and vol_val >= 1) or (hmm_val == 0 and vol_val == 2):
                unified_regimes.append(1)
            else:
                unified_regimes.append(0)
        
        return pd.Series(unified_regimes, name='unified_regimes')
        
    except Exception as e:
        logger.error(f"Unified regime series creation failed: {e}")
        return None

def _assess_regime_analysis_quality(comprehensive_results: Dict, returns: pd.Series) -> Dict[str, Any]:
    """Assess overall quality of regime analysis"""
    try:
        quality_assessment = {
            'overall_score': 0,
            'data_quality': 'Unknown',
            'method_reliability': 'Unknown',
            'regime_interpretability': 'Unknown',
            'components_successful': 0,
            'total_components': 0
        }
        
        # Assess data quality
        data_obs = len(returns)
        if data_obs >= 500:
            quality_assessment['data_quality'] = 'High'
            quality_assessment['overall_score'] += 3
        elif data_obs >= 200:
            quality_assessment['data_quality'] = 'Medium'
            quality_assessment['overall_score'] += 2
        else:
            quality_assessment['data_quality'] = 'Low'
            quality_assessment['overall_score'] += 1
        
        # Count successful components
        components = comprehensive_results.get('analysis_components', {})
        for component_name, component_data in components.items():
            quality_assessment['total_components'] += 1
            if isinstance(component_data, dict) and component_data.get('success'):
                quality_assessment['components_successful'] += 1
                quality_assessment['overall_score'] += 1
        
        # Assess method reliability
        if quality_assessment['total_components'] > 0:
            success_rate = quality_assessment['components_successful'] / quality_assessment['total_components']
            if success_rate >= 0.8:
                quality_assessment['method_reliability'] = 'High'
            elif success_rate >= 0.5:
                quality_assessment['method_reliability'] = 'Medium'
            else:
                quality_assessment['method_reliability'] = 'Low'
        
        # Assess interpretability based on cross-validation
        cross_val = comprehensive_results.get('cross_validation', {})
        if 'method_agreement' in cross_val:
            agreement = cross_val['method_agreement']
            if agreement >= 0.7:
                quality_assessment['regime_interpretability'] = 'High'
                quality_assessment['overall_score'] += 2
            elif agreement >= 0.5:
                quality_assessment['regime_interpretability'] = 'Medium'
                quality_assessment['overall_score'] += 1
            else:
                quality_assessment['regime_interpretability'] = 'Low'
        
        # Normalize overall score to 0-10 scale
        max_score = 3 + quality_assessment['total_components'] + 2  # Data + components + interpretability
        if max_score > 0:
            quality_assessment['overall_score'] = min(10, (quality_assessment['overall_score'] / max_score) * 10)
        
        return quality_assessment
        
    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        return {
            'overall_score': 0,
            'error': str(e)
        }

# Integration status function
def get_regime_tools_integration_status() -> Dict[str, Any]:
    """Get integration status for regime analysis tools"""
    try:
        status = {
            'fmp_integration': HAS_FMP_INTEGRATION,
            'statistical_libraries': {
                'hmmlearn': HAS_HMMLEARN,
                'sklearn': HAS_SKLEARN,
                'scipy': HAS_SCIPY
            },
            'available_methods': [],
            'data_providers': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Available methods
        if HAS_HMMLEARN:
            status['available_methods'].append('Hidden Markov Model (HMM)')
        status['available_methods'].extend(['Volatility-based Regimes', 'Simple Clustering'])
        
        # Data providers
        if HAS_FMP_INTEGRATION:
            status['data_providers'].append('Financial Modeling Prep (FMP)')
        status['data_providers'].append('Synthetic Data Generator')
        
        # Quick functionality test
        try:
            test_returns, test_source = get_regime_returns_data(
                symbols=['AAPL'], use_real_data=False, period='6months'
            )
            status['quick_test'] = {
                'success': test_returns is not None,
                'data_points': len(test_returns) if test_returns is not None else 0,
                'data_source': test_source
            }
        except Exception as e:
            status['quick_test'] = {
                'success': False,
                'error': str(e)
            }
        
        return status
        
    except Exception as e:
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }