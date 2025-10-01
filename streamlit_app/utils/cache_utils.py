"""
Cache utilities for Streamlit app
Handles caching and timestamp tracking for analysis results
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any

# Convert mutable types to hashable tuples
def to_hashable(symbols: List[str], weights: List[float] = None) -> Tuple:
    """Convert lists to tuples for caching"""
    if weights is None:
        return (tuple(symbols),)
    return (tuple(symbols), tuple(weights))


# ============================================================================
# CACHED API CALLS - Note: _api_client prefix tells Streamlit not to hash it
# ============================================================================

@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_risk_analysis(symbols: tuple, weights: tuple, period: str, _api_client):
    """Cached comprehensive risk analysis"""
    result = _api_client.analyze_risk(list(symbols), list(weights), period)
    return result


@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_optimization(symbols: tuple, method: str, period: str, _api_client):
    """Cached portfolio optimization"""
    result = _api_client.optimize_portfolio(list(symbols), method, period)
    return result


@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_correlation_analysis(symbols: tuple, period: str, _api_client):
    """Cached correlation analysis"""
    result = _api_client.correlation_analysis(list(symbols), period)
    return result


@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_comprehensive_correlation(symbols: tuple, period: str, _api_client):
    """Cached comprehensive correlation analysis"""
    result = _api_client.comprehensive_correlation(list(symbols), period)
    return result


@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_advanced_metrics(symbols: tuple, weights: tuple, period: str, _api_client):
    """Cached advanced portfolio metrics"""
    result = _api_client.advanced_analytics(list(symbols), list(weights), period)
    return result


@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_risk_attribution(symbols: tuple, weights: tuple, period: str, _api_client):
    """Cached risk attribution analysis"""
    result = _api_client.risk_attribution(list(symbols), list(weights), period)
    return result


@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_performance_attribution(symbols: tuple, weights: tuple, benchmark: str, period: str, _api_client):
    """Cached performance attribution analysis"""
    result = _api_client.performance_attribution(list(symbols), list(weights), benchmark, period)
    return result


@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_var_calculation(symbols: tuple, weights: tuple, confidence: float, _api_client):
    """Cached VaR calculation"""
    result = _api_client.calculate_var(list(symbols), list(weights), confidence)
    return result


@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_stress_test(symbols: tuple, weights: tuple, _api_client):
    """Cached stress test"""
    result = _api_client.stress_test(list(symbols), list(weights))
    return result


@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_rolling_correlations(symbols: tuple, window_size: int, period: str, _api_client):
    """Cached rolling correlations"""
    result = _api_client.rolling_correlations(list(symbols), window_size, period)
    return result


@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_regime_correlations(symbols: tuple, regime_method: str, period: str, _api_client):
    """Cached regime correlations"""
    result = _api_client.regime_correlations(list(symbols), regime_method, period)
    return result


@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_correlation_clustering(symbols: tuple, period: str, _api_client):
    """Cached correlation clustering"""
    result = _api_client.correlation_clustering(list(symbols), period)
    return result


@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_correlation_network(symbols: tuple, period: str, _api_client):
    """Cached correlation network analysis"""
    result = _api_client.correlation_network(list(symbols), period)
    return result


@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_forecast_returns(symbols: tuple, period: str, _api_client):
    """Cached return forecasting"""
    result = _api_client.forecast_returns(list(symbols), period)
    return result


@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_volatility_forecast(symbols: tuple, forecast_horizon: int, period: str, _api_client):
    """Cached volatility forecasting"""
    result = _api_client.forecast_volatility_garch(list(symbols), forecast_horizon, period)
    return result


# ============================================================================
# TIMESTAMP MANAGEMENT
# ============================================================================

def save_analysis_timestamp(analysis_type: str):
    """Save timestamp when analysis was last run"""
    if 'analysis_timestamps' not in st.session_state:
        st.session_state['analysis_timestamps'] = {}
    st.session_state['analysis_timestamps'][analysis_type] = datetime.now()


def get_analysis_timestamp(analysis_type: str) -> datetime:
    """Get timestamp of last analysis"""
    if 'analysis_timestamps' not in st.session_state:
        return None
    return st.session_state['analysis_timestamps'].get(analysis_type)


def format_time_ago(dt: datetime) -> str:
    """Format datetime as 'X ago' string"""
    if dt is None:
        return "Never"
    
    now = datetime.now()
    diff = now - dt
    
    if diff < timedelta(minutes=1):
        return "Just now"
    elif diff < timedelta(hours=1):
        mins = int(diff.total_seconds() / 60)
        return f"{mins} minute{'s' if mins != 1 else ''} ago"
    elif diff < timedelta(days=1):
        hours = int(diff.total_seconds() / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = diff.days
        return f"{days} day{'s' if days != 1 else ''} ago"


def show_last_updated_badge(analysis_type: str):
    """Show 'Last Updated: X ago' badge"""
    timestamp = get_analysis_timestamp(analysis_type)
    if timestamp:
        time_ago = format_time_ago(timestamp)
        st.caption(f"🕐 Last Updated: {time_ago}")


def show_staleness_warning(analysis_type: str, staleness_days: int = 3):
    """Show warning if analysis is stale"""
    timestamp = get_analysis_timestamp(analysis_type)
    if timestamp:
        age = datetime.now() - timestamp
        if age > timedelta(days=staleness_days):
            st.warning(
                f"⚠️ This analysis is {format_time_ago(timestamp)} old. "
                "Consider re-running for current data.",
                icon="⚠️"
            )