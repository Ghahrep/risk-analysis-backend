"""
conftest.py - Shared test fixtures and configuration
Place this in your tests/ directory
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# SESSION-LEVEL FIXTURES
# ============================================================================

@pytest.fixture(scope='session')
def test_data_dir():
    """Directory for test data files"""
    return Path(__file__).parent / 'test_data'


# ============================================================================
# PORTFOLIO FIXTURES
# ============================================================================

@pytest.fixture
def sample_portfolio():
    """Standard 3-stock portfolio for testing"""
    return {
        'symbols': ['AAPL', 'MSFT', 'GOOGL'],
        'weights': [0.4, 0.3, 0.3]
    }

@pytest.fixture
def balanced_portfolio():
    """Balanced portfolio (60/40 stocks/bonds)"""
    return {
        'symbols': ['VTI', 'BND', 'VEA', 'GLD'],
        'weights': [0.40, 0.30, 0.20, 0.10]
    }

@pytest.fixture
def aggressive_portfolio():
    """Aggressive growth portfolio"""
    return {
        'symbols': ['QQQ', 'ARKK', 'VUG', 'TSLA'],
        'weights': [0.35, 0.25, 0.25, 0.15]
    }

@pytest.fixture
def minimal_portfolio():
    """Minimum viable portfolio (2 stocks)"""
    return {
        'symbols': ['AAPL', 'MSFT'],
        'weights': [0.6, 0.4]
    }


# ============================================================================
# MOCK API FIXTURES
# ============================================================================

@pytest.fixture
def mock_risk_api_success():
    """Mock successful risk analysis API response"""
    client = Mock()
    client.analyze_risk.return_value = {
        'metrics': {
            'sharpe_ratio': 1.25,
            'sortino_ratio': 1.45,
            'annualized_volatility': 0.18,
            'max_drawdown_pct': -15.5,
            'portfolio_var_95': -0.025,
            'portfolio_cvar_95': -0.035,
            'skewness': -0.15,
            'kurtosis': 3.2
        },
        'status': 'success'
    }
    return client

@pytest.fixture
def mock_risk_api_failure():
    """Mock failed API response"""
    client = Mock()
    client.analyze_risk.side_effect = ConnectionError("API unavailable")
    return client

@pytest.fixture
def mock_optimization_api():
    """Mock portfolio optimization API"""
    client = Mock()
    client.optimize_portfolio.return_value = {
        'optimized_weights': {
            'AAPL': 0.35,
            'MSFT': 0.35,
            'GOOGL': 0.30
        },
        'expected_return': 0.12,
        'volatility': 0.16,
        'sharpe_ratio': 1.35
    }
    return client

@pytest.fixture
def mock_var_api():
    """Mock VaR calculation API"""
    client = Mock()
    client.calculate_var.return_value = {
        'var_cvar_estimates': {
            '95%': {
                'var': -0.025,
                'cvar': -0.035
            },
            '99%': {
                'var': -0.042,
                'cvar': -0.055
            }
        }
    }
    return client

@pytest.fixture
def mock_stress_test_api():
    """Mock stress testing API"""
    client = Mock()
    client.stress_test.return_value = {
        'stress_scenarios': {
            'market_crash_2008': {
                'total_loss_pct': -35.5,
                'max_daily_loss_pct': -8.2,
                'max_drawdown_pct': -42.1
            },
            'covid_crash_2020': {
                'total_loss_pct': -28.3,
                'max_daily_loss_pct': -12.1,
                'max_drawdown_pct': -35.2
            }
        },
        'worst_case_scenario': 'market_crash_2008',
        'resilience_score': 62.5
    }
    return client


# ============================================================================
# SESSION STATE FIXTURES
# ============================================================================

@pytest.fixture
def mock_streamlit_session():
    """Mock Streamlit session state"""
    return {
        'portfolio_symbols': [],
        'portfolio_weights': [],
        'last_analysis_time': None,
        'dashboard_metrics': {},
        'optimization_result': None,
        'comprehensive_risk': None
    }

@pytest.fixture
def populated_session(sample_portfolio):
    """Session state with a portfolio loaded"""
    return {
        'portfolio_symbols': sample_portfolio['symbols'],
        'portfolio_weights': sample_portfolio['weights'],
        'last_analysis_time': datetime.now(),
        'dashboard_metrics': {
            'sharpe_ratio': 1.2,
            'annual_volatility': 0.18
        }
    }


# ============================================================================
# CACHE FIXTURES
# ============================================================================

@pytest.fixture
def mock_cache():
    """Mock Streamlit cache"""
    cache_store = {}
    
    def cache_data(ttl=600):
        def decorator(func):
            def wrapper(*args, **kwargs):
                key = str(args) + str(kwargs)
                if key not in cache_store:
                    cache_store[key] = func(*args, **kwargs)
                return cache_store[key]
            return wrapper
        return decorator
    
    return cache_data


# ============================================================================
# TIME-RELATED FIXTURES
# ============================================================================

@pytest.fixture
def recent_timestamp():
    """Timestamp from 1 hour ago"""
    return datetime.now() - timedelta(hours=1)

@pytest.fixture
def stale_timestamp():
    """Timestamp from 5 days ago"""
    return datetime.now() - timedelta(days=5)

@pytest.fixture
def fresh_timestamp():
    """Current timestamp"""
    return datetime.now()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@pytest.fixture
def assert_weights_normalized():
    """Helper to assert weights sum to 1.0"""
    def _assert(weights, tolerance=1e-10):
        total = sum(weights)
        assert abs(total - 1.0) < tolerance, f"Weights sum to {total}, expected 1.0"
        return True
    return _assert

@pytest.fixture
def assert_valid_symbols():
    """Helper to assert symbols are valid format"""
    def _assert(symbols):
        for symbol in symbols:
            assert isinstance(symbol, str), f"Symbol {symbol} is not a string"
            assert symbol.isalpha(), f"Symbol {symbol} contains non-alphabetic chars"
            assert len(symbol) <= 5, f"Symbol {symbol} is too long"
            assert len(symbol) >= 1, f"Symbol is empty"
        return True
    return _assert


# ============================================================================
# PYTEST CONFIGURATION HOOKS
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# ============================================================================
# AUTO-USE FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment before each test"""
    # Clear any global state
    yield
    # Cleanup after test


@pytest.fixture(autouse=True)
def mock_streamlit_imports(monkeypatch):
    """Mock Streamlit imports to avoid UI dependencies in tests"""
    # Mock the streamlit module
    mock_st = MagicMock()
    mock_st.session_state = {}
    mock_st.cache_data = lambda ttl=600: lambda f: f
    
    monkeypatch.setattr('streamlit.session_state', mock_st.session_state)
    monkeypatch.setattr('streamlit.cache_data', mock_st.cache_data)