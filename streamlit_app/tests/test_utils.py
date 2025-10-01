"""
Critical Utility Functions Test Suite - FIXED
Tests for portfolio_manager, error_handler, and cache_utils
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.portfolio_manager import (
    normalize_weights, 
    get_portfolio,
    set_portfolio,
    initialize_portfolio
)
from utils.error_handler import validate_portfolio, safe_api_call
from utils.cache_utils import to_hashable


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def valid_portfolio():
    """Standard valid portfolio for testing"""
    return {
        'symbols': ['AAPL', 'MSFT', 'GOOGL'],
        'weights': [0.4, 0.3, 0.3]
    }

@pytest.fixture
def mock_session_state():
    """Mock Streamlit session state"""
    return {
        'portfolio_symbols': [],
        'portfolio_weights': []
    }

@pytest.fixture
def mock_api_client():
    """Mock API client with typical response"""
    client = Mock()
    client.analyze_risk.return_value = {
        'metrics': {
            'sharpe_ratio': 1.2,
            'annualized_volatility': 0.18,
            'max_drawdown_pct': -15.5
        }
    }
    return client


# ============================================================================
# TEST: normalize_weights
# ============================================================================

class TestNormalizeWeights:
    """Test suite for normalize_weights function"""
    
    def test_already_normalized(self):
        """Weights that sum to 1.0 should remain unchanged"""
        weights = [0.4, 0.3, 0.3]
        result = normalize_weights(weights)
        assert result == weights
        assert abs(sum(result) - 1.0) < 1e-10
    
    def test_sum_greater_than_one(self):
        """Weights > 1.0 should be scaled down"""
        weights = [0.5, 0.5, 0.5]  # Sum = 1.5
        result = normalize_weights(weights)
        assert abs(sum(result) - 1.0) < 1e-10
        # Should be [0.333, 0.333, 0.333]
        for w in result:
            assert abs(w - 0.333333) < 1e-5
    
    def test_sum_less_than_one(self):
        """Weights < 1.0 should be scaled up"""
        weights = [0.2, 0.2, 0.1]  # Sum = 0.5
        result = normalize_weights(weights)
        assert abs(sum(result) - 1.0) < 1e-10
        # Should be [0.4, 0.4, 0.2]
        assert abs(result[0] - 0.4) < 1e-10
        assert abs(result[1] - 0.4) < 1e-10
        assert abs(result[2] - 0.2) < 1e-10
    
    def test_zero_sum(self):
        """Weights summing to 0 should raise ValueError"""
        weights = [0.0, 0.0, 0.0]
        with pytest.raises(ValueError, match="sum to zero"):
            normalize_weights(weights)
    
    def test_negative_weights(self):
        """Should handle negative weights (short positions)"""
        weights = [0.6, 0.3, -0.1]
        # This should work - sum is 0.8, normalize to 1.0
        result = normalize_weights(weights)
        assert abs(sum(result) - 1.0) < 1e-10
    
    def test_single_weight(self):
        """Single weight should become 1.0"""
        weights = [0.5]
        result = normalize_weights(weights)
        assert result == [1.0]
    
    def test_empty_list(self):
        """Empty list should raise ValueError"""
        with pytest.raises(ValueError, match="empty"):
            normalize_weights([])
    
    def test_very_small_weights(self):
        """Very small weights should still normalize correctly"""
        weights = [0.0001, 0.0001, 0.0001]
        result = normalize_weights(weights)
        assert abs(sum(result) - 1.0) < 1e-10
        for w in result:
            assert abs(w - 0.333333) < 1e-5
    
    def test_precision_preservation(self):
        """Should maintain reasonable precision"""
        weights = [0.333333, 0.333333, 0.333334]
        result = normalize_weights(weights)
        assert abs(sum(result) - 1.0) < 1e-10
        # All should be approximately equal
        assert abs(result[0] - result[1]) < 1e-5


# ============================================================================
# TEST: validate_portfolio (error_handler version)
# ============================================================================

class TestValidatePortfolio:
    """Test suite for validate_portfolio function"""
    
    def test_valid_portfolio(self, valid_portfolio):
        """Valid portfolio should return True, None"""
        is_valid, error = validate_portfolio(
            valid_portfolio['symbols'],
            valid_portfolio['weights']
        )
        assert is_valid is True
        assert error is None
    
    def test_mismatched_lengths(self):
        """Symbols and weights different lengths should fail"""
        symbols = ['AAPL', 'MSFT']
        weights = [0.5, 0.3, 0.2]
        is_valid, error = validate_portfolio(symbols, weights)
        assert is_valid is False
        assert 'length' in error.lower()
    
    def test_empty_symbols(self):
        """Empty symbols list should fail"""
        is_valid, error = validate_portfolio([], [])
        assert is_valid is False
        assert 'at least' in error.lower() or 'empty' in error.lower()
    
    def test_single_symbol(self):
        """Single symbol should fail (need at least 2)"""
        is_valid, error = validate_portfolio(['AAPL'], [1.0])
        assert is_valid is False
        assert 'at least 2' in error.lower()
    
    def test_invalid_symbol_format(self):
        """Invalid symbol format should fail"""
        invalid_cases = [
            (['AAPL', '123'], [0.5, 0.5]),  # Numbers
            (['AAPL', 'MS-FT'], [0.5, 0.5]),  # Special chars
            (['AAPL', ''], [0.5, 0.5]),  # Empty string
            (['AAPL', 'TOOLONG'], [0.5, 0.5]),  # Too long
        ]
        
        for symbols, weights in invalid_cases:
            is_valid, error = validate_portfolio(symbols, weights)
            assert is_valid is False
            assert error is not None
    
    def test_negative_weights(self):
        """Negative weights should fail"""
        symbols = ['AAPL', 'MSFT']
        weights = [0.6, -0.1]
        is_valid, error = validate_portfolio(symbols, weights)
        assert is_valid is False
        assert 'negative' in error.lower()
    
    def test_weights_not_normalized(self):
        """Weights not summing to ~1.0 should fail"""
        symbols = ['AAPL', 'MSFT']
        weights = [0.3, 0.3]  # Sum = 0.6
        is_valid, error = validate_portfolio(symbols, weights)
        assert is_valid is False
        assert '100%' in error or 'sum' in error.lower()
    
    def test_weights_tolerance(self):
        """Weights within tolerance should pass"""
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        weights = [0.333, 0.333, 0.334]  # Sum = 1.000
        is_valid, error = validate_portfolio(symbols, weights)
        assert is_valid is True
    
    def test_duplicate_symbols(self):
        """Duplicate symbols should be allowed (user might want this)"""
        symbols = ['AAPL', 'AAPL']
        weights = [0.5, 0.5]
        is_valid, error = validate_portfolio(symbols, weights)
        # This should pass - user might intentionally double a position
        assert is_valid is True
    
    def test_case_sensitivity(self):
        """Symbol validation should be case-insensitive"""
        symbols = ['aapl', 'MSFT']
        weights = [0.5, 0.5]
        # Should pass - symbols will be uppercased
        is_valid, error = validate_portfolio(symbols, weights)
        # This depends on implementation - document behavior
        assert is_valid is True


# ============================================================================
# TEST: safe_api_call
# ============================================================================

class TestSafeAPICall:
    """Test suite for safe_api_call error handler"""
    
    def test_successful_call(self):
        """Successful API call should return result"""
        def successful_func():
            return {'status': 'success', 'data': [1, 2, 3]}
        
        result = safe_api_call(successful_func, "test operation")
        assert result == {'status': 'success', 'data': [1, 2, 3]}
    
    def test_exception_handling(self):
        """Failed API call should return None and log error"""
        def failing_func():
            raise ConnectionError("API unavailable")
        
        result = safe_api_call(failing_func, "test operation")
        assert result is None
    
    def test_timeout_handling(self):
        """Timeout should be caught and return None"""
        def timeout_func():
            raise TimeoutError("Request timeout")
        
        result = safe_api_call(timeout_func, "test operation")
        assert result is None
    
    def test_generic_exception(self):
        """Generic exceptions should be caught"""
        def error_func():
            raise Exception("Unexpected error")
        
        result = safe_api_call(error_func, "test operation")
        assert result is None
    
    @patch('utils.request_logger.request_logger.log_error')
    def test_logging_on_error(self, mock_log_error):
        """Errors should be logged"""
        def failing_func():
            raise ValueError("Test error")
        
        safe_api_call(failing_func, "test operation")
        
        # Verify logger was called
        assert mock_log_error.called


# ============================================================================
# TEST: to_hashable (cache_utils)
# ============================================================================

class TestToHashable:
    """Test suite for to_hashable function"""
    
    def test_list_to_tuple(self):
        """Lists should convert to tuples"""
        symbols = ['AAPL', 'MSFT']
        weights = [0.5, 0.5]
        
        h_symbols, h_weights = to_hashable(symbols, weights)
        
        assert isinstance(h_symbols, tuple)
        assert isinstance(h_weights, tuple)
        assert h_symbols == ('AAPL', 'MSFT')
        assert h_weights == (0.5, 0.5)
    
    def test_already_hashable(self):
        """Already hashable types should pass through"""
        symbols = ('AAPL', 'MSFT')
        weights = (0.5, 0.5)
        
        h_symbols, h_weights = to_hashable(symbols, weights)
        
        assert h_symbols == symbols
        assert h_weights == weights
    
    def test_nested_lists(self):
        """Nested structures should be fully converted"""
        complex_data = [['A', 'B'], [0.5, 0.5]]
        
        result = to_hashable(complex_data, [])
        
        # Should convert nested lists to nested tuples
        assert isinstance(result[0], tuple)
    
    def test_empty_inputs(self):
        """Empty inputs should work"""
        result = to_hashable([], [])
        assert result == ((), ())


# ============================================================================
# TEST: Portfolio State Management
# ============================================================================

class TestPortfolioStateManagement:
    """Test portfolio get/set/initialize functions"""
    
    @patch('streamlit.session_state', {})
    def test_initialize_portfolio(self):
        """Initialize should create empty state"""
        with patch('streamlit.session_state', {}) as mock_state:
            initialize_portfolio()
            assert 'portfolio_symbols' in mock_state
            assert 'portfolio_weights' in mock_state
    
    @patch('streamlit.session_state', {})
    def test_set_and_get_portfolio(self):
        """Set then get should return same portfolio"""
        mock_state = {
            'portfolio_symbols': [],
            'portfolio_weights': []
        }
        
        with patch('streamlit.session_state', mock_state):
            symbols = ['AAPL', 'MSFT']
            weights = [0.6, 0.4]
            
            set_portfolio(symbols, weights)
            
            retrieved_symbols, retrieved_weights = get_portfolio()
            
            assert retrieved_symbols == symbols
            assert retrieved_weights == weights
    
    @patch('streamlit.session_state', {})
    def test_get_empty_portfolio(self):
        """Get on empty state should return empty lists"""
        mock_state = {
            'portfolio_symbols': [],
            'portfolio_weights': []
        }
        
        with patch('streamlit.session_state', mock_state):
            symbols, weights = get_portfolio()
            assert symbols == []
            assert weights == []


# ============================================================================
# TEST: Edge Cases and Integration
# ============================================================================

class TestEdgeCases:
    """Test edge cases and integration scenarios"""
    
    def test_normalize_then_validate(self):
        """Normalized weights should always pass validation"""
        test_cases = [
            [0.5, 0.5, 0.5],  # Sum > 1
            [0.2, 0.2, 0.2],  # Sum < 1
            [0.99, 0.01],  # Highly skewed
        ]
        
        for weights in test_cases:
            normalized = normalize_weights(weights)
            symbols = ['AAPL', 'MSFT'] if len(weights) == 2 else ['AAPL', 'MSFT', 'GOOGL']
            is_valid, error = validate_portfolio(symbols, normalized)
            assert is_valid, f"Normalized weights failed validation: {error}"
    
    def test_large_portfolio(self):
        """Should handle portfolios with many holdings"""
        symbols = [f'SYM{i}' for i in range(50)]
        weights = [1.0/50] * 50
        
        is_valid, error = validate_portfolio(symbols, weights)
        assert is_valid, f"Large portfolio failed: {error}"
    
    def test_precision_edge_case(self):
        """Test floating point precision issues"""
        # Weights that might have precision issues
        weights = [0.1] * 10  # Should sum to 1.0, but might be 0.9999999
        symbols = [f'SYM{i}' for i in range(10)]
        
        is_valid, error = validate_portfolio(symbols, weights)
        assert is_valid, f"Precision edge case failed: {error}"
    
    def test_unicode_symbols(self):
        """Should reject non-ASCII symbols"""
        symbols = ['AAPL', 'МSFT']  # Second uses Cyrillic M
        weights = [0.5, 0.5]
        
        is_valid, error = validate_portfolio(symbols, weights)
        # Should fail - symbols must be ASCII
        assert is_valid is False, "Unicode symbols should be rejected"


# ============================================================================
# PERFORMANCE TESTS (Optional)
# ============================================================================

class TestPerformance:
    """Performance regression tests"""
    
    def test_normalize_performance(self):
        """Normalize should be fast even with large portfolios"""
        import time
        
        weights = [1.0/1000] * 1000
        
        start = time.time()
        result = normalize_weights(weights)
        duration = time.time() - start
        
        assert duration < 0.01  # Should complete in <10ms
        assert abs(sum(result) - 1.0) < 1e-10
    
    def test_validate_performance(self):
        """Validation should be fast"""
        import time
        
        symbols = [f'SYM{i}' for i in range(100)]
        weights = [1.0/100] * 100
        
        start = time.time()
        is_valid, error = validate_portfolio(symbols, weights)
        duration = time.time() - start
        
        assert duration < 0.01  # Should complete in <10ms
        assert is_valid, f"Performance test portfolio failed: {error}"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])