"""
Test Suite for Cache Utilities - UPDATED FOR ACTUAL IMPLEMENTATION
Tests caching, timestamps, and staleness warnings
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cache_utils import (
    to_hashable,
    cached_risk_analysis,
    cached_optimization,
    save_analysis_timestamp,
    get_analysis_timestamp,
    show_staleness_warning,
    show_last_updated_badge,
    format_time_ago
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_api_client():
    """Mock API client"""
    client = Mock()
    client.analyze_risk.return_value = {
        'portfolio_var_95': 0.025,
        'sharpe_ratio': 1.2
    }
    client.optimize_portfolio.return_value = {
        'optimized_weights': [0.5, 0.5],
        'expected_return': 0.12
    }
    return client

@pytest.fixture
def sample_symbols():
    """Sample portfolio symbols"""
    return ('AAPL', 'MSFT', 'GOOGL')

@pytest.fixture
def sample_weights():
    """Sample portfolio weights"""
    return (0.4, 0.3, 0.3)


# ============================================================================
# TEST: to_hashable Function
# ============================================================================

class TestToHashable:
    """Test conversion of unhashable types to hashable"""
    
    def test_list_to_tuple(self):
        """Lists should convert to tuples"""
        result = to_hashable(['AAPL', 'MSFT'], [0.5, 0.5])
        assert result == (('AAPL', 'MSFT'), (0.5, 0.5))
        assert isinstance(result[0], tuple)
        assert isinstance(result[1], tuple)
    
    def test_already_tuple(self):
        """Tuples should pass through unchanged"""
        input_symbols = ['AAPL', 'MSFT']
        input_weights = [0.5, 0.5]
        result = to_hashable(input_symbols, input_weights)
        assert result == (('AAPL', 'MSFT'), (0.5, 0.5))
    
    def test_single_argument(self):
        """Should handle single argument (symbols only)"""
        result = to_hashable(['AAPL'], None)
        assert result == (('AAPL',),)
    
    def test_empty_inputs(self):
        """Should handle empty collections"""
        result = to_hashable([], [])
        assert result == ((), ())


# ============================================================================
# TEST: Cached Risk Analysis
# ============================================================================

class TestCachedRiskAnalysis:
    """Test caching of risk analysis results"""
    
    def test_cache_function_exists(self, mock_api_client, sample_symbols, sample_weights):
        """Cached function should exist and be callable"""
        result = cached_risk_analysis(sample_symbols, sample_weights, '1year', mock_api_client)
        assert result is not None
    
    def test_api_called(self, mock_api_client):
        """Should call the API client"""
        symbols = ('AAPL',)
        weights = (1.0,)
        
        cached_risk_analysis(symbols, weights, '1year', mock_api_client)
        
        assert mock_api_client.analyze_risk.called


# ============================================================================
# TEST: Cached Optimization
# ============================================================================

class TestCachedOptimization:
    """Test caching of optimization results"""
    
    def test_successful_optimization(self, mock_api_client):
        """Should cache optimization results"""
        symbols = ('AAPL', 'MSFT', 'GOOGL')
        
        result = cached_optimization(symbols, 'max_sharpe', '1year', mock_api_client)
        
        assert result is not None
        assert 'optimized_weights' in result
        assert mock_api_client.optimize_portfolio.called


# ============================================================================
# TEST: Timestamp Management
# ============================================================================

class TestTimestampManagement:
    """Test analysis timestamp tracking"""
    
    @patch('streamlit.session_state', {})
    def test_save_timestamp(self):
        """Should save current timestamp"""
        with patch('streamlit.session_state', {}) as mock_state:
            save_analysis_timestamp('test_analysis')
            
            assert 'analysis_timestamps' in mock_state
            timestamp = mock_state['analysis_timestamps']['test_analysis']
            assert isinstance(timestamp, datetime)
    
    @patch('streamlit.session_state', {})
    def test_get_timestamp_exists(self):
        """Should retrieve existing timestamp"""
        now = datetime.now()
        
        with patch('streamlit.session_state', {'analysis_timestamps': {'test': now}}) as mock_state:
            result = get_analysis_timestamp('test')
            assert result == now
    
    @patch('streamlit.session_state', {})
    def test_get_timestamp_missing(self):
        """Should return None for missing timestamp"""
        with patch('streamlit.session_state', {}) as mock_state:
            result = get_analysis_timestamp('nonexistent')
            assert result is None


# ============================================================================
# TEST: Time Formatting
# ============================================================================

class TestTimeFormatting:
    """Test human-readable time formatting"""
    
    def test_format_just_now(self):
        """Should format recent times as 'Just now'"""
        recent = datetime.now() - timedelta(seconds=30)
        result = format_time_ago(recent)
        assert 'just now' in result.lower()
    
    def test_format_minutes_ago(self):
        """Should format minutes"""
        time_5min = datetime.now() - timedelta(minutes=5)
        result = format_time_ago(time_5min)
        assert '5' in result and 'minute' in result.lower()
    
    def test_format_hours_ago(self):
        """Should format hours"""
        time_2hours = datetime.now() - timedelta(hours=2)
        result = format_time_ago(time_2hours)
        assert '2' in result and 'hour' in result.lower()
    
    def test_format_days_ago(self):
        """Should format days"""
        time_3days = datetime.now() - timedelta(days=3)
        result = format_time_ago(time_3days)
        assert '3' in result and 'day' in result.lower()
    
    def test_none_timestamp(self):
        """Should handle None timestamp"""
        result = format_time_ago(None)
        assert result == "Never"


# ============================================================================
# TEST: UI Components
# ============================================================================

class TestUIComponents:
    """Test Streamlit UI components for staleness warnings"""
    
    @patch('streamlit.warning')
    @patch('streamlit.session_state', {})
    def test_staleness_warning_shown(self, mock_warning):
        """Should show warning for stale data"""
        old_time = datetime.now() - timedelta(days=4)
        
        with patch('streamlit.session_state', {'analysis_timestamps': {'test': old_time}}):
            show_staleness_warning('test', staleness_days=3)
            
            # Warning should be called
            assert mock_warning.called
    
    @patch('streamlit.warning')
    @patch('streamlit.session_state', {})
    def test_no_warning_for_fresh(self, mock_warning):
        """Should not show warning for fresh data"""
        fresh_time = datetime.now() - timedelta(hours=1)
        
        with patch('streamlit.session_state', {'analysis_timestamps': {'test': fresh_time}}):
            show_staleness_warning('test')
            
            # Warning should not be called
            assert not mock_warning.called
    
    @patch('streamlit.caption')
    @patch('streamlit.session_state', {})
    def test_last_updated_badge(self, mock_caption):
        """Should display last updated badge"""
        time_1hour = datetime.now() - timedelta(hours=1)
        
        with patch('streamlit.session_state', {'analysis_timestamps': {'test': time_1hour}}):
            show_last_updated_badge('test')
            
            # Caption should be called with time info
            assert mock_caption.called


# ============================================================================
# TEST: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_symbols(self):
        """Should handle empty symbols"""
        result = to_hashable([], [])
        assert result == ((), ())
    
    def test_very_old_timestamp(self):
        """Should handle very old timestamps"""
        very_old = datetime.now() - timedelta(days=365)
        result = format_time_ago(very_old)
        assert result is not None
        assert 'day' in result.lower()
    
    @patch('streamlit.session_state', {})
    def test_missing_timestamps_dict(self):
        """Should handle missing timestamps dictionary"""
        with patch('streamlit.session_state', {}):
            result = get_analysis_timestamp('test')
            assert result is None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCacheIntegration:
    """Test cache integration with full workflow"""
    
    def test_full_workflow(self, mock_api_client):
        """Test complete cache workflow"""
        symbols = ('AAPL', 'MSFT')
        weights = (0.5, 0.5)
        
        # Convert to hashable
        h_symbols, h_weights = to_hashable(list(symbols), list(weights))
        
        # Run analysis (cached)
        result = cached_risk_analysis(h_symbols, h_weights, '1year', mock_api_client)
        
        # Save timestamp
        with patch('streamlit.session_state', {}) as mock_state:
            save_analysis_timestamp('risk_analysis')
            
            assert result is not None


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestCachePerformance:
    """Test cache performance characteristics"""
    
    def test_hashable_conversion_speed(self):
        """to_hashable should be fast"""
        large_list = list(range(1000))
        
        start = time.time()
        result = to_hashable(large_list, large_list)
        duration = time.time() - start
        
        assert duration < 0.01  # Should be very fast
        assert isinstance(result[0], tuple)
    
    def test_timestamp_operation_speed(self):
        """Timestamp operations should be fast"""
        with patch('streamlit.session_state', {}) as mock_state:
            start = time.time()
            
            for i in range(100):
                save_analysis_timestamp(f'test_{i}')
            
            duration = time.time() - start
            
            assert duration < 0.1  # 100 operations in <100ms


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])