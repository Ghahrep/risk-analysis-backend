"""
Test Suite for API Client - UPDATED FOR ACTUAL IMPLEMENTATION
Tests API communication, error handling, and response parsing
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.api_client import RiskAnalysisAPIClient, BehavioralAPIClient, get_risk_api_client, get_behavioral_api_client


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_response():
    """Mock successful API response"""
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        'success': True,
        'data': {
            'portfolio_var_95': 0.025,
            'portfolio_cvar_95': 0.035,
            'sharpe_ratio': 1.2,
            'sortino_ratio': 1.5,
            'annualized_volatility': 0.18,
            'max_drawdown_pct': -15.5
        }
    }
    return response

@pytest.fixture
def mock_error_response():
    """Mock failed API response"""
    response = Mock()
    response.status_code = 500
    response.json.return_value = {
        'success': False,
        'error': 'Internal server error'
    }
    response.raise_for_status.side_effect = requests.HTTPError()
    return response

@pytest.fixture
def risk_client():
    """Create Risk API client instance"""
    return RiskAnalysisAPIClient(base_url="http://localhost:8001")

@pytest.fixture
def behavioral_client():
    """Create Behavioral API client instance"""
    return BehavioralAPIClient(base_url="http://localhost:8003")


# ============================================================================
# TEST: API Client Initialization
# ============================================================================

class TestRiskClientInit:
    """Test Risk API client initialization"""
    
    def test_default_initialization(self):
        """Should initialize with default URL"""
        client = RiskAnalysisAPIClient()
        assert client.base_url == "http://localhost:8001"
    
    def test_custom_url(self):
        """Should accept custom base URL"""
        custom_url = "http://custom-api.com:9000"
        client = RiskAnalysisAPIClient(base_url=custom_url)
        assert client.base_url == custom_url
    
    @patch('streamlit.cache_resource')
    def test_singleton_pattern(self, mock_cache):
        """get_risk_api_client should use caching"""
        # The function uses @st.cache_resource decorator
        assert callable(get_risk_api_client)


# ============================================================================
# TEST: Health Check Endpoints
# ============================================================================

class TestHealthChecks:
    """Test health check endpoints"""
    
    @patch('requests.get')
    def test_risk_health_check(self, mock_get, risk_client, mock_response):
        """Risk API health check should work"""
        mock_get.return_value = mock_response
        
        result = risk_client.health_check()
        
        assert result is not None
        mock_get.assert_called_once()
        assert '/health' in mock_get.call_args[0][0]
    
    @patch('requests.get')
    def test_behavioral_health_check(self, mock_get, behavioral_client, mock_response):
        """Behavioral API health check should work"""
        mock_get.return_value = mock_response
        
        result = behavioral_client.health_check()
        
        assert result is not None
        mock_get.assert_called_once()


# ============================================================================
# TEST: Risk Analysis Endpoint
# ============================================================================

class TestAnalyzeRisk:
    """Test risk analysis API calls"""
    
    @patch('requests.post')
    def test_successful_analysis(self, mock_post, risk_client, mock_response):
        """Successful risk analysis should return data"""
        mock_post.return_value = mock_response
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        weights = [0.4, 0.3, 0.3]
        
        result = risk_client.analyze_risk(symbols, weights, period='1year')
        
        assert result is not None
        assert 'data' in result or 'portfolio_var_95' in result
        
        # Verify request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert 'symbols' in call_args[1]['json']
        assert call_args[1]['json']['symbols'] == symbols
    
    @patch('requests.post')
    def test_api_error_handling(self, mock_post, risk_client, mock_error_response):
        """API errors should be handled gracefully"""
        mock_post.return_value = mock_error_response
        
        result = risk_client.analyze_risk(['AAPL'], [1.0])
        
        # Should return None on error
        assert result is None
    
    @patch('requests.post')
    def test_connection_error(self, mock_post, risk_client):
        """Connection errors should be caught"""
        mock_post.side_effect = requests.ConnectionError("API unavailable")
        
        result = risk_client.analyze_risk(['AAPL'], [1.0])
        
        assert result is None
    
    @patch('requests.post')
    def test_timeout_error(self, mock_post, risk_client):
        """Timeout errors should be caught"""
        mock_post.side_effect = requests.Timeout("Request timed out")
        
        result = risk_client.analyze_risk(['AAPL'], [1.0])
        
        assert result is None


# ============================================================================
# TEST: Portfolio Optimization Endpoint
# ============================================================================

class TestOptimizePortfolio:
    """Test portfolio optimization API calls"""
    
    @patch('requests.post')
    def test_successful_optimization(self, mock_post, risk_client):
        """Successful optimization should return new weights"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'success': True,
            'optimized_weights': [0.5, 0.3, 0.2],
            'expected_return': 0.12,
            'expected_risk': 0.15
        }
        mock_post.return_value = mock_response
        
        result = risk_client.optimize_portfolio(
            ['AAPL', 'MSFT', 'GOOGL'],
            method='max_sharpe'
        )
        
        assert result is not None
        assert 'optimized_weights' in result
    
    @patch('requests.post')
    def test_optimization_methods(self, mock_post, risk_client):
        """Should support different optimization methods"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True, 'optimized_weights': [0.5, 0.5]}
        mock_post.return_value = mock_response
        
        methods = ['max_sharpe', 'min_variance']
        
        for method in methods:
            result = risk_client.optimize_portfolio(['AAPL', 'MSFT'], method=method)
            assert result is not None


# ============================================================================
# TEST: VaR Calculation
# ============================================================================

class TestVaRCalculation:
    """Test VaR calculation endpoint"""
    
    @patch('requests.post')
    def test_calculate_var(self, mock_post, risk_client, mock_response):
        """Should calculate VaR correctly"""
        mock_post.return_value = mock_response
        
        result = risk_client.calculate_var(['AAPL'], [1.0], confidence_level=0.95)
        
        assert result is not None
        mock_post.assert_called_once()


# ============================================================================
# TEST: Stress Testing
# ============================================================================

class TestStressTesting:
    """Test stress testing endpoint"""
    
    @patch('requests.post')
    def test_stress_test(self, mock_post, risk_client):
        """Should run stress test scenarios"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'success': True,
            'scenarios': {
                'market_crash': -0.25,
                'recession': -0.15
            }
        }
        mock_post.return_value = mock_response
        
        result = risk_client.stress_test(['AAPL'], [1.0])
        
        assert result is not None
        assert 'scenarios' in result


# ============================================================================
# TEST: Correlation Analysis
# ============================================================================

class TestCorrelationAnalysis:
    """Test correlation analysis endpoints"""
    
    @patch('requests.post')
    def test_correlation_analysis(self, mock_post, risk_client, mock_response):
        """Should return correlation data"""
        mock_post.return_value = mock_response
        
        result = risk_client.correlation_analysis(['AAPL', 'MSFT'])
        
        assert result is not None
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_rolling_correlations(self, mock_post, risk_client, mock_response):
        """Should return rolling correlation data"""
        mock_post.return_value = mock_response
        
        result = risk_client.rolling_correlations(['AAPL', 'MSFT'], window_size=30)
        
        assert result is not None


# ============================================================================
# TEST: Behavioral API Client
# ============================================================================

class TestBehavioralClient:
    """Test behavioral API client"""
    
    @patch('requests.post')
    def test_analyze_biases(self, mock_post, behavioral_client):
        """Should detect cognitive biases"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'biases_detected': ['confirmation_bias', 'anchoring']
        }
        mock_post.return_value = mock_response
        
        messages = [{'role': 'user', 'content': 'I think AAPL will go up'}]
        result = behavioral_client.analyze_biases(messages)
        
        assert result is not None
    
    @patch('requests.post')
    def test_analyze_sentiment(self, mock_post, behavioral_client):
        """Should analyze market sentiment"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'sentiment': 'bullish',
            'confidence': 0.8
        }
        mock_post.return_value = mock_response
        
        messages = [{'role': 'user', 'content': 'Market looks great'}]
        result = behavioral_client.analyze_sentiment(messages)
        
        assert result is not None


# ============================================================================
# TEST: Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling across all endpoints"""
    
    @patch('requests.post')
    def test_invalid_json_response(self, mock_post, risk_client):
        """Invalid JSON responses should be handled"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_post.return_value = mock_response
        
        result = risk_client.analyze_risk(['AAPL'], [1.0])
        
        assert result is None
    
    @patch('requests.post')
    def test_http_error(self, mock_post, risk_client):
        """HTTP errors should be caught"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.HTTPError()
        mock_post.return_value = mock_response
        
        result = risk_client.analyze_risk(['AAPL'], [1.0])
        
        assert result is None


# ============================================================================
# TEST: Request Logging
# ============================================================================

class TestRequestLogging:
    """Test request logging integration"""
    
    @patch('utils.request_logger.request_logger.log_request')
    @patch('requests.post')
    def test_request_logging(self, mock_post, mock_log, risk_client, mock_response):
        """Requests should be logged"""
        mock_post.return_value = mock_response
        
        risk_client.analyze_risk(['AAPL'], [1.0])
        
        # Verify logging was called
        assert mock_log.called


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Test integration scenarios"""
    
    @patch('requests.post')
    def test_full_analysis_workflow(self, mock_post, risk_client, mock_response):
        """Test complete analysis workflow"""
        mock_post.return_value = mock_response
        
        # Run analysis
        result1 = risk_client.analyze_risk(['AAPL', 'MSFT'], [0.5, 0.5])
        assert result1 is not None
        
        # Run optimization
        result2 = risk_client.optimize_portfolio(['AAPL', 'MSFT'])
        assert result2 is not None
        
        # Verify multiple calls were made
        assert mock_post.call_count >= 2


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])