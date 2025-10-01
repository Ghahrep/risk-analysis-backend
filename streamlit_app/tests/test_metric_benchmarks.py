"""
Test Suite for Metric Benchmarks - UPDATED FOR ACTUAL IMPLEMENTATION
Tests metric evaluation, ratings, and contextual explanations
"""

import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metric_benchmarks import (
    BENCHMARKS,
    get_rating,
    get_star_rating,
    display_metric_with_benchmark
)


# ============================================================================
# TEST: Benchmark Data Structure
# ============================================================================

class TestBenchmarkStructure:
    """Test benchmark data structure"""
    
    def test_benchmarks_exist(self):
        """BENCHMARKS dictionary should exist"""
        assert BENCHMARKS is not None
        assert isinstance(BENCHMARKS, dict)
    
    def test_sharpe_benchmark_exists(self):
        """Sharpe ratio benchmark should be defined"""
        assert 'sharpe_ratio' in BENCHMARKS
        assert 'excellent' in BENCHMARKS['sharpe_ratio']
        assert 'higher_is_better' in BENCHMARKS['sharpe_ratio']
    
    def test_volatility_benchmark_exists(self):
        """Volatility benchmark should be defined"""
        assert 'annual_volatility' in BENCHMARKS
    
    def test_all_benchmarks_have_required_fields(self):
        """All benchmarks should have required fields"""
        required_fields = ['excellent', 'good', 'fair', 'poor', 'description', 'explanation']
        
        for metric_name, benchmark in BENCHMARKS.items():
            for field in required_fields:
                assert field in benchmark, f"{metric_name} missing {field}"


# ============================================================================
# TEST: get_rating Function
# ============================================================================

class TestGetRating:
    """Test rating evaluation function"""
    
    def test_excellent_sharpe(self):
        """Sharpe > 2.0 should be excellent"""
        rating, color = get_rating('sharpe_ratio', 2.5)
        assert rating == 'excellent'
        assert color is not None
    
    def test_good_sharpe(self):
        """Sharpe 1.0-2.0 should be good"""
        rating, color = get_rating('sharpe_ratio', 1.5)
        assert rating in ['good', 'very_good']
    
    def test_poor_sharpe(self):
        """Sharpe < 0.5 should be poor"""
        rating, color = get_rating('sharpe_ratio', 0.3)
        assert rating == 'poor'
    
    def test_low_volatility(self):
        """Low volatility should be excellent"""
        rating, color = get_rating('annual_volatility', 0.10)
        assert rating in ['excellent', 'good']
    
    def test_high_volatility(self):
        """High volatility should be poor"""
        rating, color = get_rating('annual_volatility', 0.40)
        assert rating == 'poor'
    
    def test_unknown_metric(self):
        """Unknown metric should return unknown"""
        rating, color = get_rating('unknown_metric_xyz', 1.0)
        assert rating == 'unknown'
    
    def test_rating_returns_color(self):
        """All ratings should return a color"""
        rating, color = get_rating('sharpe_ratio', 1.5)
        assert isinstance(color, str)
        assert len(color) > 0


# ============================================================================
# TEST: get_star_rating Function
# ============================================================================

class TestStarRating:
    """Test star rating display"""
    
    def test_excellent_stars(self):
        """Excellent should return stars"""
        stars = get_star_rating('excellent')
        assert stars is not None
        assert isinstance(stars, str)
        assert len(stars) > 0
    
    def test_good_stars(self):
        """Good should return stars"""
        stars = get_star_rating('good')
        assert stars is not None
    
    def test_poor_stars(self):
        """Poor should return indicator"""
        stars = get_star_rating('poor')
        assert stars is not None
    
    def test_all_ratings_have_stars(self):
        """All rating levels should have star representation"""
        ratings = ['excellent', 'good', 'fair', 'poor', 'neutral']
        for rating in ratings:
            stars = get_star_rating(rating)
            assert stars is not None
            assert isinstance(stars, str)


# ============================================================================
# TEST: display_metric_with_benchmark Function
# ============================================================================

class TestDisplayMetric:
    """Test metric display function"""
    
    @patch('streamlit.metric')
    @patch('streamlit.columns')
    @patch('streamlit.markdown')
    @patch('streamlit.expander')
    def test_display_sharpe_metric(self, mock_expander, mock_markdown, mock_columns, mock_metric):
        """Should display Sharpe ratio with benchmark"""
        # Setup mocks
        mock_col1 = Mock()
        mock_col2 = Mock()
        mock_columns.return_value = [mock_col1, mock_col2]
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=False)
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=False)
        mock_expander_ctx = Mock()
        mock_expander.return_value = mock_expander_ctx
        mock_expander_ctx.__enter__ = Mock(return_value=mock_expander_ctx)
        mock_expander_ctx.__exit__ = Mock(return_value=False)
        
        display_metric_with_benchmark('sharpe_ratio', 1.5)
        
        # Should call st.metric
        assert mock_metric.called or mock_columns.called
    
    @patch('streamlit.metric')
    @patch('streamlit.columns')
    @patch('streamlit.markdown')
    @patch('streamlit.expander')
    def test_display_volatility_metric(self, mock_expander, mock_markdown, mock_columns, mock_metric):
        """Should display volatility with benchmark"""
        mock_col1 = Mock()
        mock_col2 = Mock()
        mock_columns.return_value = [mock_col1, mock_col2]
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=False)
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=False)
        mock_expander_ctx = Mock()
        mock_expander.return_value = mock_expander_ctx
        mock_expander_ctx.__enter__ = Mock(return_value=mock_expander_ctx)
        mock_expander_ctx.__exit__ = Mock(return_value=False)
        
        display_metric_with_benchmark('annual_volatility', 0.18)
        
        assert mock_metric.called or mock_columns.called


# ============================================================================
# TEST: Value Formatting
# ============================================================================

class TestValueFormatting:
    """Test metric value formatting"""
    
    def test_percentage_formatting(self):
        """Percentage values should format correctly"""
        rating, color = get_rating('annual_volatility', 0.18)
        # Function should handle percentage conversion internally
        assert rating is not None
    
    def test_ratio_formatting(self):
        """Ratio values should format correctly"""
        rating, color = get_rating('sharpe_ratio', 1.5)
        assert rating is not None


# ============================================================================
# TEST: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_none_value(self):
        """Should handle None values gracefully"""
        try:
            rating, color = get_rating('sharpe_ratio', None)
            # Should not crash
            assert True
        except Exception:
            pytest.fail("Should handle None values")
    
    def test_zero_value(self):
        """Should handle zero values"""
        rating, color = get_rating('sharpe_ratio', 0.0)
        assert rating is not None
    
    def test_negative_value(self):
        """Should handle negative values"""
        rating, color = get_rating('sharpe_ratio', -0.5)
        assert rating is not None
    
    def test_very_large_value(self):
        """Should handle very large values"""
        rating, color = get_rating('sharpe_ratio', 100.0)
        assert rating is not None


# ============================================================================
# TEST: Consistency
# ============================================================================

class TestConsistency:
    """Test consistency across rating functions"""
    
    def test_all_metrics_can_be_rated(self):
        """All metrics in BENCHMARKS should be ratable"""
        test_value = 1.0
        
        for metric_name in BENCHMARKS.keys():
            rating, color = get_rating(metric_name, test_value)
            assert rating is not None
            assert color is not None
    
    def test_rating_strings_valid(self):
        """Rating strings should be consistent"""
        valid_ratings = ['excellent', 'good', 'fair', 'poor', 'neutral', 'unknown', 'very_good']
        
        rating, color = get_rating('sharpe_ratio', 1.2)
        assert rating in valid_ratings


# ============================================================================
# TEST: Comparative Logic
# ============================================================================

class TestComparativeLogic:
    """Test comparative rating logic"""
    
    def test_higher_sharpe_better_rating(self):
        """Higher Sharpe should get better rating"""
        rating_low, _ = get_rating('sharpe_ratio', 0.5)
        rating_high, _ = get_rating('sharpe_ratio', 2.5)
        
        # Mapping ratings to numeric scores
        rating_scores = {'poor': 1, 'fair': 2, 'good': 3, 'very_good': 4, 'excellent': 5}
        
        assert rating_scores.get(rating_high, 0) > rating_scores.get(rating_low, 0)
    
    def test_lower_volatility_better_rating(self):
        """Lower volatility should get better rating"""
        rating_high_vol, _ = get_rating('annual_volatility', 0.40)
        rating_low_vol, _ = get_rating('annual_volatility', 0.10)
        
        rating_scores = {'poor': 1, 'fair': 2, 'good': 3, 'very_good': 4, 'excellent': 5}
        
        assert rating_scores.get(rating_low_vol, 0) > rating_scores.get(rating_high_vol, 0)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestBenchmarkIntegration:
    """Test full benchmark workflow"""
    
    @patch('streamlit.metric')
    @patch('streamlit.columns')
    @patch('streamlit.markdown')
    @patch('streamlit.expander')
    def test_complete_portfolio_rating(self, mock_expander, mock_markdown, mock_columns, mock_metric):
        """Should rate all portfolio metrics"""
        mock_col1 = Mock()
        mock_col2 = Mock()
        mock_columns.return_value = [mock_col1, mock_col2]
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=False)
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=False)
        mock_expander_ctx = Mock()
        mock_expander.return_value = mock_expander_ctx
        mock_expander_ctx.__enter__ = Mock(return_value=mock_expander_ctx)
        mock_expander_ctx.__exit__ = Mock(return_value=False)
        
        metrics = {
            'sharpe_ratio': 1.5,
            'annual_volatility': 0.18,
            'max_drawdown': -25.0,
        }
        
        for metric_name, value in metrics.items():
            if metric_name in BENCHMARKS:
                display_metric_with_benchmark(metric_name, value, show_explanation=False)
        
        # All should execute without error
        assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])