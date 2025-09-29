# data/utils/market_data_manager.py (Updated for FMP)
"""
Market Data Manager - Financial Modeling Prep Integration
=========================================================

Enhanced market data management using Financial Modeling Prep API.
Superior data quality and features compared to Alpha Vantage.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta

from ..providers.financial_modeling_prep import FinancialModelingPrepProvider
from .data_validator import DataValidator

logger = logging.getLogger(__name__)

class MarketDataManager:
    """
    Enhanced market data management using Financial Modeling Prep
    
    Provides institutional-grade data integration for your four-way platform:
    - Adjusted historical prices (proper split/dividend handling)
    - Company fundamentals for enhanced analysis  
    - Better data quality and validation
    - Higher rate limits for production use
    """
    
    def __init__(self, provider: Optional[FinancialModelingPrepProvider] = None):
        """Initialize with FMP provider and validator"""
        self.provider = provider or FinancialModelingPrepProvider()
        self.validator = DataValidator()
        
        # Cache for processed data
        self._portfolio_cache = {}
        self._returns_cache = {}
        self._fundamentals_cache = {}
        
    def get_portfolio_returns(self, 
                        symbols: List[str], 
                        period: str = "1year",
                        clean_data: bool = True) -> Optional[pd.DataFrame]:
        """
        Get portfolio returns data for risk analysis tools
        Enhanced with proper market index validation
        """
        cache_key = f"{'-'.join(sorted(symbols))}_{period}"
        
        # Check cache
        if cache_key in self._returns_cache:
            logger.info(f"Using cached returns for {len(symbols)} symbols")
            return self._returns_cache[cache_key]
        
        # Get raw price data from FMP
        portfolio_data = self.provider.get_portfolio_data(symbols, period)
        
        if not portfolio_data:
            logger.error("Failed to retrieve portfolio data from FMP")
            return None
        
        # Enhanced validation with market index support
        if clean_data:
            validation_report = self.validate_returns_data(portfolio_data)
            logger.info(f"Data validation: {validation_report['summary']}")
            
            # Keep only valid symbols
            portfolio_data = {
                symbol: df for symbol, df in portfolio_data.items()
                if symbol in validation_report['valid_symbols']
            }
            
            # Log market indices detected
            if validation_report.get('market_indices_detected'):
                logger.info(f"Market indices detected: {validation_report['market_indices_detected']}")
            
            # Log any warnings separately
            if validation_report.get('warnings'):
                for symbol, warnings in validation_report['warnings'].items():
                    logger.info(f"Validation warnings for {symbol}: {warnings}")
            
            # Clean data
            portfolio_data = {
                symbol: self.validator.clean_price_data(df)
                for symbol, df in portfolio_data.items()
            }
        
        # Convert to returns with proper column handling
        returns_data = {}
        min_length = float('inf')
        
        for symbol, df in portfolio_data.items():
            if len(df) > 1:
                # Use adjusted close with fallback to close
                if 'AdjClose' in df.columns:
                    returns = df['AdjClose'].pct_change().dropna()
                elif 'Close' in df.columns:
                    returns = df['Close'].pct_change().dropna()
                    logger.info(f"Using Close price for {symbol} (AdjClose not available)")
                else:
                    logger.warning(f"No price column found for {symbol}")
                    continue
                
                returns_data[symbol] = returns
                min_length = min(min_length, len(returns))
        
        if not returns_data:
            logger.error("No valid returns data available")
            return None
        
        # Align all series to same date range
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()  # Remove dates with any missing data
        
        if len(returns_df) < 20:
            logger.error(f"Insufficient aligned data: {len(returns_df)} days")
            return None
        
        # Cache result
        self._returns_cache[cache_key] = returns_df
        
        logger.info(f"Generated returns matrix: {returns_df.shape} ({len(returns_df)} days, {len(returns_df.columns)} symbols)")
        
        return returns_df
    
    def validate_returns_data(self, portfolio_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Validate portfolio returns data using the improved DataValidator logic
        """
        validation_report = {
            'valid_symbols': [],
            'invalid_symbols': [],
            'issues': {},
            'warnings': {},
            'market_indices_detected': [],
            'summary': {}
        }
        
        for symbol, df in portfolio_data.items():
            try:
                # Calculate returns
                if 'AdjClose' in df.columns:
                    returns = df['AdjClose'].pct_change().dropna()
                elif 'Close' in df.columns:
                    returns = df['Close'].pct_change().dropna()
                else:
                    validation_report['invalid_symbols'].append(symbol)
                    validation_report['issues'][symbol] = ["Missing price data columns"]
                    continue
                
                # Use DataValidator's improved method
                is_valid, messages = self.validator.validate_returns_data(returns, symbol=symbol)
                
                # Track market indices
                if self.validator.is_market_index(symbol):
                    validation_report['market_indices_detected'].append(symbol)
                
                if is_valid:
                    validation_report['valid_symbols'].append(symbol)
                    
                    # Separate warnings from issues in messages
                    warnings = [msg for msg in messages if 
                            'normal for market index' in msg.lower() or 
                            'acceptable for market index' in msg.lower() or
                            'monitored for market index' in msg.lower()]
                    if warnings:
                        validation_report['warnings'][symbol] = warnings
                    
                else:
                    # Check if "invalid" is actually just warnings for market indices
                    actual_issues = [msg for msg in messages if not (
                        'normal for market index' in msg.lower() or 
                        'acceptable for market index' in msg.lower() or
                        'monitored for market index' in msg.lower()
                    )]
                    
                    if actual_issues:
                        validation_report['invalid_symbols'].append(symbol)
                        validation_report['issues'][symbol] = actual_issues
                    else:
                        # Only warnings - treat as valid
                        validation_report['valid_symbols'].append(symbol)
                        validation_report['warnings'][symbol] = messages
                        logger.info(f"Market index {symbol} passes validation with warnings")
                        
            except Exception as e:
                logger.error(f"Validation error for {symbol}: {str(e)}")
                validation_report['invalid_symbols'].append(symbol)
                validation_report['issues'][symbol] = [f"Validation error: {str(e)}"]
        
        # Summary statistics
        total_symbols = len(portfolio_data)
        valid_count = len(validation_report['valid_symbols'])
        market_indices_count = len(validation_report['market_indices_detected'])
        
        validation_report['summary'] = {
            'total_symbols': total_symbols,
            'valid_symbols': valid_count,
            'invalid_symbols': total_symbols - valid_count,
            'market_indices_count': market_indices_count,
            'success_rate': valid_count / total_symbols if total_symbols > 0 else 0
        }
        
        return validation_report
    
    def get_enhanced_market_data_for_behavioral_analysis(self, 
                                                       symbols: List[str],
                                                       period: str = "3months") -> Dict[str, Any]:
        """
        Enhanced market context using FMP's fundamental data
        
        Provides richer context than basic price analysis:
        - Market volatility and recent performance
        - Sector composition and diversification
        - Valuation metrics for behavioral context
        - Beta and risk characteristics
        """
        returns_df = self.get_portfolio_returns(symbols, period)
        
        if returns_df is None:
            return {}
        
        # Basic market context from price data
        portfolio_returns = returns_df.mean(axis=1)  # Equal weighted portfolio
        
        market_context = {
            'current_volatility': portfolio_returns.std() * np.sqrt(252),
            'recent_return': portfolio_returns.tail(30).mean() * 252,
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'volatility_regime': self._classify_volatility_regime(portfolio_returns),
            'market_sentiment': self._estimate_market_sentiment(portfolio_returns)
        }
        
        # Enhanced context using FMP fundamental data
        try:
            enhanced_context = self.provider.get_enhanced_market_context(symbols)
            market_context.update({
                'portfolio_composition': enhanced_context.get('portfolio_metrics', {}),
                'sector_analysis': enhanced_context.get('sector_analysis', {}),
                'valuation_metrics': enhanced_context.get('valuation_metrics', {}),
                'risk_indicators': enhanced_context.get('risk_indicators', {})
            })
        except Exception as e:
            logger.warning(f"Could not get enhanced context: {e}")
        
        return market_context
    
    def get_fundamental_analysis_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get fundamental analysis data for enhanced decision making
        
        This data can inform both portfolio optimization and behavioral analysis
        """
        fundamental_data = {}
        
        for symbol in symbols:
            try:
                # Company profile
                profile = self.provider.get_company_profile(symbol)
                
                # Key metrics
                metrics = self.provider.get_key_metrics(symbol)
                
                # Financial ratios
                ratios = self.provider.get_financial_ratios(symbol)
                
                if profile or metrics or ratios:
                    fundamental_data[symbol] = {
                        'profile': profile,
                        'metrics': metrics, 
                        'ratios': ratios
                    }
                    
            except Exception as e:
                logger.warning(f"Failed to get fundamentals for {symbol}: {e}")
                continue
        
        return fundamental_data
    
    def get_portfolio_risk_characteristics(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Enhanced portfolio risk assessment using FMP data
        
        Combines price-based risk metrics with fundamental risk indicators
        """
        # Get returns-based risk metrics
        returns_df = self.get_portfolio_returns(symbols, period='1year')
        
        risk_characteristics = {
            'price_based_risk': {},
            'fundamental_risk': {},
            'diversification_analysis': {},
            'sector_exposure': {}
        }
        
        if returns_df is not None:
            # Individual asset risk metrics
            for symbol in returns_df.columns:
                returns_series = returns_df[symbol].dropna()
                
                if len(returns_series) > 30:
                    risk_characteristics['price_based_risk'][symbol] = {
                        'volatility': returns_series.std() * np.sqrt(252),
                        'sharpe_estimate': (returns_series.mean() * 252) / (returns_series.std() * np.sqrt(252)) if returns_series.std() > 0 else 0,
                        'max_1day_loss': returns_series.min(),
                        'var_95': returns_series.quantile(0.05),
                        'skewness': returns_series.skew(),
                        'kurtosis': returns_series.kurtosis()
                    }
        
        # Enhanced fundamental risk analysis
        try:
            fundamentals = self.get_fundamental_analysis_data(symbols)
            
            sector_counts = {}
            beta_values = []
            debt_ratios = []
            
            for symbol, data in fundamentals.items():
                profile = data.get('profile', {})
                metrics = data.get('metrics', {})
                
                if profile:
                    # Sector diversification
                    sector = profile.get('sector', 'Unknown')
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                    
                    # Beta collection
                    beta = profile.get('beta')
                    if beta and isinstance(beta, (int, float)):
                        beta_values.append(beta)
                
                if metrics:
                    # Debt analysis
                    debt_ratio = metrics.get('debtToEquity')
                    if debt_ratio and isinstance(debt_ratio, (int, float)):
                        debt_ratios.append(debt_ratio)
            
            # Portfolio-level fundamental characteristics
            risk_characteristics['fundamental_risk'] = {
                'average_beta': np.mean(beta_values) if beta_values else None,
                'beta_std': np.std(beta_values) if len(beta_values) > 1 else None,
                'average_debt_ratio': np.mean(debt_ratios) if debt_ratios else None,
                'high_debt_count': len([d for d in debt_ratios if d > 1.0]) if debt_ratios else 0
            }
            
            risk_characteristics['sector_exposure'] = sector_counts
            
            # Diversification score
            total_stocks = len(symbols)
            unique_sectors = len(sector_counts)
            risk_characteristics['diversification_analysis'] = {
                'sector_count': unique_sectors,
                'diversification_ratio': unique_sectors / total_stocks if total_stocks > 0 else 0,
                'largest_sector_weight': max(sector_counts.values()) / total_stocks if sector_counts else 0
            }
            
        except Exception as e:
            logger.warning(f"Enhanced fundamental analysis failed: {e}")
        
        return risk_characteristics
    
    def validate_symbols(self, symbols: List[str]) -> Dict[str, bool]:
        """Validate symbols using FMP"""
        validation_results = {}
        
        for symbol in symbols:
            try:
                is_valid = self.provider.validate_symbol(symbol)
                validation_results[symbol] = is_valid
            except Exception as e:
                logger.error(f"Failed to validate {symbol}: {e}")
                validation_results[symbol] = False
        
        return validation_results
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of FMP data integration"""
        return {
            'provider': 'Financial Modeling Prep',
            'provider_connected': self.provider.api_key is not None,
            'cache_size': len(self._returns_cache),
            'fundamentals_cache_size': len(self._fundamentals_cache),
            'last_update': datetime.now(),
            'enhanced_features': [
                'adjusted_prices',
                'company_fundamentals',
                'financial_ratios',
                'sector_analysis',
                'enhanced_risk_metrics'
            ],
            'available_functions': [
                'get_portfolio_returns',  # For risk_tools.py
                'get_enhanced_market_data_for_behavioral_analysis',  # For behavioral_tools.py
                'get_fundamental_analysis_data',  # Enhanced features
                'get_portfolio_risk_characteristics',  # Enhanced risk analysis
                'validate_symbols'
            ]
        }
    
    # Helper methods (same as before)
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _classify_volatility_regime(self, returns: pd.Series) -> str:
        """Classify current volatility regime"""
        current_vol = returns.tail(30).std() * np.sqrt(252)
        
        if current_vol < 0.15:
            return "low"
        elif current_vol < 0.25:
            return "normal" 
        else:
            return "high"
    
    def _estimate_market_sentiment(self, returns: pd.Series) -> str:
        """Estimate market sentiment from recent performance"""
        recent_returns = returns.tail(20)
        
        if recent_returns.mean() > 0.001:
            return "bullish"
        elif recent_returns.mean() < -0.001:
            return "bearish"
        else:
            return "neutral"

# Example integration with existing tools (Enhanced for FMP)
def integrate_with_risk_tools_fmp():
    """
    Enhanced integration example using FMP's superior data
    """
    # This would import your existing risk analysis tools
    # from tools.risk_tools import calculate_portfolio_risk_metrics
    
    # Initialize enhanced data manager
    data_manager = MarketDataManager()
    
    # Get real market data with adjusted prices
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    returns_df = data_manager.get_portfolio_returns(symbols, period='1year')
    
    if returns_df is not None:
        # Portfolio returns (equal-weighted)
        portfolio_returns = returns_df.mean(axis=1)
        
        # Enhanced risk characteristics
        risk_chars = data_manager.get_portfolio_risk_characteristics(symbols)
        
        print("Enhanced Risk Analysis with FMP Data:")
        print(f"Portfolio has {len(returns_df.columns)} assets over {len(returns_df)} days")
        print(f"Sector diversification: {risk_chars['diversification_analysis']}")
        print(f"Average portfolio beta: {risk_chars['fundamental_risk'].get('average_beta', 'N/A')}")
        
        # This would integrate with your existing risk tools
        # risk_metrics = calculate_portfolio_risk_metrics(portfolio_returns)
        # print(f"VaR (95%): {risk_metrics.get('var_95', 'N/A')}")
        
        return {
            'returns_data': portfolio_returns,
            'enhanced_characteristics': risk_chars
        }
    
    return None

if __name__ == "__main__":
    integrate_with_risk_tools_fmp()