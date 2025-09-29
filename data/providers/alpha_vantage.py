# data/providers/alpha_vantage.py
"""
Alpha Vantage Market Data Provider
=================================

Real market data integration for Phase 5+ financial intelligence platform.
Provides historical and real-time market data with caching and validation.

Features:
- Historical daily/intraday price data
- Real-time quotes and fundamental data
- Data quality validation and cleaning
- Local caching to minimize API calls
- Error handling and rate limiting
- Integration with existing pandas-based tools
"""

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlphaVantageProvider:
    """
    Alpha Vantage API integration for institutional-grade market data.
    
    Free tier: 5 API calls/minute, 500 calls/day
    Premium tiers available for higher usage
    """
    
    def __init__(self, api_key: str = None, cache_dir: str = "data/cache"):
        """
        Initialize Alpha Vantage provider
        
        Args:
            api_key: Alpha Vantage API key (get free at alphavantage.co)
            cache_dir: Directory for caching market data
        """
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            logger.warning("No Alpha Vantage API key provided. Set ALPHA_VANTAGE_API_KEY environment variable.")
        
        self.base_url = "https://www.alphavantage.co/query"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.last_call_time = 0
        self.min_call_interval = 12  # seconds (5 calls/minute = 12 seconds between calls)
        
        # Data quality thresholds
        self.min_data_points = 20
        self.max_missing_ratio = 0.1
        
    def _make_api_call(self, params: Dict[str, str]) -> Optional[Dict]:
        """Make rate-limited API call to Alpha Vantage"""
        if not self.api_key:
            logger.error("API key required for Alpha Vantage calls")
            return None
            
        # Rate limiting
        time_since_last_call = time.time() - self.last_call_time
        if time_since_last_call < self.min_call_interval:
            sleep_time = self.min_call_interval - time_since_last_call
            logger.info(f"Rate limiting: sleeping {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
        
        params['apikey'] = self.api_key
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            self.last_call_time = time.time()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return None
            elif 'Note' in data:
                logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                return None
                
            return data
            
        except requests.RequestException as e:
            logger.error(f"API call failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {e}")
            return None
    
    def _get_cache_path(self, symbol: str, data_type: str) -> Path:
        """Get cache file path for symbol and data type"""
        return self.cache_dir / f"{symbol.upper()}_{data_type}.json"
    
    def _load_from_cache(self, symbol: str, data_type: str, max_age_hours: int = 24) -> Optional[Dict]:
        """Load data from cache if available and fresh"""
        cache_path = self._get_cache_path(symbol, data_type)
        
        if not cache_path.exists():
            return None
            
        try:
            # Check file age
            file_age = time.time() - cache_path.stat().st_mtime
            if file_age > max_age_hours * 3600:
                logger.info(f"Cache expired for {symbol} {data_type}")
                return None
            
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loaded {symbol} {data_type} from cache")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load from cache: {e}")
            return None
    
    def _save_to_cache(self, symbol: str, data_type: str, data: Dict) -> None:
        """Save data to cache"""
        cache_path = self._get_cache_path(symbol, data_type)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            logger.info(f"Cached {symbol} {data_type}")
        except Exception as e:
            logger.error(f"Failed to save to cache: {e}")
    
    def get_historical_data(self, 
                          symbol: str, 
                          period: str = "1year",
                          interval: str = "daily",
                          use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Get historical price data for symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            period: Data period ('1month', '3months', '1year', '2years', 'max')
            interval: Data interval ('daily', 'weekly', 'monthly')
            use_cache: Use cached data if available
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
            or None if data unavailable
        """
        symbol = symbol.upper()
        cache_key = f"{interval}_{period}"
        
        # Try cache first
        if use_cache:
            cached_data = self._load_from_cache(symbol, cache_key)
            if cached_data:
                return self._parse_time_series_data(cached_data)
        
        # Determine Alpha Vantage function
        function_map = {
            'daily': 'TIME_SERIES_DAILY_ADJUSTED',
            'weekly': 'TIME_SERIES_WEEKLY_ADJUSTED', 
            'monthly': 'TIME_SERIES_MONTHLY_ADJUSTED'
        }
        
        if interval not in function_map:
            logger.error(f"Unsupported interval: {interval}")
            return None
        
        params = {
            'function': function_map[interval],
            'symbol': symbol,
            'outputsize': 'full' if period in ['2years', 'max'] else 'compact'
        }
        
        data = self._make_api_call(params)
        if not data:
            return None
        
        # Save to cache
        if use_cache:
            self._save_to_cache(symbol, cache_key, data)
        
        df = self._parse_time_series_data(data)
        
        # Filter by period
        if df is not None and period != 'max':
            df = self._filter_by_period(df, period)
        
        return df
    
    def _parse_time_series_data(self, data: Dict) -> Optional[pd.DataFrame]:
        """Parse Alpha Vantage time series response into DataFrame"""
        # Find time series key (varies by function)
        time_series_key = None
        for key in data.keys():
            if 'Time Series' in key or 'Weekly' in key or 'Monthly' in key:
                time_series_key = key
                break
        
        if not time_series_key:
            logger.error("No time series data found in response")
            return None
        
        time_series = data[time_series_key]
        
        # Convert to DataFrame
        df_data = []
        for date_str, values in time_series.items():
            row = {
                'Date': pd.to_datetime(date_str),
                'Open': float(values.get('1. open', values.get('1. open', 0))),
                'High': float(values.get('2. high', values.get('2. high', 0))),
                'Low': float(values.get('3. low', values.get('3. low', 0))),
                'Close': float(values.get('4. close', values.get('4. close', 0))),
                'AdjClose': float(values.get('5. adjusted close', values.get('4. close', 0))),
                'Volume': int(float(values.get('6. volume', values.get('5. volume', 0))))
            }
            df_data.append(row)
        
        if not df_data:
            logger.error("No data points found")
            return None
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Data quality validation
        if not self._validate_data_quality(df):
            logger.warning(f"Data quality issues detected")
            return None
        
        return df
    
    def _filter_by_period(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """Filter DataFrame by period"""
        if period == 'max':
            return df
        
        period_days = {
            '1month': 30,
            '3months': 90, 
            '1year': 365,
            '2years': 730
        }
        
        if period not in period_days:
            logger.warning(f"Unknown period: {period}")
            return df
        
        cutoff_date = datetime.now() - timedelta(days=period_days[period])
        return df[df['Date'] >= cutoff_date].reset_index(drop=True)
    
    def get_real_time_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time quote for symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with current price, change, volume, etc.
        """
        symbol = symbol.upper()
        
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol
        }
        
        data = self._make_api_call(params)
        if not data or 'Global Quote' not in data:
            return None
        
        quote = data['Global Quote']
        
        return {
            'symbol': symbol,
            'price': float(quote.get('05. price', 0)),
            'change': float(quote.get('09. change', 0)),
            'change_percent': quote.get('10. change percent', '0%').strip('%'),
            'volume': int(float(quote.get('06. volume', 0))),
            'previous_close': float(quote.get('08. previous close', 0)),
            'timestamp': datetime.now()
        }
    
    def get_company_overview(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company fundamental data"""
        symbol = symbol.upper()
        
        # Try cache first (fundamentals change slowly)
        cached_data = self._load_from_cache(symbol, 'overview', max_age_hours=168)  # 1 week
        if cached_data:
            return cached_data.get('company_data')
        
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol
        }
        
        data = self._make_api_call(params)
        if not data:
            return None
        
        # Save to cache
        cache_data = {'company_data': data}
        self._save_to_cache(symbol, 'overview', cache_data)
        
        return data
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate that symbol exists and has data"""
        quote = self.get_real_time_quote(symbol)
        return quote is not None and quote['price'] > 0
    
    def _validate_data_quality(self, df: pd.DataFrame) -> bool:
        """Validate data quality for financial analysis"""
        if df is None or len(df) < self.min_data_points:
            logger.error(f"Insufficient data points: {len(df) if df is not None else 0}")
            return False
        
        # Check for missing values
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_ratio > self.max_missing_ratio:
            logger.error(f"Too many missing values: {missing_ratio:.2%}")
            return False
        
        # Check for valid price data
        if df['Close'].min() <= 0:
            logger.error("Invalid price data (non-positive values)")
            return False
        
        # Check for reasonable price movements (no >50% daily moves unless splits)
        returns = df['Close'].pct_change().dropna()
        extreme_moves = (returns.abs() > 0.5).sum()
        if extreme_moves > len(returns) * 0.01:  # More than 1% extreme moves
            logger.warning(f"Detected {extreme_moves} extreme price movements")
        
        return True
    
    def get_portfolio_data(self, symbols: List[str], period: str = "1year") -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols (for portfolio analysis)
        
        Args:
            symbols: List of stock symbols
            period: Data period
            
        Returns:
            Dict mapping symbol to DataFrame
        """
        portfolio_data = {}
        
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}")
            df = self.get_historical_data(symbol, period=period)
            
            if df is not None:
                portfolio_data[symbol] = df
            else:
                logger.warning(f"Failed to get data for {symbol}")
        
        return portfolio_data
    
    def prepare_for_risk_analysis(self, symbols: List[str], period: str = "1year") -> Optional[pd.DataFrame]:
        """
        Prepare data specifically for risk analysis tools
        
        Returns DataFrame with returns data compatible with existing risk_tools.py
        """
        portfolio_data = self.get_portfolio_data(symbols, period)
        
        if not portfolio_data:
            return None
        
        # Create returns DataFrame
        returns_data = {}
        
        for symbol, df in portfolio_data.items():
            if len(df) > 1:
                returns = df['AdjClose'].pct_change().dropna()
                returns_data[symbol] = returns
        
        if not returns_data:
            return None
        
        # Align dates and create unified DataFrame
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 20:  # Minimum for meaningful analysis
            logger.error("Insufficient aligned data for analysis")
            return None
        
        return returns_df

# Usage Examples and Integration
def create_sample_portfolio_analysis():
    """
    Example showing integration with existing risk analysis tools
    """
    # Initialize provider
    provider = AlphaVantageProvider()
    
    # Sample portfolio
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    # Get data for risk analysis
    returns_df = provider.prepare_for_risk_analysis(symbols, period="1year")
    
    if returns_df is not None:
        print(f"Retrieved {len(returns_df)} days of data for {len(returns_df.columns)} symbols")
        print(f"Data range: {returns_df.index.min()} to {returns_df.index.max()}")
        
        # This DataFrame can now be passed to your existing risk analysis tools
        # Example: calculate_portfolio_risk_metrics(returns_df)
        
        return returns_df
    else:
        print("Failed to retrieve portfolio data")
        return None

if __name__ == "__main__":
    # Test the provider
    provider = AlphaVantageProvider()
    
    # Test single symbol
    print("Testing AAPL data retrieval...")
    aapl_data = provider.get_historical_data('AAPL', period='3months')
    
    if aapl_data is not None:
        print(f"Retrieved {len(aapl_data)} days of AAPL data")
        print(aapl_data.tail())
    
    # Test portfolio analysis
    print("\nTesting portfolio data preparation...")
    create_sample_portfolio_analysis()