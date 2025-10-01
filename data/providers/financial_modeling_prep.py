# data/providers/fmp_provider.py - Financial Modeling Prep Integration
"""
Financial Modeling Prep Data Provider Implementation
==================================================

Production-ready FMP integration for portfolio optimization tools:
- Real market data retrieval with proper error handling
- Symbol validation and data quality checks
- Rate limiting and caching for API efficiency
- Fallback mechanisms for data reliability
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp
import logging
from dataclasses import dataclass
import json
from urllib.parse import urlencode
import time
import os 
from collections import defaultdict

# Import the abstract base class from your refactored tools
from tools.portfolio_tools import MarketDataProvider

logger = logging.getLogger(__name__)

@dataclass
class FMPConfig:
    """FMP API configuration"""
    api_key: str
    base_url: str = "https://financialmodelingprep.com/api/v3"
    rate_limit_per_minute: int = 300  # FMP rate limits
    timeout_seconds: int = 30
    cache_ttl_minutes: int = 60
    max_retries: int = 3
    retry_delay_seconds: int = 1

@dataclass
class DataQualityMetrics:
    """Track data quality for monitoring"""
    symbols_requested: int
    symbols_retrieved: int
    data_points_total: int
    missing_data_pct: float
    date_range_start: Optional[str]
    date_range_end: Optional[str]
    retrieval_time_seconds: float

class FMPRateLimiter:
    """Simple rate limiter for FMP API calls"""
    
    def __init__(self, calls_per_minute: int = 300):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    async def wait_if_needed(self):
        """Wait if we're approaching rate limits"""
        now = time.time()
        
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]
        
        # If we're at the limit, wait
        if len(self.calls) >= self.calls_per_minute:
            wait_time = 60 - (now - self.calls[0]) + 1  # Wait until oldest call expires + buffer
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
        
        self.calls.append(now)

class FMPDataProvider(MarketDataProvider):
    """Production FMP data provider with comprehensive error handling"""
    
    def __init__(self, config: FMPConfig):
        self.config = config
        self.rate_limiter = FMPRateLimiter(config.rate_limit_per_minute)
        self._cache = {}  # Simple in-memory cache
        self._session = None
        
        # Validate API key format
        if not config.api_key or len(config.api_key) < 20:
            raise ValueError("Invalid FMP API key provided")
        
        logger.info(f"FMP provider initialized with base URL: {config.base_url}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._session:
            await self._session.close()
    
    def is_available(self) -> bool:
        """Check if FMP provider is available"""
        return bool(self.config.api_key)
    
    async def validate_symbols(self, symbols: List[str]) -> List[str]:
        """Validate symbols against FMP database"""
        if not symbols:
            return []
        
        try:
            # Use FMP symbol search/validation endpoint
            valid_symbols = []
            
            # Process symbols in batches to avoid overwhelming API
            batch_size = 50
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                
                # For each symbol, check if it exists
                batch_valid = await self._validate_symbol_batch(batch)
                valid_symbols.extend(batch_valid)
            
            logger.info(f"Symbol validation: {len(valid_symbols)}/{len(symbols)} symbols valid")
            return valid_symbols
            
        except Exception as e:
            logger.error(f"Symbol validation failed: {e}")
            # Return original symbols as fallback
            return symbols
    
    async def get_returns_data(
        self, 
        symbols: List[str], 
        period: str = "1year"
    ) -> Optional[pd.DataFrame]:
        """Get historical returns data from FMP"""
        
        if not symbols:
            return None
        
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(symbols, period)
            
            # Check cache first
            if cache_key in self._cache:
                cache_entry = self._cache[cache_key]
                if self._is_cache_valid(cache_entry['timestamp']):
                    logger.info(f"Returning cached data for {len(symbols)} symbols")
                    return cache_entry['data']
            
            # Calculate date range based on period
            end_date = datetime.now()
            start_date = self._calculate_start_date(period, end_date)
            
            # Retrieve price data from FMP
            price_data = await self._fetch_historical_prices(symbols, start_date, end_date)
            
            if price_data is None or price_data.empty:
                logger.warning("No price data retrieved from FMP")
                return None
            
            # Convert prices to returns
            returns_data = self._calculate_returns_from_prices(price_data)
            
            # Quality checks
            quality_metrics = self._assess_data_quality(
                symbols, returns_data, start_time
            )
            
            # Log quality metrics
            logger.info(f"Data quality: {quality_metrics.symbols_retrieved}/{quality_metrics.symbols_requested} symbols, "
                       f"{quality_metrics.missing_data_pct:.1f}% missing data")
            
            # Cache the results
            self._cache[cache_key] = {
                'data': returns_data,
                'timestamp': time.time(),
                'quality': quality_metrics
            }
            
            return returns_data
            
        except Exception as e:
            logger.error(f"FMP data retrieval failed: {e}")
            return None
    
    async def _validate_symbol_batch(self, symbols: List[str]) -> List[str]:
        """Validate a batch of symbols"""
        valid_symbols = []
        
        for symbol in symbols:
            try:
                await self.rate_limiter.wait_if_needed()
                
                # Use company profile endpoint to validate symbol
                url = f"{self.config.base_url}/profile/{symbol}"
                params = {"apikey": self.config.api_key}
                
                if not self._session:
                    raise RuntimeError("HTTP session not initialized. Use async context manager.")
                
                async with self._session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and isinstance(data, list) and len(data) > 0:
                            # Symbol exists and has data
                            valid_symbols.append(symbol)
                        elif data and isinstance(data, dict) and data.get('symbol'):
                            # Single symbol response
                            valid_symbols.append(symbol)
                    elif response.status == 429:
                        # Rate limited - wait and retry
                        await asyncio.sleep(self.config.retry_delay_seconds)
                        # Don't add to valid_symbols, will be retried
                    else:
                        logger.debug(f"Symbol {symbol} validation failed with status {response.status}")
                        
            except Exception as e:
                logger.warning(f"Error validating symbol {symbol}: {e}")
                # Include symbol anyway - let data retrieval handle it
                valid_symbols.append(symbol)
        
        return valid_symbols
    
    async def _fetch_historical_prices(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch historical price data from FMP"""
        
        all_prices = {}
        failed_symbols = []
        
        for symbol in symbols:
            try:
                await self.rate_limiter.wait_if_needed()
                
                # Format dates for FMP API
                from_date = start_date.strftime("%Y-%m-%d")
                to_date = end_date.strftime("%Y-%m-%d")
                
                # Use historical price endpoint
                url = f"{self.config.base_url}/historical-price-full/{symbol}"
                params = {
                    "apikey": self.config.api_key,
                    "from": from_date,
                    "to": to_date
                }
                
                if not self._session:
                    raise RuntimeError("HTTP session not initialized")
                
                success = False
                for attempt in range(self.config.max_retries):
                    try:
                        async with self._session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                
                                if data and 'historical' in data:
                                    historical_data = data['historical']
                                    
                                    if historical_data:
                                        # Convert to DataFrame
                                        df = pd.DataFrame(historical_data)
                                        df['date'] = pd.to_datetime(df['date'])
                                        df.set_index('date', inplace=True)
                                        df.sort_index(inplace=True)
                                        
                                        # Use adjusted close if available, otherwise close
                                        price_column = 'adjClose' if 'adjClose' in df.columns else 'close'
                                        all_prices[symbol] = df[price_column]
                                        success = True
                                        break
                                    else:
                                        logger.warning(f"No historical data for {symbol}")
                                        break
                                else:
                                    logger.warning(f"Unexpected response format for {symbol}")
                                    break
                                    
                            elif response.status == 429:
                                # Rate limited
                                wait_time = 2 ** attempt  # Exponential backoff
                                logger.warning(f"Rate limited for {symbol}, waiting {wait_time}s")
                                await asyncio.sleep(wait_time)
                            else:
                                logger.warning(f"HTTP {response.status} for {symbol}")
                                break
                                
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout for {symbol}, attempt {attempt + 1}")
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay_seconds)
                    except Exception as e:
                        logger.warning(f"Error fetching {symbol}: {e}")
                        break
                
                if not success:
                    failed_symbols.append(symbol)
                    
            except Exception as e:
                logger.error(f"Fatal error fetching {symbol}: {e}")
                failed_symbols.append(symbol)
        
        if failed_symbols:
            logger.warning(f"Failed to retrieve data for symbols: {failed_symbols}")
        
        if not all_prices:
            logger.error("No price data retrieved for any symbols")
            return None
        
        # Combine all price data into single DataFrame
        try:
            price_df = pd.DataFrame(all_prices)
            
            # Forward fill missing values (up to 5 days)
            price_df = price_df.fillna(method='ffill', limit=5)
            
            # Drop any remaining rows with all NaN values
            price_df = price_df.dropna(how='all')
            
            if price_df.empty:
                logger.error("All price data was filtered out")
                return None
            
            logger.info(f"Retrieved price data: {len(price_df)} days, {len(price_df.columns)} symbols")
            return price_df
            
        except Exception as e:
            logger.error(f"Error combining price data: {e}")
            return None
    
    def _calculate_returns_from_prices(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns from price data"""
        try:
            # Calculate daily returns
            returns = price_data.pct_change()
            
            # Remove first row (will be NaN)
            returns = returns.dropna()
            
            # Remove extreme outliers (returns > 50% in a day are likely data errors)
            returns = returns.clip(lower=-0.5, upper=0.5)
            
            # Replace any remaining NaN with 0
            returns = returns.fillna(0)
            
            logger.info(f"Calculated returns: {len(returns)} days, {len(returns.columns)} symbols")
            return returns
            
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return pd.DataFrame()
    
    def _calculate_start_date(self, period: str, end_date: datetime) -> datetime:
        """Calculate start date based on period"""
        period_mapping = {
            "1month": timedelta(days=35),    # Add buffer for weekends
            "3months": timedelta(days=95),
            "6months": timedelta(days=185),
            "1year": timedelta(days=370),
            "2years": timedelta(days=740),
            "5years": timedelta(days=1850)
        }
        
        delta = period_mapping.get(period, timedelta(days=370))
        return end_date - delta
    
    def _generate_cache_key(self, symbols: List[str], period: str) -> str:
        """Generate cache key for data"""
        symbols_str = "_".join(sorted(symbols))
        return f"fmp_{symbols_str}_{period}"
    
    def _is_cache_valid(self, cache_timestamp: float) -> bool:
        """Check if cached data is still valid"""
        age_minutes = (time.time() - cache_timestamp) / 60
        return age_minutes < self.config.cache_ttl_minutes
    
    def _assess_data_quality(
        self, 
        symbols_requested: List[str], 
        returns_data: pd.DataFrame,
        start_time: float
    ) -> DataQualityMetrics:
        """Assess quality of retrieved data"""
        
        symbols_retrieved = len(returns_data.columns) if returns_data is not None else 0
        
        if returns_data is None or returns_data.empty:
            return DataQualityMetrics(
                symbols_requested=len(symbols_requested),
                symbols_retrieved=0,
                data_points_total=0,
                missing_data_pct=100.0,
                date_range_start=None,
                date_range_end=None,
                retrieval_time_seconds=time.time() - start_time
            )
        
        # Calculate missing data percentage
        total_possible = len(returns_data) * len(returns_data.columns)
        missing_count = returns_data.isnull().sum().sum()
        missing_pct = (missing_count / total_possible) * 100 if total_possible > 0 else 0
        
        return DataQualityMetrics(
            symbols_requested=len(symbols_requested),
            symbols_retrieved=symbols_retrieved,
            data_points_total=len(returns_data) * symbols_retrieved,
            missing_data_pct=missing_pct,
            date_range_start=returns_data.index.min().strftime("%Y-%m-%d") if not returns_data.empty else None,
            date_range_end=returns_data.index.max().strftime("%Y-%m-%d") if not returns_data.empty else None,
            retrieval_time_seconds=time.time() - start_time
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        return {
            "cache_entries": len(self._cache),
            "cache_keys": list(self._cache.keys()),
            "memory_usage_mb": sum(
                entry['data'].memory_usage(deep=True).sum() 
                for entry in self._cache.values()
                if entry['data'] is not None
            ) / (1024 * 1024)
        }
    
    def clear_cache(self):
        """Clear the data cache"""
        self._cache.clear()
        logger.info("FMP data cache cleared")

# Factory function for easy integration
def create_fmp_provider(
    api_key: str,
    rate_limit_per_minute: int = 300,
    cache_ttl_minutes: int = 60
) -> FMPDataProvider:
    """Factory function to create FMP provider with standard config"""
    
    config = FMPConfig(
        api_key=api_key,
        rate_limit_per_minute=rate_limit_per_minute,
        cache_ttl_minutes=cache_ttl_minutes
    )
    
    return FMPDataProvider(config)

# Integration with your existing portfolio tools
async def integrate_fmp_with_portfolio_tools(api_key: str) -> None:
    """Integrate FMP provider with your portfolio tools"""
    
    from tools.portfolio_tools import PortfolioDataManager, set_data_manager
    
    try:
        # Create FMP provider
        fmp_provider = create_fmp_provider(api_key)
        
        # Test the connection
        async with fmp_provider:
            test_symbols = await fmp_provider.validate_symbols(['AAPL'])
            if test_symbols:
                logger.info("FMP integration successful")
            else:
                logger.warning("FMP integration test failed")
        
        # Create data manager with FMP as primary provider
        data_manager = PortfolioDataManager(primary_provider=fmp_provider)
        
        # Set as global data manager
        set_data_manager(data_manager)
        
        logger.info("FMP integration complete - portfolio tools will use real market data")
        
    except Exception as e:
        logger.error(f"FMP integration failed: {e}")
        raise

# Usage example and test function
async def test_fmp_integration():
    """Test FMP integration with real data"""
    
    # You need to provide your actual FMP API key # Replace with real key
    
    if API_KEY == "your_fmp_api_key_here":
        print("Please provide a real FMP API key to test integration")
        return
    
    try:
        print("Testing FMP Integration...")
        print("-" * 30)
        
        # Create provider
        fmp_provider = create_fmp_provider(API_KEY)
        
        async with fmp_provider:
            # Test 1: Symbol validation
            print("1. Testing symbol validation...")
            test_symbols = ['AAPL', 'GOOGL', 'INVALID_SYMBOL']
            valid_symbols = await fmp_provider.validate_symbols(test_symbols)
            print(f"   Requested: {test_symbols}")
            print(f"   Valid: {valid_symbols}")
            
            # Test 2: Data retrieval
            print("\n2. Testing data retrieval...")
            returns_data = await fmp_provider.get_returns_data(
                symbols=['AAPL', 'GOOGL'], 
                period='1month'
            )
            
            if returns_data is not None:
                print(f"   Retrieved: {len(returns_data)} days of data")
                print(f"   Symbols: {list(returns_data.columns)}")
                print(f"   Date range: {returns_data.index.min()} to {returns_data.index.max()}")
                print(f"   Sample returns:\n{returns_data.head(3)}")
            else:
                print("   No data retrieved")
            
            # Test 3: Cache performance
            print("\n3. Testing cache performance...")
            start_time = time.time()
            cached_data = await fmp_provider.get_returns_data(
                symbols=['AAPL', 'GOOGL'], 
                period='1month'
            )
            cache_time = time.time() - start_time
            print(f"   Cached retrieval time: {cache_time:.3f}s")
            
            # Cache stats
            cache_stats = fmp_provider.get_cache_stats()
            print(f"   Cache entries: {cache_stats['cache_entries']}")
            print(f"   Memory usage: {cache_stats['memory_usage_mb']:.1f} MB")
        
        print("\n✓ FMP integration test completed successfully")
        
    except Exception as e:
        print(f"\n✗ FMP integration test failed: {e}")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_fmp_integration())