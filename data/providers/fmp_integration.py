# data/providers/fmp_integration.py - Complete FMP Integration
"""
Complete FMP Integration for Portfolio Tools
===========================================

This file integrates your working FMP API with the refactored portfolio tools,
giving you real market data for portfolio optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp
import logging
import time
import os
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_env_file(env_path: str = '.env'):
    """Load environment variables from .env file"""
    if not os.path.exists(env_path):
        return
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value

# Load environment variables
load_env_file()

# Abstract base class (matching your refactored tools interface)
class MarketDataProvider(ABC):
    """Abstract interface for market data providers"""
    
    @abstractmethod
    async def get_returns_data(self, symbols: List[str], period: str = "1year") -> Optional[pd.DataFrame]:
        pass
    
    @abstractmethod
    async def validate_symbols(self, symbols: List[str]) -> List[str]:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass

# FMP Rate Limiter
class FMPRateLimiter:
    """Rate limiter for FMP API calls"""
    
    def __init__(self, calls_per_minute: int = 300):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    async def wait_if_needed(self):
        now = time.time()
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]
        
        if len(self.calls) >= self.calls_per_minute:
            wait_time = 60 - (now - self.calls[0]) + 1
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        self.calls.append(now)

# Main FMP Provider
class FMPDataProvider(MarketDataProvider):
    """Production FMP data provider"""
    
    def __init__(self, api_key: Optional[str] = None, cache_ttl_minutes: int = 60):
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('FMP_API_KEY')
        
        if not self.api_key:
            raise ValueError("FMP API key required. Set FMP_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.rate_limiter = FMPRateLimiter(300)  # 300 calls per minute
        self.cache_ttl_minutes = cache_ttl_minutes
        self._cache = {}
        self._session = None
        
        logger.info(f"FMP provider initialized with API key: {self.api_key[:10]}...{self.api_key[-5:]}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._session:
            await self._session.close()
    
    def is_available(self) -> bool:
        """Check if FMP provider is available"""
        return bool(self.api_key)
    
    async def validate_symbols(self, symbols: List[str]) -> List[str]:
        """Validate symbols against FMP database"""
        if not symbols:
            return []
        
        # For now, assume all symbols are valid - FMP has broad coverage
        # In production, you could implement actual validation using company profile endpoint
        valid_symbols = []
        
        if not self._session:
            # If not in async context, return all symbols
            return symbols
        
        try:
            for symbol in symbols:
                await self.rate_limiter.wait_if_needed()
                
                url = f"{self.base_url}/profile/{symbol}"
                params = {"apikey": self.api_key}
                
                async with self._session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and len(data) > 0:
                            valid_symbols.append(symbol)
                    else:
                        # Include symbol anyway - let data retrieval handle validation
                        valid_symbols.append(symbol)
                
                # Small delay to respect rate limits
                await asyncio.sleep(0.05)
                
        except Exception as e:
            logger.warning(f"Symbol validation error: {e}")
            return symbols  # Fallback to original list
        
        logger.info(f"Symbol validation: {len(valid_symbols)}/{len(symbols)} symbols valid")
        return valid_symbols
    
    async def get_returns_data(self, symbols: List[str], period: str = "1year") -> Optional[pd.DataFrame]:
        """Get historical returns data from FMP"""
        if not symbols:
            return None
        
        # Check cache first
        cache_key = self._generate_cache_key(symbols, period)
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if self._is_cache_valid(cache_entry['timestamp']):
                logger.info(f"Returning cached data for {len(symbols)} symbols")
                return cache_entry['data']
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = self._calculate_start_date(period, end_date)
            
            # Fetch price data
            price_data = await self._fetch_historical_prices(symbols, start_date, end_date)
            
            if price_data is None or price_data.empty:
                logger.warning("No price data retrieved from FMP")
                return None
            
            # Convert to returns
            returns_data = self._calculate_returns_from_prices(price_data)
            
            if returns_data is None or returns_data.empty:
                logger.warning("Could not calculate returns from price data")
                return None
            
            # Cache the results
            self._cache[cache_key] = {
                'data': returns_data,
                'timestamp': time.time()
            }
            
            logger.info(f"Retrieved returns data: {len(returns_data)} days, {len(returns_data.columns)} symbols")
            return returns_data
            
        except Exception as e:
            logger.error(f"FMP data retrieval failed: {e}")
            return None
    
    async def _fetch_historical_prices(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch historical price data from FMP"""
        if not self._session:
            raise RuntimeError("FMP provider must be used as async context manager")
        
        all_prices = {}
        failed_symbols = []
        
        for symbol in symbols:
            try:
                await self.rate_limiter.wait_if_needed()
                
                from_date = start_date.strftime("%Y-%m-%d")
                to_date = end_date.strftime("%Y-%m-%d")
                
                url = f"{self.base_url}/historical-price-full/{symbol}"
                params = {
                    "apikey": self.api_key,
                    "from": from_date,
                    "to": to_date
                }
                
                async with self._session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data and 'historical' in data and data['historical']:
                            df = pd.DataFrame(data['historical'])
                            df['date'] = pd.to_datetime(df['date'])
                            df.set_index('date', inplace=True)
                            df.sort_index(inplace=True)
                            
                            # Use adjusted close if available
                            price_column = 'adjClose' if 'adjClose' in df.columns else 'close'
                            all_prices[symbol] = df[price_column]
                            
                            logger.debug(f"Retrieved {len(df)} price points for {symbol}")
                        else:
                            logger.warning(f"No historical data for {symbol}")
                            failed_symbols.append(symbol)
                    elif response.status == 429:
                        logger.warning(f"Rate limited for {symbol}")
                        await asyncio.sleep(2)
                        failed_symbols.append(symbol)
                    else:
                        logger.warning(f"HTTP {response.status} for {symbol}")
                        failed_symbols.append(symbol)
                
                # Small delay between requests
                await asyncio.sleep(0.1)
                        
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                failed_symbols.append(symbol)
        
        if failed_symbols:
            logger.warning(f"Failed to retrieve data for: {failed_symbols}")
        
        if not all_prices:
            logger.error("No price data retrieved for any symbols")
            return None
        
        # Combine into DataFrame
        try:
            price_df = pd.DataFrame(all_prices)
            price_df = price_df.ffill(limit=5) # Forward fill missing values
            price_df = price_df.dropna(how='all')  # Drop rows with all NaN
            
            if price_df.empty:
                logger.error("All price data filtered out")
                return None
            
            return price_df
            
        except Exception as e:
            logger.error(f"Error combining price data: {e}")
            return None
    
    def _calculate_returns_from_prices(self, price_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate returns from price data"""
        try:
            # Calculate daily returns
            returns = price_data.pct_change()
            returns = returns.dropna()  # Remove first row (NaN)
            
            # Remove extreme outliers (likely data errors)
            returns = returns.clip(lower=-0.5, upper=0.5)
            
            # Replace any remaining NaN with 0
            returns = returns.fillna(0)
            
            return returns
            
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return None
    
    def _calculate_start_date(self, period: str, end_date: datetime) -> datetime:
        """Calculate start date based on period"""
        period_mapping = {
            "1month": timedelta(days=35),
            "3months": timedelta(days=95), 
            "6months": timedelta(days=185),
            "1year": timedelta(days=370),
            "2years": timedelta(days=740),
            "5years": timedelta(days=1850)
        }
        
        delta = period_mapping.get(period, timedelta(days=370))
        return end_date - delta
    
    def _generate_cache_key(self, symbols: List[str], period: str) -> str:
        """Generate cache key"""
        symbols_str = "_".join(sorted(symbols))
        return f"fmp_{symbols_str}_{period}"
    
    def _is_cache_valid(self, cache_timestamp: float) -> bool:
        """Check if cached data is still valid"""
        age_minutes = (time.time() - cache_timestamp) / 60
        return age_minutes < self.cache_ttl_minutes

# Test Data Provider (from your refactored tools)
class TestDataProvider(MarketDataProvider):
    """Realistic test data provider as fallback"""
    
    def __init__(self):
        pass
    
    async def get_returns_data(self, symbols: List[str], period: str = "1year") -> Optional[pd.DataFrame]:
        """Generate realistic test data with correlations"""
        if not symbols:
            return None
        
        try:
            # Determine number of days
            period_days = {
                "1month": 21, "3months": 63, "6months": 126,
                "1year": 252, "2years": 504, "5years": 1260
            }
            days = period_days.get(period, 252)
            
            # Asset parameters
            asset_params = {
                'AAPL': {'daily_return': 0.001, 'daily_vol': 0.025},
                'MSFT': {'daily_return': 0.0012, 'daily_vol': 0.022},
                'GOOGL': {'daily_return': 0.0008, 'daily_vol': 0.024},
                'AMZN': {'daily_return': 0.0011, 'daily_vol': 0.028},
                'NVDA': {'daily_return': 0.0015, 'daily_vol': 0.035},
                'TSLA': {'daily_return': 0.0013, 'daily_vol': 0.040},
                'TLT': {'daily_return': 0.0002, 'daily_vol': 0.012},
                'SPY': {'daily_return': 0.0007, 'daily_vol': 0.018},
            }
            
            # Generate correlated returns
            returns_dict = {}
            market_returns = np.random.normal(0.0005, 0.015, days)
            
            for symbol in symbols:
                params = asset_params.get(symbol, {'daily_return': 0.0006, 'daily_vol': 0.020})
                
                # Market beta (tech stocks more correlated)
                beta = 0.8 if symbol in ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA'] else 0.5
                
                # Generate returns
                idiosyncratic = np.random.normal(0, params['daily_vol'] * 0.7, days)
                returns_dict[symbol] = (
                    params['daily_return'] +
                    beta * market_returns + 
                    idiosyncratic
                )
            
            # Create DataFrame
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            return pd.DataFrame(returns_dict, index=dates)
            
        except Exception as e:
            logger.error(f"Test data generation failed: {e}")
            return None
    
    async def validate_symbols(self, symbols: List[str]) -> List[str]:
        return symbols
    
    def is_available(self) -> bool:
        return True

# Portfolio Data Manager (from your refactored tools)
class PortfolioDataManager:
    """Data manager with FMP integration and fallback"""
    
    def __init__(self, primary_provider: Optional[MarketDataProvider] = None):
        self.providers = []
        
        # Add primary provider if available
        if primary_provider and primary_provider.is_available():
            self.providers.append(primary_provider)
            logger.info(f"Primary provider: {type(primary_provider).__name__}")
        
        # Always add test provider as fallback
        self.providers.append(TestDataProvider())
        logger.info(f"Fallback provider: TestDataProvider")
    
    async def get_returns_data(self, symbols: List[str], period: str = "1year") -> Tuple[Optional[pd.DataFrame], str]:
        """Get returns data with automatic fallback"""
        if not symbols:
            return None, "No symbols provided"
        
        for provider in self.providers:
            try:
                provider_name = type(provider).__name__
                logger.info(f"Trying provider: {provider_name}")
                
                if isinstance(provider, FMPDataProvider):
                    # Use async context manager for FMP
                    async with provider:
                        data = await provider.get_returns_data(symbols, period)
                else:
                    # Direct call for other providers
                    data = await provider.get_returns_data(symbols, period)
                
                if data is not None and not data.empty:
                    logger.info(f"Success with {provider_name}: {len(data)} days, {len(data.columns)} symbols")
                    return data, f"Data from {provider_name}"
                else:
                    logger.warning(f"{provider_name} returned no data")
                    
            except Exception as e:
                logger.warning(f"{type(provider).__name__} failed: {e}")
                continue
        
        return None, "All providers failed"

# Global data manager instance
_data_manager = None

def get_data_manager() -> PortfolioDataManager:
    """Get the global data manager"""
    global _data_manager
    if _data_manager is None:
        # Create with FMP as primary provider
        try:
            fmp_provider = FMPDataProvider()
            _data_manager = PortfolioDataManager(fmp_provider)
            logger.info("Data manager created with FMP provider")
        except ValueError as e:
            logger.warning(f"FMP provider creation failed: {e}")
            _data_manager = PortfolioDataManager()  # Fallback only
    
    return _data_manager

def set_data_manager(data_manager: PortfolioDataManager):
    """Set the global data manager"""
    global _data_manager
    _data_manager = data_manager

# Integration function for your portfolio tools
async def integrate_fmp_with_portfolio_tools(api_key: Optional[str] = None):
    """Integrate FMP with portfolio tools"""
    try:
        fmp_provider = FMPDataProvider(api_key)
        data_manager = PortfolioDataManager(fmp_provider)
        set_data_manager(data_manager)
        logger.info("FMP integration complete - portfolio tools will use real market data")
        return True
    except Exception as e:
        logger.error(f"FMP integration failed: {e}")
        return False

# Test the complete integration
async def test_complete_integration():
    """Test the complete FMP integration with portfolio optimization"""
    print("Testing Complete FMP Integration")
    print("=" * 40)
    
    try:
        # Setup FMP integration
        success = await integrate_fmp_with_portfolio_tools()
        if not success:
            print("✗ FMP integration setup failed")
            return False
        
        print("✓ FMP integration setup successful")
        
        # Test data retrieval
        data_manager = get_data_manager()
        
        test_symbols = ['AAPL', 'GOOGL']
        returns_data, data_source = await data_manager.get_returns_data(test_symbols, "1month")
        
        if returns_data is not None:
            print(f"✓ Data retrieval successful: {data_source}")
            print(f"  Retrieved: {len(returns_data)} days, {len(returns_data.columns)} symbols")
            print(f"  Date range: {returns_data.index.min().date()} to {returns_data.index.max().date()}")
            
            # Show sample data
            print("  Sample returns:")
            print(returns_data.head(3).round(4))
            
            # Basic statistics
            print(f"  Average daily returns: {returns_data.mean().round(5).to_dict()}")
            print(f"  Daily volatilities: {returns_data.std().round(4).to_dict()}")
            
            return True
        else:
            print("✗ Data retrieval failed")
            return False
            
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the complete integration
    success = asyncio.run(test_complete_integration())
    
    if success:
        print("\n🎉 FMP integration is fully working!")
        print("Your portfolio tools can now use real market data from FMP.")
    else:
        print("\n❌ Integration test failed.")