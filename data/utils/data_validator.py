# data/utils/data_validator.py (Updated with Market Index Support)
"""
Data Quality Validation Utilities - Updated for Market Indices
==============================================================

Validation functions for market data used in financial analysis.
Ensures data quality standards for institutional-grade analytics.
Updated to handle market indices with naturally high kurtosis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataValidator:
    """Data quality validation for financial analysis"""
    
    def __init__(self):
        # Quality thresholds
        self.min_data_points = 20
        self.max_missing_ratio = 0.10
        self.max_zero_volume_ratio = 0.05
        self.max_extreme_return = 0.50  # 50% daily return threshold
        self.min_price = 0.01
        
        # Market indices that naturally have high kurtosis
        self.market_indices = {
            'SPY', 'QQQ', 'IWM', 'VTI', 'EFA', 'EEM', 'GLD', 'TLT', 
            'SHY', 'AGG', 'BND', 'HYG', 'LQD', 'ARKK', 'XLF', 'XLK',
            'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLRE', 'XLB',
            'VEA', 'VWO', 'IEFA', 'IEMG', 'VOO', 'VEU', 'BND', 'BNDX',
            'VXUS', 'VTV', 'VUG', 'VB', 'VBK', 'VBR', 'VO', 'VOE', 'VOT',
            'SPX', 'NDX', 'RUT', 'DJI'  # Index symbols
        }
    
    def validate_price_data(self, df: pd.DataFrame, symbol: str = None) -> Tuple[bool, List[str]]:
        """
        Comprehensive price data validation with market index support
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Optional symbol for specialized validation
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if df is None or df.empty:
            return False, ["No data provided"]
        
        # Required columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Sufficient data points
        if len(df) < self.min_data_points:
            issues.append(f"Insufficient data points: {len(df)} < {self.min_data_points}")
        
        # Missing values check
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_ratio > self.max_missing_ratio:
            issues.append(f"Too many missing values: {missing_ratio:.2%}")
        
        # Price validity
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns:
                if (df[col] <= 0).any():
                    issues.append(f"Non-positive prices found in {col}")
                if (df[col] < self.min_price).any():
                    issues.append(f"Suspiciously low prices in {col}")
        
        # OHLC logic validation
        if all(col in df.columns for col in price_cols):
            # High should be >= Low
            if (df['High'] < df['Low']).any():
                issues.append("High < Low detected")
            
            # Close should be between High and Low
            if ((df['Close'] > df['High']) | (df['Close'] < df['Low'])).any():
                issues.append("Close outside High-Low range")
        
        # Volume validation
        if 'Volume' in df.columns:
            if (df['Volume'] < 0).any():
                issues.append("Negative volume detected")
            
            zero_volume_ratio = (df['Volume'] == 0).sum() / len(df)
            if zero_volume_ratio > self.max_zero_volume_ratio:
                issues.append(f"Too many zero volume days: {zero_volume_ratio:.2%}")
        
        # Return validation
        if 'Close' in df.columns and len(df) > 1:
            returns = df['Close'].pct_change().dropna()
            extreme_returns = (returns.abs() > self.max_extreme_return).sum()
            if extreme_returns > 0:
                issues.append(f"Extreme returns detected: {extreme_returns} days > {self.max_extreme_return:.0%}")
        
        # Date validation
        if 'Date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                issues.append("Date column is not datetime type")
            else:
                # Check for future dates
                future_dates = (df['Date'] > datetime.now()).sum()
                if future_dates > 0:
                    issues.append(f"Future dates detected: {future_dates}")
                
                # Check for reasonable date range
                date_range = (df['Date'].max() - df['Date'].min()).days
                if date_range < 1:
                    issues.append("Invalid date range")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    

    def validate_returns_data(self, returns: pd.Series, symbol: str = None) -> Tuple[bool, List[str]]:
        """
        Validate returns series for risk analysis with market index support
        
        Args:
            returns: Returns series
            symbol: Optional symbol for specialized validation
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        warnings = []  # Separate warnings from validation failures
        
        if returns is None or returns.empty:
            return False, ["No returns data provided"]
        
        # Determine if this is a market index early for context-aware validation
        is_market_index = symbol and symbol.upper() in self.market_indices
        
        # Basic data sufficiency checks
        if len(returns) < self.min_data_points:
            issues.append(f"Insufficient returns: {len(returns)} < {self.min_data_points}")
        
        # Missing values check
        missing_count = returns.isnull().sum()
        missing_ratio = missing_count / len(returns)
        if missing_ratio > self.max_missing_ratio:
            issues.append(f"Too many missing returns: {missing_ratio:.2%}")
        
        # Work with clean data for further validation
        clean_returns = returns.dropna()
        
        if clean_returns.empty:
            return False, ["No valid return data after cleaning"]
        
        # Extreme values check with context-aware thresholds
        extreme_threshold = self.max_extreme_return
        if is_market_index:
            # Market indices can have more extreme moves during crises
            extreme_threshold = min(0.75, self.max_extreme_return * 1.5)  # 75% max or 1.5x normal
        
        extreme_count = (clean_returns.abs() > extreme_threshold).sum()
        if extreme_count > 0:
            if is_market_index and extreme_count <= 3:  # Allow few extreme days for indices
                warnings.append(f"Extreme returns detected: {extreme_count} observations > {extreme_threshold:.0%} (acceptable for market index)")
            else:
                issues.append(f"Extreme returns: {extreme_count} observations > {extreme_threshold:.0%}")
        
        # Statistical properties validation
        if len(clean_returns) > 1:  # Need at least 2 points for std calculation
            returns_std = clean_returns.std()
            
            # Check for constant returns (no variation)
            if returns_std == 0:
                issues.append("No variation in returns (constant values)")
            elif returns_std < 1e-6:  # Near-zero volatility
                warnings.append(f"Very low volatility detected: {returns_std:.6f}")
            
            # Statistical distribution checks
            try:
                skewness = clean_returns.skew()
                kurtosis = clean_returns.kurtosis()
                
                # Skewness validation (less strict for market indices)
                skew_threshold = 6 if is_market_index else 5
                if abs(skewness) > skew_threshold:
                    if is_market_index:
                        warnings.append(f"High skewness: {skewness:.2f} (monitored for market index)")
                    else:
                        issues.append(f"Extreme skewness: {skewness:.2f}")
                
                # Kurtosis validation with market index handling
                if is_market_index:
                    # Market indices: more lenient kurtosis thresholds
                    if kurtosis > 50:  # Extremely high even for indices
                        warnings.append(f"Very high kurtosis: {kurtosis:.2f} (extreme even for market index)")
                    elif kurtosis > 20:
                        warnings.append(f"High kurtosis: {kurtosis:.2f} (normal for market index)")
                        logger.info(f"High kurtosis ({kurtosis:.2f}) detected for market index {symbol} - within acceptable range")
                else:
                    # Individual stocks: standard thresholds
                    if kurtosis > 20:
                        issues.append(f"Extreme kurtosis: {kurtosis:.2f}")
                    elif kurtosis > 10:
                        warnings.append(f"High kurtosis: {kurtosis:.2f}")
                
            except Exception as e:
                warnings.append(f"Could not compute statistical properties: {str(e)}")
        
        # Additional market index specific checks
        if is_market_index:
            # Check for reasonable correlation with market behavior
            # (This could be enhanced with actual market correlation checks)
            mean_return = clean_returns.mean()
            if abs(mean_return) > 0.01:  # More than 1% average daily return
                warnings.append(f"Unusually high average daily return: {mean_return:.3f}")
        
        # Temporal consistency checks
        if len(clean_returns) > 5:
            # Look for suspicious patterns
            consecutive_zeros = 0
            max_consecutive_zeros = 0
            
            for ret in clean_returns:
                if ret == 0:
                    consecutive_zeros += 1
                    max_consecutive_zeros = max(max_consecutive_zeros, consecutive_zeros)
                else:
                    consecutive_zeros = 0
            
            if max_consecutive_zeros > 5:
                issues.append(f"Too many consecutive zero returns: {max_consecutive_zeros}")
        
        # Combine issues and warnings for reporting
        all_messages = issues + warnings
        
        # Validation decision: only issues (not warnings) cause failure
        is_valid = len(issues) == 0
        
        # Log context for debugging
        if warnings and symbol:
            logger.info(f"Validation warnings for {symbol}: {len(warnings)} warnings, {len(issues)} issues")
        
        return is_valid, all_messages
    
    def clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare price data for analysis"""
        if df is None or df.empty:
            return df
        
        cleaned_df = df.copy()
        
        # Sort by date
        if 'Date' in cleaned_df.columns:
            cleaned_df = cleaned_df.sort_values('Date').reset_index(drop=True)
        
        # UPDATED: Use modern pandas method
        price_cols = ['Open', 'High', 'Low', 'Close', 'AdjClose']
        for col in price_cols:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].ffill()  # Updated from fillna(method='ffill')
        
        # Handle volume (fill with median)
        if 'Volume' in cleaned_df.columns:
            median_volume = cleaned_df['Volume'].median()
            cleaned_df['Volume'] = cleaned_df['Volume'].fillna(median_volume)
        
        # Remove rows with still missing critical data
        critical_cols = ['Close']
        cleaned_df = cleaned_df.dropna(subset=critical_cols)
        
        return cleaned_df
    
    def validate_portfolio_data(self, portfolio_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Validate entire portfolio dataset with symbol-aware validation
        
        Args:
            portfolio_data: Dictionary mapping symbol to DataFrame
            
        Returns:
            Validation report with market index considerations
        """
        validation_report = {
            'valid_symbols': [],
            'invalid_symbols': [],
            'issues': {},
            'market_indices_detected': [],
            'summary': {}
        }
        
        for symbol, df in portfolio_data.items():
            is_valid, issues = self.validate_price_data(df, symbol=symbol)
            
            # Track market indices separately
            if symbol.upper() in self.market_indices:
                validation_report['market_indices_detected'].append(symbol)
            
            if is_valid:
                validation_report['valid_symbols'].append(symbol)
            else:
                validation_report['invalid_symbols'].append(symbol)
                validation_report['issues'][symbol] = issues
        
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
    
    def is_market_index(self, symbol: str) -> bool:
        """Check if symbol is a known market index"""
        return symbol.upper() in self.market_indices if symbol else False
    
    def get_validation_summary(self, symbol: str = None) -> Dict[str, Any]:
        """Get validation parameters and thresholds"""
        return {
            'min_data_points': self.min_data_points,
            'max_missing_ratio': self.max_missing_ratio,
            'max_extreme_return': self.max_extreme_return,
            'market_index_support': True,
            'known_market_indices': len(self.market_indices),
            'symbol_is_market_index': self.is_market_index(symbol) if symbol else False,
            'special_handling': {
                'market_indices': 'High kurtosis warnings do not fail validation',
                'individual_stocks': 'Standard validation rules apply'
            }
        }
    
