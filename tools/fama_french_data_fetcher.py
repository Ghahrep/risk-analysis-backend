"""
Fama-French Factor Data Fetcher
================================
Downloads actual Fama-French factor data from Ken French's data library
"""

import pandas as pd
import numpy as np
import requests
import zipfile
import io
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class FamaFrenchDataFetcher:
    """Fetch actual Fama-French factors from Ken French's data library"""
    
    BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
    
    FACTOR_FILES = {
        "3factor": "F-F_Research_Data_Factors_daily_CSV.zip",
        "5factor": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
        "momentum": "F-F_Momentum_Factor_daily_CSV.zip"
    }
    
    def __init__(self):
        self.cache = {}
        self.cache_timestamp = {}
        self.cache_expiry = timedelta(days=1)  # Cache for 1 day
    
    def fetch_factors(self, model_type: str = "3factor", 
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch Fama-French factors from data library
        
        Args:
            model_type: "3factor" or "5factor"
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with daily factor returns (decimals, not percentages)
        """
        try:
            # Check cache
            cache_key = f"{model_type}_{start_date}_{end_date}"
            if cache_key in self.cache:
                if datetime.now() - self.cache_timestamp[cache_key] < self.cache_expiry:
                    logger.info(f"Using cached Fama-French {model_type} data")
                    return self.cache[cache_key]
            
            # Download and parse
            url = f"{self.BASE_URL}/{self.FACTOR_FILES[model_type]}"
            logger.info(f"Downloading Fama-French {model_type} data from {url}")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Extract CSV from ZIP
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                csv_filename = zf.namelist()[0]
                with zf.open(csv_filename) as f:
                    # Ken French files have header rows we need to skip
                    # Read all lines to find where data starts
                    lines = f.read().decode('utf-8').split('\n')
                    
                    # Find start of data (after blank line following header)
                    data_start = 0
                    for i, line in enumerate(lines):
                        if line.strip() == '' and i > 0:
                            data_start = i + 1
                            break
                    
                    # Parse data
                    df = pd.read_csv(
                        io.StringIO('\n'.join(lines[data_start:])),
                        skipinitialspace=True
                    )
            
            # Clean and process
            df = self._process_fama_french_data(df, model_type)
            
            # Filter by date range
            if start_date:
                df = df[df.index >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df.index <= pd.Timestamp(end_date)]
            
            # Cache result
            self.cache[cache_key] = df
            self.cache_timestamp[cache_key] = datetime.now()
            
            logger.info(f"Successfully fetched {len(df)} days of Fama-French data")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch Fama-French data: {e}")
            raise
    
    def _process_fama_french_data(self, df: pd.DataFrame, model_type: str) -> pd.DataFrame:
        """Process raw Fama-French CSV data"""
        # First column is date (YYYYMMDD format)
        date_col = df.columns[0]
        df['date'] = pd.to_datetime(df[date_col], format='%Y%m%d', errors='coerce')
        
        # Remove rows with invalid dates (footer text)
        df = df.dropna(subset=['date'])
        df.set_index('date', inplace=True)
        
        # Select and rename factor columns based on model type
        if model_type == "3factor":
            # Mkt-RF, SMB, HML, RF
            factor_cols = {
                'Mkt-RF': 'Market',
                'SMB': 'SMB', 
                'HML': 'HML',
                'RF': 'RF'
            }
        else:  # 5factor
            # Mkt-RF, SMB, HML, RMW, CMA, RF
            factor_cols = {
                'Mkt-RF': 'Market',
                'SMB': 'SMB',
                'HML': 'HML', 
                'RMW': 'RMW',
                'CMA': 'CMA',
                'RF': 'RF'
            }
        
        # Select relevant columns
        available_cols = [col for col in factor_cols.keys() if col in df.columns]
        df_factors = df[available_cols].copy()
        
        # Rename columns
        df_factors.columns = [factor_cols[col] for col in df_factors.columns]
        
        # Convert from percentages to decimals
        for col in df_factors.columns:
            df_factors[col] = pd.to_numeric(df_factors[col], errors='coerce') / 100.0
        
        # Remove any remaining invalid data
        df_factors = df_factors.dropna()
        
        return df_factors
    
    def get_factor_period(self, symbols: List[str], period: str = "1year") -> pd.DataFrame:
        """
        Get factors aligned with stock data period
        
        Args:
            symbols: Stock symbols (not used for factors, but for date alignment)
            period: Period string like "1year", "2years", etc.
            
        Returns:
            DataFrame with factor returns for the period
        """
        # Calculate date range
        end_date = datetime.now()
        days_mapping = {
            '1month': 30, '3months': 90, '6months': 180,
            '1year': 365, '2years': 730, '5years': 1825
        }
        start_date = end_date - timedelta(days=days_mapping.get(period, 365))
        
        return self.fetch_factors("3factor", start_date, end_date)