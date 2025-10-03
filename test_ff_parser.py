import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from tools.factor_analysis_tools import FamaFrenchDataFetcher

# Test the parser
fetcher = FamaFrenchDataFetcher()

try:
    factors = fetcher.fetch_factors("3factor")
    print(f"SUCCESS: Fetched {len(factors)} days of data")
    print(f"Columns: {factors.columns.tolist()}")
    
    # Check if Market column exists
    if 'Market' in factors.columns:
        print("✓ Market column found")
    else:
        print("✗ Market column MISSING")
    
    print(f"\nFirst 5 rows:\n{factors.head()}")
    print(f"\nLast 5 rows:\n{factors.tail()}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()