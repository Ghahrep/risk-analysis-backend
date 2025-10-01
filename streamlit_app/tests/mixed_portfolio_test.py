"""
Test mixed portfolios (stocks, ETFs, bonds)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.api_client import get_risk_api_client
import time

TEST_PORTFOLIOS = {
    "stocks_only": {
        "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "weights": [0.25, 0.25, 0.25, 0.25],
        "description": "100% Individual Stocks"
    },
    "etfs_only": {
        "symbols": ["SPY", "QQQ", "IWM", "VTI"],
        "weights": [0.25, 0.25, 0.25, 0.25],
        "description": "100% ETFs"
    },
    "stocks_and_etfs": {
        "symbols": ["AAPL", "MSFT", "SPY", "QQQ"],
        "weights": [0.25, 0.25, 0.25, 0.25],
        "description": "50% Stocks, 50% ETFs"
    },
    "diversified_etf": {
        "symbols": ["SPY", "TLT", "GLD", "VNQ"],
        "weights": [0.40, 0.30, 0.15, 0.15],
        "description": "Multi-Asset ETF Portfolio"
    },
    "bond_heavy": {
        "symbols": ["TLT", "AGG", "BND", "SPY"],
        "weights": [0.30, 0.30, 0.20, 0.20],
        "description": "Bond-Heavy Portfolio"
    }
}

def test_mixed_portfolios():
    api_client = get_risk_api_client()
    
    print("\n" + "="*70)
    print("MIXED ASSET CLASS TESTING")
    print("="*70)
    
    results = {}
    
    for portfolio_name, portfolio in TEST_PORTFOLIOS.items():
        print(f"\nTesting: {portfolio['description']}")
        print(f"Symbols: {', '.join(portfolio['symbols'])}")
        print("-" * 70)
        
        start = time.time()
        
        try:
            result = api_client.analyze_risk(
                portfolio['symbols'],
                portfolio['weights'],
                "1year"
            )
            
            elapsed = time.time() - start
            
            if result and not result.get('error'):
                metrics = result.get('metrics', {})
                print(f"Status: SUCCESS ({elapsed:.2f}s)")
                print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
                print(f"Volatility: {metrics.get('annualized_volatility', 0):.2%}")
                results[portfolio_name] = "PASS"
            else:
                print(f"Status: FAILED - {result.get('error', 'Unknown error')}")
                results[portfolio_name] = "FAIL"
                
        except Exception as e:
            print(f"Status: EXCEPTION - {str(e)}")
            results[portfolio_name] = "ERROR"
        
        time.sleep(1)  # Rate limiting
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, status in results.items():
        symbol = "✓" if status == "PASS" else "✗"
        print(f"{symbol} {name}: {status}")
    
    pass_rate = sum(1 for s in results.values() if s == "PASS") / len(results)
    print(f"\nPass Rate: {pass_rate:.0%}")
    
    return results

if __name__ == "__main__":
    test_mixed_portfolios()