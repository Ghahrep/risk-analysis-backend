"""
Performance testing with real API calls
"""

import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.api_client import get_risk_api_client

# Test configurations
TEST_PORTFOLIOS = {
    "small": {
        "symbols": ["AAPL", "MSFT", "GOOGL"],
        "weights": [0.33, 0.33, 0.34]
    },
    "medium": {
        "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"],
        "weights": [0.1] * 10
    },
    "large": {
        "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ",
                   "WMT", "PG", "UNH", "HD", "BAC", "XOM", "CVX", "ABBV", "PFE", "KO"],
        "weights": [0.05] * 20
    }
}

def test_endpoint(api_client, endpoint_name, symbols, weights, period="1year"):
    """Test a specific endpoint"""
    start = time.time()
    try:
        if endpoint_name == "analyze_risk":
            result = api_client.analyze_risk(symbols, weights, period)
        elif endpoint_name == "optimize_portfolio":
            result = api_client.optimize_portfolio(symbols, "max_sharpe", period)
        elif endpoint_name == "stress_test":
            result = api_client.stress_test(symbols, weights)
        elif endpoint_name == "correlation_analysis":
            result = api_client.correlation_analysis(symbols, period)
        else:
            result = None
        
        end = time.time()
        elapsed = end - start
        success = result is not None and not result.get('error')
        
        return {
            "elapsed": elapsed,
            "success": success,
            "result": result
        }
    except Exception as e:
        end = time.time()
        return {
            "elapsed": end - start,
            "success": False,
            "error": str(e)
        }

def run_performance_tests():
    """Run comprehensive performance tests"""
    api_client = get_risk_api_client()
    
    endpoints = ["analyze_risk", "optimize_portfolio", "stress_test", "correlation_analysis"]
    
    print("\n" + "="*70)
    print("REAL API PERFORMANCE TEST")
    print("="*70)
    
    all_results = {}
    
    for size, portfolio in TEST_PORTFOLIOS.items():
        print(f"\n{size.upper()} PORTFOLIO ({len(portfolio['symbols'])} symbols)")
        print("-" * 70)
        
        size_results = {}
        
        for endpoint in endpoints:
            print(f"  Testing {endpoint}...", end=" ", flush=True)
            
            # Run test
            result = test_endpoint(
                api_client,
                endpoint,
                portfolio['symbols'],
                portfolio['weights']
            )
            
            size_results[endpoint] = result
            
            if result['success']:
                print(f"✓ {result['elapsed']:.2f}s")
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"✗ FAILED - {error_msg}")
            
            # Rate limiting: wait 1 second between calls
            time.sleep(1)
        
        all_results[size] = size_results
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for size, endpoints_results in all_results.items():
        total_time = sum(r['elapsed'] for r in endpoints_results.values())
        success_count = sum(1 for r in endpoints_results.values() if r['success'])
        total_count = len(endpoints_results)
        
        print(f"\n{size.upper()}:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Success rate: {success_count}/{total_count}")
        print(f"  Status: {'PASS' if success_count == total_count else 'FAIL'}")
    
    # API call count
    print("\n" + "="*70)
    print("API USAGE")
    print("="*70)
    total_calls = len(TEST_PORTFOLIOS) * len(endpoints)
    print(f"Total API calls made: {total_calls}")
    print(f"Estimated daily limit usage: {(total_calls/250)*100:.1f}%")
    
    return all_results

if __name__ == "__main__":
    print("WARNING: This will make real API calls and count against your daily limit.")
    print("Estimated calls: ~12 (4 endpoints × 3 portfolio sizes)")
    response = input("Continue? (yes/no): ")
    
    if response.lower() == 'yes':
        results = run_performance_tests()
    else:
        print("Test cancelled.")