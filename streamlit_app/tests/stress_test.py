"""
Stress test API rate limits and concurrent analysis
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.api_client import get_risk_api_client
import time

def test_rapid_requests():
    """Test how the API handles rapid successive requests"""
    api_client = get_risk_api_client()
    
    print("\n" + "="*70)
    print("RAPID REQUEST STRESS TEST")
    print("="*70)
    
    portfolio = {
        "symbols": ["AAPL", "MSFT", "GOOGL"],
        "weights": [0.33, 0.33, 0.34]
    }
    
    num_requests = 5
    results = []
    
    print(f"\nSending {num_requests} requests with NO delay...")
    print("-" * 70)
    
    for i in range(num_requests):
        start = time.time()
        try:
            result = api_client.analyze_risk(
                portfolio['symbols'],
                portfolio['weights'],
                "1year"
            )
            elapsed = time.time() - start
            
            success = result and not result.get('error')
            results.append({
                'request': i+1,
                'success': success,
                'time': elapsed
            })
            
            status = "✓" if success else "✗"
            print(f"Request {i+1}: {status} ({elapsed:.2f}s)")
            
        except Exception as e:
            results.append({
                'request': i+1,
                'success': False,
                'time': time.time() - start,
                'error': str(e)
            })
            print(f"Request {i+1}: ✗ ERROR - {str(e)}")
    
    # Analysis
    success_count = sum(1 for r in results if r['success'])
    avg_time = sum(r['time'] for r in results) / len(results)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Success Rate: {success_count}/{num_requests}")
    print(f"Average Time: {avg_time:.2f}s")
    print(f"Status: {'PASS' if success_count == num_requests else 'FAIL'}")
    
    return results

if __name__ == "__main__":
    print("WARNING: This will make 5 rapid API calls")
    response = input("Continue? (yes/no): ")
    
    if response.lower() == 'yes':
        test_rapid_requests()
    else:
        print("Test cancelled")