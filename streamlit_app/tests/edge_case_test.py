"""
Test edge cases and error handling
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.api_client import get_risk_api_client

def test_edge_cases():
    api_client = get_risk_api_client()
    
    print("\n" + "="*70)
    print("EDGE CASE TESTING")
    print("="*70)
    
    tests = [
        {
            "name": "Invalid Symbol",
            "symbols": ["AAPL", "INVALID123", "MSFT"],
            "weights": [0.33, 0.34, 0.33],
            "expected": "Should handle gracefully"
        },
        {
            "name": "Single Symbol",
            "symbols": ["AAPL"],
            "weights": [1.0],
            "expected": "Should work or show appropriate message"
        },
        {
            "name": "Unequal Weights",
            "symbols": ["AAPL", "MSFT"],
            "weights": [0.9, 0.1],
            "expected": "Should accept any valid weights"
        },
        {
            "name": "Zero Weight",
            "symbols": ["AAPL", "MSFT", "GOOGL"],
            "weights": [0.5, 0.5, 0.0],
            "expected": "Should handle or normalize"
        },
        {
            "name": "Many Symbols (50+)",
            "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"] * 10,
            "weights": [0.02] * 50,
            "expected": "May timeout or fail gracefully"
        }
    ]
    
    for test in tests:
        print(f"\nTest: {test['name']}")
        print(f"Expected: {test['expected']}")
        print("-" * 70)
        
        try:
            result = api_client.analyze_risk(
                test['symbols'],
                test['weights'],
                "1year"
            )
            
            if result and not result.get('error'):
                print("Result: SUCCESS")
            elif result and result.get('error'):
                print(f"Result: ERROR - {result.get('error')}")
            else:
                print("Result: FAILED - No response")
                
        except Exception as e:
            print(f"Result: EXCEPTION - {str(e)}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    test_edge_cases()