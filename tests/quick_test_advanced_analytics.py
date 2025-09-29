# quick_test_advanced_analytics.py
"""
Quick Test Runner for Advanced Analytics
=======================================

Simple test script to validate advanced analytics endpoints are working.
Run this after adding the advanced analytics tools to your minimal_api.py.

Usage:
    python quick_test_advanced_analytics.py
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8001"

def test_endpoint(endpoint_name, url, payload, expected_keys):
    """Test a single endpoint with payload and validate response"""
    print(f"\n🧪 Testing {endpoint_name}...")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=30)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("status") == "success":
                # Check for expected keys
                missing_keys = []
                for key in expected_keys:
                    if key not in data:
                        missing_keys.append(key)
                
                if not missing_keys:
                    print(f"✅ {endpoint_name} - SUCCESS ({duration:.2f}s)")
                    return True, data, duration
                else:
                    print(f"⚠️  {endpoint_name} - Missing keys: {missing_keys}")
                    return False, data, duration
            else:
                print(f"❌ {endpoint_name} - API returned error: {data.get('message', 'Unknown error')}")
                return False, data, duration
        else:
            print(f"❌ {endpoint_name} - HTTP {response.status_code}")
            return False, None, duration
            
    except requests.exceptions.Timeout:
        print(f"⏰ {endpoint_name} - TIMEOUT (>30s)")
        return False, None, 30.0
    except Exception as e:
        print(f"💥 {endpoint_name} - ERROR: {str(e)}")
        return False, None, 0.0

def main():
    """Run quick tests for all advanced analytics endpoints"""
    
    print("🚀 Quick Advanced Analytics Test Suite")
    print("=" * 50)
    print(f"Testing API at: {BASE_URL}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test portfolio
    test_portfolio = {
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "weights": [0.4, 0.35, 0.25],
        "period": "1year",
        "use_real_data": True
    }
    
    # Test cases
    tests = [
        {
            "name": "Health Check",
            "url": f"{BASE_URL}/advanced-analytics/health",
            "payload": {},
            "method": "GET",
            "expected_keys": ["status", "capabilities"]
        },
        {
            "name": "Risk Attribution",
            "url": f"{BASE_URL}/risk-attribution", 
            "payload": {**test_portfolio, "factor_model": "fama_french_3"},
            "method": "POST",
            "expected_keys": ["status", "risk_attribution", "metadata"]
        },
        {
            "name": "Performance Attribution",
            "url": f"{BASE_URL}/performance-attribution",
            "payload": {**test_portfolio, "benchmark": "SPY", "factor_model": "fama_french_3"},
            "method": "POST", 
            "expected_keys": ["status", "performance_attribution", "metadata"]
        },
        {
            "name": "Advanced Analytics",
            "url": f"{BASE_URL}/advanced-analytics",
            "payload": test_portfolio,
            "method": "POST",
            "expected_keys": ["status", "advanced_analytics", "metadata"]
        },
        {
            "name": "Correlation Analysis", 
            "url": f"{BASE_URL}/correlation-analysis",
            "payload": {
                "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
                "period": "1year",
                "use_real_data": True
            },
            "method": "POST",
            "expected_keys": ["status", "correlation_analysis", "metadata"]
        }
    ]
    
    results = []
    total_start = time.time()
    
    for test in tests:
        if test["method"] == "GET":
            try:
                start_time = time.time()
                response = requests.get(test["url"], timeout=30)
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    success = all(key in data for key in test["expected_keys"])
                    status = "PASS" if success else "FAIL"
                    print(f"✅ {test['name']} - {status} ({duration:.2f}s)")
                    results.append({"name": test["name"], "status": status, "duration": duration})
                else:
                    print(f"❌ {test['name']} - HTTP {response.status_code}")
                    results.append({"name": test["name"], "status": "FAIL", "duration": duration})
            except Exception as e:
                print(f"💥 {test['name']} - ERROR: {str(e)}")
                results.append({"name": test["name"], "status": "ERROR", "duration": 0.0})
        else:
            success, data, duration = test_endpoint(
                test["name"], 
                test["url"], 
                test["payload"], 
                test["expected_keys"]
            )
            results.append({
                "name": test["name"], 
                "status": "PASS" if success else "FAIL", 
                "duration": duration,
                "data": data
            })
    
    total_duration = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Quick Test Summary")
    print("=" * 50)
    
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] in ["FAIL", "ERROR"])
    
    print(f"✅ Passed: {passed}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")
    print(f"⏱️  Total Time: {total_duration:.2f}s")
    
    if failed == 0:
        print("\n🎉 ALL QUICK TESTS PASSED!")
        print("Advanced Analytics endpoints are working correctly.")
        
        # Show sample data from risk attribution
        risk_result = next((r for r in results if r["name"] == "Risk Attribution"), None)
        if risk_result and risk_result.get("data"):
            try:
                risk_data = risk_result["data"]["risk_attribution"]
                print(f"\n📈 Sample Risk Attribution Results:")
                print(f"   Total Risk: {risk_data['total_risk_pct']:.2f}%")
                print(f"   Systematic: {risk_data['systematic_risk_pct']:.2f}%")
                print(f"   Idiosyncratic: {risk_data['idiosyncratic_risk_pct']:.2f}%")
                
                if risk_data['factor_contributions']:
                    print(f"   Factor Contributions:")
                    for factor, contrib in risk_data['factor_contributions'].items():
                        print(f"     {factor}: {contrib:.2f}%")
            except:
                pass
                
    else:
        print(f"\n⚠️  {failed} tests failed. Check your implementation.")
        
        # Show which tests failed
        failed_tests = [r["name"] for r in results if r["status"] in ["FAIL", "ERROR"]]
        print(f"Failed tests: {', '.join(failed_tests)}")
    
    return results

if __name__ == "__main__":
    main()