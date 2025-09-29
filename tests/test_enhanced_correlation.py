# tests/test_enhanced_correlation.py
"""
Enhanced Correlation Analysis Test Suite
=======================================

Comprehensive test suite for the new enhanced correlation capabilities including:
- Rolling correlation analysis
- Regime-conditional correlations
- Hierarchical clustering
- Network analysis
- Comprehensive correlation analysis
"""

import requests
import time
from datetime import datetime

BASE_URL = "http://localhost:8001"

def test_enhanced_correlation_endpoints():
    """Test all enhanced correlation analysis endpoints"""
    
    print("🚀 Enhanced Correlation Analysis Test Suite")
    print("=" * 60)
    print(f"Testing API at: {BASE_URL}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test portfolio
    test_portfolio = {
        "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
        "period": "1year",
        "use_real_data": True
    }
    
    results = []
    total_start = time.time()
    
    # Test cases
    tests = [
        {
            "name": "Enhanced Correlations Health",
            "url": f"{BASE_URL}/enhanced-correlations/health",
            "method": "GET",
            "payload": {},
            "expected_keys": ["status", "enhanced_correlations", "capabilities"]
        },
        {
            "name": "Rolling Correlations",
            "url": f"{BASE_URL}/rolling-correlations",
            "method": "POST",
            "payload": {**test_portfolio, "window_days": 60},
            "expected_keys": ["status", "rolling_correlations", "metadata"]
        },
        {
            "name": "Regime Correlations",
            "url": f"{BASE_URL}/regime-correlations",
            "method": "POST",
            "payload": test_portfolio,
            "expected_keys": ["status", "regime_correlations", "metadata"]
        },
        {
            "name": "Correlation Clustering",
            "url": f"{BASE_URL}/correlation-clustering",
            "method": "POST",
            "payload": {**test_portfolio, "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META"]},
            "expected_keys": ["status", "correlation_clustering", "metadata"]
        },
        {
            "name": "Correlation Network",
            "url": f"{BASE_URL}/correlation-network",
            "method": "POST",
            "payload": {**test_portfolio, "correlation_threshold": 0.5},
            "expected_keys": ["status", "correlation_network", "metadata"]
        },
        {
            "name": "Comprehensive Correlation",
            "url": f"{BASE_URL}/comprehensive-correlation",
            "method": "POST",
            "payload": {**test_portfolio, "window_days": 60, "correlation_threshold": 0.5},
            "expected_keys": ["status", "comprehensive_correlation", "metadata"]
        }
    ]
    
    for test in tests:
        print(f"\n🧪 Testing {test['name']}...")
        start_time = time.time()
        
        try:
            if test["method"] == "GET":
                response = requests.get(test["url"], timeout=30)
            else:
                response = requests.post(test["url"], json=test["payload"], timeout=30)
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("status") == "success":
                    # Check for expected keys
                    missing_keys = []
                    for key in test["expected_keys"]:
                        if key not in data:
                            missing_keys.append(key)
                    
                    if not missing_keys:
                        print(f"✅ {test['name']} - SUCCESS ({duration:.2f}s)")
                        
                        # Extract some interesting metrics for display
                        if test['name'] == "Rolling Correlations" and 'rolling_correlations' in data:
                            rolling_data = data['rolling_correlations']
                            if 'stability_metrics' in rolling_data:
                                stability = rolling_data['stability_metrics'].get('stability_score', 0)
                                print(f"   📊 Correlation Stability Score: {stability:.3f}")
                        
                        elif test['name'] == "Regime Correlations" and 'regime_correlations' in data:
                            regime_data = data['regime_correlations']
                            if 'regime_sensitivity' in regime_data:
                                multiplier = regime_data['regime_sensitivity'].get('crisis_correlation_multiplier', 1)
                                print(f"   📊 Crisis Correlation Multiplier: {multiplier:.2f}x")
                        
                        elif test['name'] == "Correlation Clustering" and 'correlation_clustering' in data:
                            cluster_data = data['correlation_clustering']
                            clusters = cluster_data.get('optimal_clusters', 0)
                            print(f"   📊 Optimal Clusters: {clusters}")
                        
                        elif test['name'] == "Correlation Network" and 'correlation_network' in data:
                            network_data = data['correlation_network']
                            if 'network_health' in network_data:
                                density = network_data['network_health'].get('network_density', 0)
                                print(f"   📊 Network Density: {density:.3f}")
                        
                        elif test['name'] == "Comprehensive Correlation":
                            comprehensive_data = data['comprehensive_correlation']
                            insights = comprehensive_data.get('synthesized_insights', [])
                            print(f"   📊 Synthesized Insights: {len(insights)} generated")
                        
                        results.append({"name": test["name"], "status": "PASS", "duration": duration})
                    else:
                        print(f"⚠️  {test['name']} - Missing keys: {missing_keys}")
                        results.append({"name": test["name"], "status": "FAIL", "duration": duration})
                else:
                    error_msg = data.get('message', 'Unknown error')
                    print(f"❌ {test['name']} - API returned error: {error_msg}")
                    results.append({"name": test["name"], "status": "FAIL", "duration": duration})
            else:
                print(f"❌ {test['name']} - HTTP {response.status_code}")
                results.append({"name": test["name"], "status": "FAIL", "duration": duration})
                
        except requests.exceptions.Timeout:
            print(f"⏰ {test['name']} - TIMEOUT (>30s)")
            results.append({"name": test["name"], "status": "TIMEOUT", "duration": 30.0})
        except Exception as e:
            print(f"💥 {test['name']} - ERROR: {str(e)}")
            results.append({"name": test["name"], "status": "ERROR", "duration": 0.0})
    
    total_duration = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Enhanced Correlation Test Summary")
    print("=" * 60)
    
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] in ["FAIL", "ERROR", "TIMEOUT"])
    
    print(f"✅ Passed: {passed}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")
    print(f"⏱️  Total Time: {total_duration:.2f}s")
    
    if failed == 0:
        print("\n🎉 ALL ENHANCED CORRELATION TESTS PASSED!")
        print("Enhanced correlation analysis capabilities are working correctly.")
        print("\n📈 Available Enhanced Capabilities:")
        print("   • Rolling correlation analysis with stability metrics")
        print("   • Market regime-conditional correlation matrices")
        print("   • Hierarchical clustering with dendrogram analysis")
        print("   • Network analysis with centrality measures")
        print("   • Comprehensive multi-method correlation analysis")
        
    else:
        print(f"\n⚠️  {failed} tests failed. Check your enhanced correlation implementation.")
        failed_tests = [r["name"] for r in results if r["status"] in ["FAIL", "ERROR", "TIMEOUT"]]
        print(f"Failed tests: {', '.join(failed_tests)}")
    
    return results

if __name__ == "__main__":
    test_enhanced_correlation_endpoints()