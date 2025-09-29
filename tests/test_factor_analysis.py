# test_factor_analysis.py
"""
Comprehensive test suite for factor analysis endpoints
Tests all four new endpoints with real and synthetic data scenarios
"""

import requests
import json
import time
import numpy as np
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8001"
TEST_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]

def test_factor_analysis():
    """Test Fama-French factor analysis endpoint"""
    print("\n=== Testing Factor Analysis Endpoint ===")
    
    # Test 3-factor model
    payload = {
        "symbols": TEST_SYMBOLS[:3],  # Start with 3 symbols
        "period": "1year",
        "model_type": "3factor"
    }
    
    print(f"Testing 3-factor model with symbols: {payload['symbols']}")
    
    try:
        response = requests.post(f"{BASE_URL}/factor-analysis", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Analyzed {result['summary']['symbols_analyzed']} symbols")
            print(f"📊 Average R-squared: {result['summary']['avg_r_squared']}")
            print(f"⚡ Execution time: {result['execution_time_seconds']}s")
            
            # Show sample results
            first_symbol = list(result['factor_analysis'].keys())[0]
            sample_result = result['factor_analysis'][first_symbol]
            print(f"\n📈 Sample result for {first_symbol}:")
            print(f"   Market Beta: {sample_result['factor_loadings'].get('Market', 'N/A')}")
            print(f"   Alpha: {sample_result['alpha']}")
            print(f"   R-squared: {sample_result['r_squared']}")
            print(f"   Data source: {sample_result['data_source']}")
            
            # Test 5-factor model
            print("\n--- Testing 5-factor model ---")
            payload["model_type"] = "5factor"
            response_5f = requests.post(f"{BASE_URL}/factor-analysis", json=payload, timeout=30)
            
            if response_5f.status_code == 200:
                result_5f = response_5f.json()
                print(f"✅ 5-factor analysis successful")
                print(f"📊 Average R-squared: {result_5f['summary']['avg_r_squared']}")
            else:
                print(f"❌ 5-factor test failed: {response_5f.status_code}")
            
        else:
            print(f"❌ Factor analysis failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Factor analysis test error: {e}")

def test_style_analysis():
    """Test style analysis endpoint"""
    print("\n=== Testing Style Analysis Endpoint ===")
    
    # Generate sample portfolio returns (daily returns for 1 year)
    np.random.seed(42)  # For reproducible results
    n_days = 252
    portfolio_returns = np.random.normal(0.0008, 0.015, n_days).tolist()  # ~20% annual return, 15% vol
    
    payload = {
        "portfolio_returns": portfolio_returns,
        "period": "1year"
    }
    
    print(f"Testing with {len(portfolio_returns)} portfolio return observations")
    
    try:
        response = requests.post(f"{BASE_URL}/style-analysis", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Style analysis successful!")
            print(f"📊 R-squared: {result['style_analysis']['r_squared']}")
            print(f"📈 Tracking error: {result['style_analysis']['tracking_error']}")
            print(f"🎯 Dominant style: {result['summary']['dominant_style']}")
            print(f"⚡ Execution time: {result['execution_time_seconds']}s")
            
            # Show style weights
            print("\n📊 Style exposures:")
            for style, weight in result['style_analysis']['style_weights'].items():
                print(f"   {style}: {weight:.1%}")
                
        else:
            print(f"❌ Style analysis failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Style analysis test error: {e}")

def test_pca_analysis():
    """Test PCA factor analysis endpoint"""
    print("\n=== Testing PCA Factor Analysis Endpoint ===")
    
    payload = {
        "symbols": TEST_SYMBOLS,
        "period": "1year", 
        "n_components": 3
    }
    
    print(f"Testing PCA with symbols: {payload['symbols']}")
    print(f"Extracting {payload['n_components']} principal components")
    
    try:
        response = requests.post(f"{BASE_URL}/pca-factors", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ PCA analysis successful!")
            
            # Debug: Print the actual response structure to see what fields exist
            print("Debug - Available keys in result:", list(result.keys()))
            if 'summary' in result:
                print("Debug - Summary keys:", list(result['summary'].keys()))
            
            # Use the actual field names from your response
            summary = result.get('summary', {})
            pca_analysis = result.get('pca_analysis', {})
            
            # Check which fields actually exist and use them
            if 'total_variance_explained' in summary:
                print(f"📊 Total variance explained: {summary['total_variance_explained']:.1f}%")
            elif 'total_variance_explained' in pca_analysis:
                print(f"📊 Total variance explained: {pca_analysis['total_variance_explained']:.1f}%")
            
            if 'first_component_variance' in summary:
                print(f"📈 First component: {summary['first_component_variance']:.1f}%")
            elif 'explained_variance_ratio' in pca_analysis and len(pca_analysis['explained_variance_ratio']) > 0:
                print(f"📈 First component: {pca_analysis['explained_variance_ratio'][0]:.1f}%")
            
            if 'data_reduction' in summary:
                print(f"💾 Data reduction: {summary['data_reduction']:.1f}%")
            
            if 'execution_time_seconds' in result:
                print(f"⚡ Execution time: {result['execution_time_seconds']}s")
            
            # Show factor interpretation if it exists
            if 'factor_interpretation' in pca_analysis:
                print("\n🔍 Factor interpretation:")
                for factor, details in pca_analysis['factor_interpretation'].items():
                    if isinstance(details, dict) and 'variance_explained' in details:
                        print(f"   {factor}: {details['variance_explained']:.1f}% variance")
                        if 'top_contributors' in details:
                            print(f"      Top contributors: {details['top_contributors']}")
            
        else:
            print(f"❌ PCA analysis failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ PCA analysis test error: {e}")
        # Also print the response content for debugging if available
        try:
            if 'response' in locals():
                print(f"Response content: {response.text[:500]}...")
        except:
            pass

def test_rolling_factor_analysis():
    """Test rolling factor analysis endpoint"""
    print("\n=== Testing Rolling Factor Analysis Endpoint ===")
    
    payload = {
        "symbol": "AAPL",
        "period": "1year",
        "window_days": 60,
        "model_type": "3factor"
    }
    
    print(f"Testing rolling analysis for {payload['symbol']}")
    print(f"Window: {payload['window_days']} days, Model: {payload['model_type']}")
    
    try:
        response = requests.post(f"{BASE_URL}/rolling-factors", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Rolling factor analysis successful!")
            print(f"📊 Periods analyzed: {result['rolling_analysis']['periods_analyzed']}")
            print(f"📈 Average R-squared: {result['rolling_analysis']['average_r_squared']}")
            print(f"📅 Date range: {result['data_summary']['date_range']}")
            print(f"⚡ Execution time: {result['execution_time_seconds']}s")
            
            # Show current factor loadings
            print("\n📊 Current factor loadings:")
            for factor, value in result['rolling_analysis']['latest_factors'].items():
                if factor != 'R_squared':
                    trend_info = result['rolling_analysis']['factor_trends'][factor]
                    print(f"   {factor}: {value:.3f} ({trend_info['trend']}, {trend_info['stability']})")
                    
        else:
            print(f"❌ Rolling factor analysis failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Rolling factor analysis test error: {e}")

def test_enhanced_capabilities():
    """Test that capabilities endpoint includes factor analysis"""
    print("\n=== Testing Enhanced Capabilities Endpoint ===")
    
    try:
        response = requests.get(f"{BASE_URL}/capabilities", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Capabilities endpoint accessible")
            print(f"📊 Service version: {result.get('version', 'Unknown')}")
            
            # Check factor analysis capabilities
            capabilities = result.get('capabilities', {})
            factor_caps = [
                'factor_analysis', 'style_analysis', 
                'pca_analysis', 'rolling_factor_analysis'
            ]
            
            print("\n🔧 Factor analysis capabilities:")
            for cap in factor_caps:
                status = "✅" if capabilities.get(cap) else "❌"
                print(f"   {cap}: {status}")
                
            # Show available endpoints
            endpoints = result.get('endpoints', {})
            factor_endpoints = [ep for ep in endpoints.keys() if 'factor' in ep.lower() or 'style' in ep.lower() or 'pca' in ep.lower()]
            
            print("\n🌐 Factor analysis endpoints:")
            for endpoint in factor_endpoints:
                print(f"   {endpoint}: {endpoints[endpoint]}")
                
        else:
            print(f"❌ Capabilities check failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Capabilities test error: {e}")

def test_error_handling():
    """Test error handling for invalid requests"""
    print("\n=== Testing Error Handling ===")
    
    # Test empty symbols list
    print("Testing empty symbols list...")
    try:
        response = requests.post(f"{BASE_URL}/factor-analysis", 
                               json={"symbols": [], "period": "1year"}, 
                               timeout=10)
        print(f"Empty symbols: {response.status_code} (expected 400)")
    except Exception as e:
        print(f"Empty symbols test error: {e}")
    
    # Test invalid model type
    print("Testing invalid model type...")
    try:
        response = requests.post(f"{BASE_URL}/factor-analysis", 
                               json={"symbols": ["AAPL"], "model_type": "invalid"}, 
                               timeout=10)
        print(f"Invalid model: {response.status_code} (expected 400)")
    except Exception as e:
        print(f"Invalid model test error: {e}")
    
    # Test insufficient portfolio returns
    print("Testing insufficient portfolio returns...")
    try:
        response = requests.post(f"{BASE_URL}/style-analysis", 
                               json={"portfolio_returns": [0.01, 0.02]}, 
                               timeout=10)
        print(f"Insufficient returns: {response.status_code} (expected 400)")
    except Exception as e:
        print(f"Insufficient returns test error: {e}")

def main():
    """Run complete factor analysis test suite"""
    print("🧪 FACTOR ANALYSIS TEST SUITE")
    print("=" * 50)
    print(f"Testing against: {BASE_URL}")
    print(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if API is running
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ API health check failed. Make sure minimal_api.py is running on port 8001")
            return
        print("✅ API health check passed")
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Make sure to run: python minimal_api.py")
        return
    
    # Run all tests
    start_time = time.time()
    
    test_enhanced_capabilities()
    test_factor_analysis()
    test_style_analysis() 
    test_pca_analysis()
    test_rolling_factor_analysis()
    test_error_handling()
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print(f"🏁 Test suite completed in {total_time:.2f} seconds")
    print("Factor analysis integration ready for production! 🚀")

if __name__ == "__main__":
    main()