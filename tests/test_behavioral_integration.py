# tests/test_behavioral_integration.py - Complete Testing Suite
"""
Behavioral Analysis Service Integration Tests
============================================

Following Backend Refactoring Handbook - Phase 5 & 6
Comprehensive testing for centralized models and FMP integration
"""

import pytest
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

# Import the main application
from main import app

# Import service components for direct testing
from services.behavioral_service_updated import BehavioralAnalysisService
from models.requests import (
    ConversationMessage, BehavioralAnalysisRequest,
    AnalysisPeriod, BehavioralAnalysisResponse
)

# Create test client
client = TestClient(app)

# =============================================================================
# TEST DATA GENERATION
# =============================================================================

def generate_test_conversation_data() -> List[Dict[str, str]]:
    """Generate consistent test conversation data"""
    return [
        {"role": "user", "content": "I'm really worried about my portfolio losing money in this volatile market"},
        {"role": "assistant", "content": "I understand your concerns about market volatility. Let's analyze your risk factors."},
        {"role": "user", "content": "Everyone is buying tech stocks right now, I don't want to miss out on the gains"},
        {"role": "assistant", "content": "That sounds like FOMO. Let's look at your investment thesis."},
        {"role": "user", "content": "I'm definitely sure that AI stocks will keep going up - it's obvious"},
        {"role": "assistant", "content": "High confidence can sometimes indicate overconfidence bias."},
        {"role": "user", "content": "I bought NVDA at $300, it was such a good price back then"},
        {"role": "user", "content": "I just can't afford to lose any more money on this position"}
    ]

def generate_test_centralized_messages() -> List[ConversationMessage]:
    """Generate test data using centralized models"""
    raw_messages = generate_test_conversation_data()
    return [
        ConversationMessage(role=msg["role"], content=msg["content"])
        for msg in raw_messages
    ]

TEST_SYMBOLS = ["AAPL", "NVDA", "GOOGL", "MSFT", "TSLA"]
TEST_PERIOD = "6month"

# =============================================================================
# PHASE 5: INTEGRATION TESTING (45 minutes)
# =============================================================================

class TestBehavioralServiceDirect:
    """Direct service testing without API layer"""
    
    @pytest.fixture
    def service(self):
        """Initialize service for testing"""
        return BehavioralAnalysisService(
            risk_free_rate=0.02,
            confidence_levels=[0.95, 0.99],
            default_period="1year"
        )
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, service):
        """Test 1: Service initializes correctly"""
        assert service.service_name == "behavioral_analysis_direct"
        assert service.version == "2.1.0"
        assert service.centralized_models_available is True
        
        # Test service status
        status = service.get_service_status()
        assert status['service_name'] == "behavioral_analysis_direct"
        assert status['status'] in ['operational', 'degraded']
        assert 'centralized_models_support' in status['capabilities']
        
        print(f"✓ Service Status: {status['status']}")
        print(f"✓ Centralized Models: {status['capabilities']['centralized_models_support']}")
        print(f"✓ FMP Integration: {status['capabilities']['fmp_integration_available']}")
    
    @pytest.mark.asyncio
    async def test_legacy_message_format_compatibility(self, service):
        """Test 2: Legacy dict format still works"""
        legacy_messages = generate_test_conversation_data()
        
        result = await service.comprehensive_behavioral_analysis(
            conversation_messages=legacy_messages,
            symbols=TEST_SYMBOLS,
            period=TEST_PERIOD,
            use_real_data=False  # Use synthetic for consistent testing
        )
        
        # Test both response types
        if hasattr(result, 'success'):
            # Centralized response model
            assert result.success is True
            assert result.bias_count >= 0
            assert result.overall_risk_score >= 0
            assert isinstance(result.recommendations, list)
            print(f"✓ Legacy Format (Centralized Response): {result.bias_count} biases, {result.overall_risk_score:.1f} risk")
        else:
            # Legacy response
            assert result.success is True
            assert result.bias_count >= 0
            print(f"✓ Legacy Format (Legacy Response): {result.bias_count} biases")
    
    @pytest.mark.asyncio 
    async def test_centralized_models_format(self, service):
        """Test 3: Centralized models format works"""
        centralized_messages = generate_test_centralized_messages()
        
        result = await service.comprehensive_behavioral_analysis(
            conversation_messages=centralized_messages,
            symbols=TEST_SYMBOLS,
            period=AnalysisPeriod.SIX_MONTHS,
            use_real_data=False
        )
        
        # Should work with both response types
        success = getattr(result, 'success', False)
        bias_count = getattr(result, 'bias_count', 0)
        risk_score = getattr(result, 'overall_risk_score', 0)
        
        assert success is True
        assert bias_count >= 0
        assert risk_score >= 0
        
        print(f"✓ Centralized Models: {bias_count} biases, {risk_score:.1f} risk")
    
    @pytest.mark.asyncio
    async def test_real_fmp_integration(self, service):
        """Test 4: Real FMP data integration (if available)"""
        if not service.fmp_available:
            pytest.skip("FMP integration not available")
        
        result = await service.comprehensive_behavioral_analysis(
            conversation_messages=generate_test_conversation_data(),
            symbols=["AAPL", "MSFT"],  # Use reliable symbols
            period=TEST_PERIOD,
            use_real_data=True
        )
        
        success = getattr(result, 'success', False)
        data_source = getattr(result, 'data_source', '')
        
        assert success is True
        assert "FMP" in data_source or "Data from FMPDataProvider" in data_source
        
        print(f"✓ FMP Integration: {data_source}")
    
    @pytest.mark.asyncio
    async def test_health_check_comprehensive(self, service):
        """Test 5: Comprehensive health check"""
        health = await service.health_check()
        
        assert 'status' in health
        assert health['status'] in ['healthy', 'degraded', 'unhealthy']
        assert 'service' in health
        assert 'checks' in health
        
        checks = health['checks']
        assert 'basic_analysis' in checks
        assert 'centralized_models_support' in checks
        assert 'tools_import' in checks
        
        print(f"✓ Health Status: {health['status']}")
        print(f"✓ Basic Analysis: {checks['basic_analysis']}")
        print(f"✓ Centralized Models: {checks['centralized_models_support']}")

class TestBehavioralAPIEndpoints:
    """API layer integration testing"""
    
    def test_api_comprehensive_analysis_legacy_format(self):
        """Test 6: API endpoint with legacy message format"""
        test_request = {
            "conversation_messages": [
                {"role": "user", "content": "I'm worried about market volatility"},
                {"role": "user", "content": "Everyone is buying tech stocks"}
            ],
            "symbols": ["AAPL", "MSFT"],
            "period": "SIX_MONTHS",
            "use_real_data": False
        }
        
        response = client.post("/api/v1/behavioral/analyze", json=test_request)
        
        assert response.status_code == 200
        result = response.json()
        
        assert result["success"] is True
        assert "bias_count" in result
        assert "overall_risk_score" in result
        assert "recommendations" in result
        
        print(f"✓ API Comprehensive Analysis: {result['bias_count']} biases")
    
    def test_api_bias_detection_endpoint(self):
        """Test 7: Specific bias detection endpoint"""
        test_request = {
            "conversation_messages": [
                {"role": "user", "content": "I definitely think this stock will go up"},
                {"role": "user", "content": "I can't afford to lose money on this"}
            ],
            "bias_types": ["overconfidence", "loss_aversion"],
            "symbols": ["AAPL"],
            "period": "THREE_MONTHS",
            "use_real_data": False
        }
        
        response = client.post("/api/v1/behavioral/bias-detection", json=test_request)
        
        assert response.status_code == 200
        result = response.json()
        
        assert result["success"] is True
        print(f"✓ API Bias Detection: {result.get('bias_count', 0)} biases")
    
    def test_api_sentiment_analysis_endpoint(self):
        """Test 8: Sentiment analysis endpoint"""
        test_request = {
            "conversation_messages": [
                {"role": "user", "content": "I'm really excited about the market right now"},
                {"role": "user", "content": "But I'm also worried about a correction"}
            ],
            "symbols": ["AAPL", "MSFT"],
            "period": "SIX_MONTHS",
            "use_real_data": False
        }
        
        response = client.post("/api/v1/behavioral/sentiment-analysis", json=test_request)
        
        assert response.status_code == 200
        result = response.json()
        
        assert result["success"] is True
        print(f"✓ API Sentiment Analysis: Success")
    
    def test_api_health_check(self):
        """Test 9: API health check endpoint"""
        response = client.get("/api/v1/behavioral/health")
        
        assert response.status_code == 200
        result = response.json()
        
        assert "status" in result
        assert result["service"] == "behavioral_analysis"
        
        print(f"✓ API Health Check: {result['status']}")
    
    def test_api_capabilities(self):
        """Test 10: API capabilities endpoint"""
        response = client.get("/api/v1/behavioral/capabilities")
        
        assert response.status_code == 200
        result = response.json()
        
        assert "service_info" in result
        assert "api_endpoints" in result
        assert "supported_bias_types" in result
        
        print(f"✓ API Capabilities: {len(result['supported_bias_types'])} bias types")

class TestBehavioralPerformance:
    """Performance and reliability testing"""
    
    @pytest.mark.asyncio
    async def test_response_time_benchmark(self):
        """Test 11: Response time benchmark (< 5 seconds per handbook)"""
        service = BehavioralAnalysisService()
        messages = generate_test_conversation_data()
        
        start_time = time.time()
        
        result = await service.comprehensive_behavioral_analysis(
            conversation_messages=messages,
            symbols=["AAPL"],
            period="6month",
            use_real_data=False
        )
        
        execution_time = time.time() - start_time
        
        # Handbook requirement: < 5 seconds
        assert execution_time < 5.0, f"Response time {execution_time:.2f}s exceeds 5s limit"
        
        success = getattr(result, 'success', False)
        assert success is True
        
        print(f"✓ Performance Benchmark: {execution_time:.2f}s (< 5.0s requirement)")
    
    def test_api_response_time_benchmark(self):
        """Test 12: API response time benchmark"""
        test_request = {
            "conversation_messages": generate_test_conversation_data(),
            "symbols": ["AAPL", "MSFT"],
            "period": "SIX_MONTHS",
            "use_real_data": False
        }
        
        start_time = time.time()
        response = client.post("/api/v1/behavioral/analyze", json=test_request)
        api_time = time.time() - start_time
        
        assert response.status_code == 200
        assert api_time < 6.0  # API layer adds ~1s overhead
        
        result = response.json()
        service_time = result.get("execution_time", 0)
        
        print(f"✓ API Performance: {api_time:.2f}s total, {service_time:.2f}s service")
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test 13: Concurrent request handling"""
        service = BehavioralAnalysisService()
        messages = generate_test_conversation_data()
        
        # Create 3 concurrent requests
        tasks = []
        for i in range(3):
            task = service.comprehensive_behavioral_analysis(
                conversation_messages=messages,
                symbols=[f"TEST{i}"],
                period="6month",
                use_real_data=False
            )
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        
        # All should succeed
        for result in results:
            success = getattr(result, 'success', False)
            assert success is True
        
        # Should handle concurrency efficiently
        assert concurrent_time < 8.0  # Some overhead expected
        
        print(f"✓ Concurrent Handling: 3 requests in {concurrent_time:.2f}s")

class TestBehavioralErrorHandling:
    """Error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_empty_messages_handling(self):
        """Test 14: Empty messages handling"""
        service = BehavioralAnalysisService()
        
        result = await service.comprehensive_behavioral_analysis(
            conversation_messages=[],
            symbols=["AAPL"],
            period="6month",
            use_real_data=False
        )
        
        # Should handle gracefully
        success = getattr(result, 'success', False)
        assert success is False  # Should fail gracefully
        error = getattr(result, 'error', '')
        assert 'conversation messages' in error.lower()
        
        print("✓ Empty Messages: Handled gracefully")
    
    def test_api_invalid_request_format(self):
        """Test 15: Invalid API request format"""
        invalid_request = {
            "conversation_messages": "invalid_format",  # Should be list
            "symbols": ["AAPL"],
            "period": "INVALID_PERIOD"
        }
        
        response = client.post("/api/v1/behavioral/analyze", json=invalid_request)
        
        # Should return validation error
        assert response.status_code == 422  # Pydantic validation error
        
        print("✓ Invalid Request Format: Properly rejected")
    
    @pytest.mark.asyncio
    async def test_large_message_volume_handling(self):
        """Test 16: Large message volume handling"""
        service = BehavioralAnalysisService()
        
        # Create large conversation (50 messages)
        large_messages = []
        for i in range(50):
            large_messages.append({
                "role": "user",
                "content": f"This is test message {i} about market volatility and investment concerns"
            })
        
        start_time = time.time()
        result = await service.comprehensive_behavioral_analysis(
            conversation_messages=large_messages,
            symbols=["AAPL"],
            period="6month",
            use_real_data=False
        )
        execution_time = time.time() - start_time
        
        success = getattr(result, 'success', False)
        assert success is True
        assert execution_time < 10.0  # Should handle large volumes
        
        print(f"✓ Large Volume: 50 messages processed in {execution_time:.2f}s")

# =============================================================================
# PHASE 6: VALIDATION AND DEPLOYMENT (30 minutes)
# =============================================================================

class TestBehavioralValidationChecklist:
    """Final validation following handbook checklist"""
    
    @pytest.mark.asyncio
    async def test_functionality_preservation(self):
        """Test 17: All original features working"""
        service = BehavioralAnalysisService()
        messages = generate_test_conversation_data()
        
        # Test all major functions
        comprehensive = await service.comprehensive_behavioral_analysis(
            messages, ["AAPL"], "6month", False
        )
        
        sentiment = await service.sentiment_analysis_with_market_context(
            messages, ["AAPL"], "6month", False
        )
        
        profile = await service.behavioral_profile_assessment(
            messages, ["AAPL"], "6month", None, False
        )
        
        # All should succeed
        assert getattr(comprehensive, 'success', False) is True
        assert sentiment['success'] is True
        assert profile['success'] is True
        
        print("✓ All Original Features: Working")
    
    def test_real_data_integration_functional(self):
        """Test 18: Real data integration functional (if available)"""
        service = BehavioralAnalysisService()
        
        if service.fmp_available:
            # Test with real symbols
            test_request = {
                "conversation_messages": generate_test_conversation_data(),
                "symbols": ["AAPL"],
                "period": "THREE_MONTHS",
                "use_real_data": True
            }
            
            response = client.post("/api/v1/behavioral/analyze", json=test_request)
            assert response.status_code == 200
            
            result = response.json()
            assert result["success"] is True
            assert "FMP" in result.get("data_source", "")
            
            print("✓ Real Data Integration: Functional")
        else:
            print("⚠ Real Data Integration: Not available (synthetic fallback working)")
    
    def test_synthetic_fallback_operational(self):
        """Test 19: Synthetic fallback operational"""
        test_request = {
            "conversation_messages": generate_test_conversation_data(),
            "symbols": ["AAPL"],
            "period": "SIX_MONTHS",
            "use_real_data": False  # Force synthetic
        }
        
        response = client.post("/api/v1/behavioral/analyze", json=test_request)
        assert response.status_code == 200
        
        result = response.json()
        assert result["success"] is True
        
        print("✓ Synthetic Fallback: Operational")
    
    def test_health_checks_responsive(self):
        """Test 20: Health checks responsive"""
        # Service health
        service = BehavioralAnalysisService()
        service_health = asyncio.run(service.health_check())
        assert 'status' in service_health
        
        # API health
        response = client.get("/api/v1/behavioral/health")
        assert response.status_code == 200
        
        print("✓ Health Checks: Responsive")

# =============================================================================
# COMPREHENSIVE TEST RUNNER
# =============================================================================

async def run_behavioral_validation_suite():
    """
    Run complete validation suite following Backend Refactoring Handbook
    """
    print("=" * 70)
    print("BEHAVIORAL ANALYSIS SERVICE - VALIDATION SUITE")
    print("Following Backend Refactoring Handbook - Phase 5 & 6")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # Initialize service
        service = BehavioralAnalysisService()
        
        print(f"\n🔍 SERVICE STATUS:")
        status = service.get_service_status()
        print(f"  Status: {status['status']}")
        print(f"  Version: {status['version']}")
        print(f"  Centralized Models: {status['capabilities']['centralized_models_support']}")
        print(f"  FMP Integration: {status['capabilities']['fmp_integration_available']}")
        print(f"  Tools Available: {status['capabilities']['behavioral_tools_available']}")
        
        print(f"\n🧪 RUNNING CORE FUNCTIONALITY TESTS:")
        
        # Test 1: Legacy format compatibility
        print("  1. Testing legacy message format...")
        legacy_messages = generate_test_conversation_data()
        legacy_result = await service.comprehensive_behavioral_analysis(
            legacy_messages, ["AAPL"], "6month", False
        )
        legacy_success = getattr(legacy_result, 'success', False)
        print(f"     ✓ Legacy format: {'PASS' if legacy_success else 'FAIL'}")
        
        # Test 2: Centralized models format
        print("  2. Testing centralized models format...")
        centralized_messages = generate_test_centralized_messages()
        centralized_result = await service.comprehensive_behavioral_analysis(
            centralized_messages, ["AAPL"], AnalysisPeriod.SIX_MONTHS, False
        )
        centralized_success = getattr(centralized_result, 'success', False)
        print(f"     ✓ Centralized models: {'PASS' if centralized_success else 'FAIL'}")
        
        # Test 3: Performance benchmark
        print("  3. Testing performance benchmark...")
        perf_start = time.time()
        perf_result = await service.comprehensive_behavioral_analysis(
            legacy_messages, ["AAPL", "MSFT"], "6month", False
        )
        perf_time = time.time() - perf_start
        perf_pass = perf_time < 5.0 and getattr(perf_result, 'success', False)
        print(f"     ✓ Performance ({perf_time:.2f}s): {'PASS' if perf_pass else 'FAIL'}")
        
        # Test 4: API integration
        print("  4. Testing API integration...")
        api_request = {
            "conversation_messages": legacy_messages,
            "symbols": ["AAPL"],
            "period": "SIX_MONTHS",
            "use_real_data": False
        }
        api_response = client.post("/api/v1/behavioral/analyze", json=api_request)
        api_pass = api_response.status_code == 200 and api_response.json().get("success", False)
        print(f"     ✓ API integration: {'PASS' if api_pass else 'FAIL'}")
        
        # Test 5: Health checks
        print("  5. Testing health checks...")
        health_result = await service.health_check()
        health_pass = health_result.get('status') in ['healthy', 'degraded']
        print(f"     ✓ Health checks: {'PASS' if health_pass else 'FAIL'}")
        
        # Test 6: Error handling
        print("  6. Testing error handling...")
        error_result = await service.comprehensive_behavioral_analysis([], [], "6month", False)
        error_success = getattr(error_result, 'success', True) is False  # Should fail gracefully
        print(f"     ✓ Error handling: {'PASS' if error_success else 'FAIL'}")
        
        print(f"\n📊 COMPREHENSIVE ANALYSIS TEST:")
        
        # Full analysis with detailed output
        test_messages = [
            {"role": "user", "content": "I'm really worried about losing money in this volatile market"},
            {"role": "user", "content": "Everyone is buying AI stocks, I don't want to miss out"},
            {"role": "user", "content": "I'm definitely sure tech will keep going up"},
            {"role": "user", "content": "I bought NVDA at $300, it was a great price"}
        ]
        
        full_result = await service.comprehensive_behavioral_analysis(
            test_messages, ["AAPL", "NVDA"], "6month", False
        )
        
        if getattr(full_result, 'success', False):
            bias_count = getattr(full_result, 'bias_count', 0)
            risk_score = getattr(full_result, 'overall_risk_score', 0)
            recommendations = getattr(full_result, 'recommendations', [])
            
            print(f"  Biases Detected: {bias_count}")
            print(f"  Risk Score: {risk_score:.1f}/100")
            print(f"  Recommendations: {len(recommendations)}")
            
            if hasattr(full_result, 'detected_biases'):
                for bias in getattr(full_result, 'detected_biases', [])[:3]:
                    if isinstance(bias, dict):
                        print(f"    - {bias.get('bias_type', 'unknown')}: {bias.get('confidence', 0):.2f} confidence")
        
        # Summary
        total_time = time.time() - start_time
        
        all_tests_pass = all([
            legacy_success, centralized_success, perf_pass,
            api_pass, health_pass, error_success
        ])
        
        print(f"\n📋 VALIDATION SUMMARY:")
        print(f"  Overall Status: {'✅ PASS' if all_tests_pass else '❌ FAIL'}")
        print(f"  Total Execution Time: {total_time:.2f}s")
        print(f"  Service Version: {status['version']}")
        print(f"  Integration Level: {'Full' if status['capabilities']['fmp_integration_available'] else 'Partial'}")
        
        if all_tests_pass:
            print(f"\n🎉 BEHAVIORAL ANALYSIS SERVICE VALIDATION COMPLETE")
            print(f"   Ready for production deployment!")
        else:
            print(f"\n⚠️  Some tests failed - review before deployment")
        
        return {
            'success': all_tests_pass,
            'execution_time': total_time,
            'test_results': {
                'legacy_compatibility': legacy_success,
                'centralized_models': centralized_success,
                'performance': perf_pass,
                'api_integration': api_pass,
                'health_checks': health_pass,
                'error_handling': error_success
            },
            'service_status': status
        }
        
    except Exception as e:
        print(f"\n❌ VALIDATION SUITE FAILED: {e}")
        return {'success': False, 'error': str(e)}
    
    finally:
        print("=" * 70)

if __name__ == "__main__":
    # Run the validation suite
    result = asyncio.run(run_behavioral_validation_suite())
    
    if result['success']:
        print("\n🚀 Behavioral Analysis Service ready for next service refactoring!")
        print("   Recommended next target: Forecasting Service (high complexity)")
    else:
        print("\n🔧 Fix identified issues before proceeding to next service")