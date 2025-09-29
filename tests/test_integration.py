"""
tests/test_integration.py - Comprehensive Integration Test Suite
===============================================================

Tests the entire risk analysis stack end-to-end with proper
setup and teardown.
"""

import pytest
import asyncio
import os
import sys
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_SYMBOLS = ["AAPL", "GOOGL", "MSFT"]
TEST_WEIGHTS = {"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3}

class TestRiskAnalysisIntegration:
    """Integration tests for risk analysis system"""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Create test client with proper dependency overrides"""
        from main import app
        from core.dependencies import reset_dependencies
        
        # Reset dependencies for testing
        reset_dependencies()
        
        # Set test environment
        os.environ["ENVIRONMENT"] = "test"
        os.environ["FMP_ENABLED"] = "false"  # Use synthetic data for tests
        
        with TestClient(app) as test_client:
            yield test_client
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup for each test method"""
        logger.info("Setting up test method")
        yield
        logger.info("Tearing down test method")
    
    def test_app_startup(self, client):
        """Test that the application starts successfully"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "risk-analysis-backend"
        assert data["status"] == "healthy"
        assert "endpoints" in data
    
    def test_health_check_global(self, client):
        """Test global health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "services" in data
        assert "risk_analysis" in data["services"]
    
    def test_risk_health_check(self, client):
        """Test risk service health check"""
        response = client.get("/api/risk/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "risk_analysis"
        assert data["status"] in ["healthy", "unhealthy"]
        assert "capabilities" in data
        assert "checks" in data
    
    def test_basic_risk_analysis(self, client):
        """Test basic risk analysis with synthetic data"""
        request_data = {
            "symbols": TEST_SYMBOLS,
            "weights": TEST_WEIGHTS,
            "portfolio_id": "test_portfolio_001",
            "period": "6months",
            "use_real_data": False,  # Use synthetic for reliable testing
            "include_stress_testing": True
        }
        
        response = client.post("/api/risk/analyze", json=request_data)
        
        # Check response structure
        assert response.status_code == 200
        data = response.json()
        
        #