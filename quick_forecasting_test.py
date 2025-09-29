# quick_forecasting_test.py - Simple Forecasting Service Test
"""
Quick test to verify forecasting service works with your actual file structure
"""

import sys
import os
import asyncio
import logging

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_forecasting_import():
    """Test if forecasting service can be imported"""
    try:
        logger.info("Testing forecasting service import...")
        from services.forecasting_service_updated import ForecastingService
        logger.info("✓ Import successful")
        return True, ForecastingService
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False, None

def test_forecasting_basic(ForecastingService):
    """Test basic forecasting functionality"""
    try:
        logger.info("Testing service initialization...")
        service = ForecastingService()
        logger.info(f"✓ Service initialized: {service.service_name} v{service.version}")
        
        logger.info("Testing basic forecasting...")
        result = service.forecast_returns(
            symbols=["AAPL"],
            period="6months",
            use_real_data=False,
            forecast_horizon=5,
            model_type="simple_mean"
        )
        
        success = getattr(result, 'success', False) if hasattr(result, 'success') else result.get('success', False)
        
        if success:
            logger.info("✓ Basic forecasting test passed")
            return True
        else:
            error = getattr(result, 'error', 'Unknown') if hasattr(result, 'error') else result.get('error', 'Unknown')
            logger.error(f"✗ Basic forecasting failed: {error}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Basic test failed: {e}")
        return False

def test_forecasting_health(ForecastingService):
    """Test health check"""
    try:
        logger.info("Testing health check...")
        service = ForecastingService()
        health = service.health_check()
        
        status = health.get('status', 'unknown')
        logger.info(f"✓ Health check: {status}")
        return status in ['healthy', 'degraded']
        
    except Exception as e:
        logger.error(f"✗ Health check failed: {e}")
        return False

def main():
    """Run quick forecasting tests"""
    logger.info("=" * 50)
    logger.info("QUICK FORECASTING SERVICE TEST")
    logger.info("=" * 50)
    
    # Test 1: Import
    import_success, ForecastingService = test_forecasting_import()
    if not import_success:
        logger.error("Cannot proceed - import failed")
        return False
    
    # Test 2: Basic functionality
    basic_success = test_forecasting_basic(ForecastingService)
    
    # Test 3: Health check
    health_success = test_forecasting_health(ForecastingService)
    
    # Summary
    logger.info("=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Import: {'✓' if import_success else '✗'}")
    logger.info(f"Basic Function: {'✓' if basic_success else '✗'}")
    logger.info(f"Health Check: {'✓' if health_success else '✗'}")
    
    overall_success = import_success and basic_success and health_success
    logger.info(f"Overall: {'✓ PASS' if overall_success else '✗ FAIL'}")
    
    if overall_success:
        logger.info("🎉 Forecasting service ready for integration!")
    else:
        logger.info("⚠️ Issues need to be resolved before integration")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)