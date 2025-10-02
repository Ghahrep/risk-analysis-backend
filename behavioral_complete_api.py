""""
Complete Behavioral Analysis API - Port 8003
===========================================

Following proven minimal API pattern with all behavioral endpoints.
Comprehensive behavioral finance analysis with real FMP market data integration.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)
app = FastAPI(
    title="Behavioral Analysis API - Complete",
    description="Comprehensive behavioral finance analysis with FMP market data integration",
    version="1.0.0"
)

def convert_numpy_types(obj):
    """Convert numpy types to JSON serializable types"""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

@app.post("/analyze-biases")
async def analyze_behavioral_biases_endpoint(request: dict):
    """
    Comprehensive behavioral bias analysis with real FMP market data
    
    Request format:
    {
        "conversation_messages": [
            {"role": "user", "content": "I'm worried about my portfolio"},
            {"role": "user", "content": "Tech stocks always go up"}
        ],
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "period": "1year",
        "use_real_data": true
    }
    """
    try:
        logger.info("Starting behavioral bias analysis")
        
        # Import here to avoid startup issues
        from tools.behavioral_tools_standalone import analyze_behavioral_biases_with_real_data
        
        # Validate required fields
        if 'conversation_messages' not in request:
            raise HTTPException(status_code=400, detail="conversation_messages field is required")
        
        conversation_messages = request.get('conversation_messages', [])
        symbols = request.get('symbols', [])
        period = request.get('period', '1year')
        use_real_data = request.get('use_real_data', True)
        
        if not conversation_messages:
            raise HTTPException(status_code=400, detail="At least one conversation message is required")
        
        # Call behavioral analysis function
        analysis_result = await analyze_behavioral_biases_with_real_data(
            conversation_messages=conversation_messages,
            symbols=symbols,
            period=period,
            use_real_data=use_real_data
        )
        
        if not analysis_result.get('success', False):
            raise HTTPException(
                status_code=400,
                detail=f"Bias analysis failed: {analysis_result.get('error', 'Unknown error')}"
            )
        
        converted_result = convert_numpy_types(analysis_result)
        logger.info(f"✓ Bias analysis completed: {converted_result.get('bias_count', 0)} biases detected")
        return JSONResponse(content=converted_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bias analysis endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bias analysis error: {str(e)}")

@app.post("/analyze-sentiment")
async def analyze_sentiment_endpoint(request: dict):
    """
    Market sentiment analysis with real FMP market context
    
    Request format:
    {
        "conversation_messages": [
            {"role": "user", "content": "Market looks great today!"},
            {"role": "user", "content": "I'm bullish on tech"}
        ],
        "symbols": ["AAPL", "GOOGL"],
        "period": "6months",
        "use_real_data": true
    }
    """
    try:
        logger.info("Starting market sentiment analysis")
        
        # Import here to avoid startup issues
        from tools.behavioral_tools_standalone import analyze_market_sentiment_with_real_data
        
        # Validate required fields
        if 'conversation_messages' not in request:
            raise HTTPException(status_code=400, detail="conversation_messages field is required")
        
        conversation_messages = request.get('conversation_messages', [])
        symbols = request.get('symbols', [])
        period = request.get('period', '1year')
        use_real_data = request.get('use_real_data', True)
        
        if not conversation_messages:
            raise HTTPException(status_code=400, detail="At least one conversation message is required")
        
        # Call sentiment analysis function
        sentiment_result = await analyze_market_sentiment_with_real_data(
            conversation_messages=conversation_messages,
            symbols=symbols,
            period=period,
            use_real_data=use_real_data
        )
        
        if not sentiment_result.get('success', False):
            raise HTTPException(
                status_code=400,
                detail=f"Sentiment analysis failed: {sentiment_result.get('error', 'Unknown error')}"
            )
        
        converted_result = convert_numpy_types(sentiment_result)
        logger.info(f"✓ Sentiment analysis completed: {converted_result.get('sentiment', 'neutral')} sentiment")
        return JSONResponse(content=converted_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sentiment analysis endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis error: {str(e)}")

@app.post("/assess-profile")
async def assess_behavioral_profile_endpoint(request: dict):
    """
    Comprehensive behavioral profile assessment with FMP market context
    
    Request format:
    {
        "conversation_messages": [
            {"role": "user", "content": "I always stick to my investment plan"},
            {"role": "user", "content": "Market volatility doesn't bother me"}
        ],
        "symbols": ["AAPL", "MSFT", "GOOGL"],
        "period": "1year",
        "user_demographics": {"experience": "intermediate"},
        "use_real_data": true
    }
    """
    try:
        logger.info("Starting behavioral profile assessment")
        
        # Import here to avoid startup issues
        from tools.behavioral_tools_standalone import assess_behavioral_profile_with_real_data
        
        # Validate required fields
        if 'conversation_messages' not in request:
            raise HTTPException(status_code=400, detail="conversation_messages field is required")
        
        conversation_messages = request.get('conversation_messages', [])
        symbols = request.get('symbols', [])
        period = request.get('period', '1year')
        user_demographics = request.get('user_demographics', None)
        use_real_data = request.get('use_real_data', True)
        
        if not conversation_messages:
            raise HTTPException(status_code=400, detail="At least one conversation message is required")
        
        # Call behavioral profile assessment function
        profile_result = await assess_behavioral_profile_with_real_data(
            conversation_messages=conversation_messages,
            symbols=symbols,
            period=period,
            user_demographics=user_demographics,
            use_real_data=use_real_data
        )
        
        if not profile_result.get('success', False):
            raise HTTPException(
                status_code=400,
                detail=f"Profile assessment failed: {profile_result.get('error', 'Unknown error')}"
            )
        
        converted_result = convert_numpy_types(profile_result)
        maturity_level = converted_result.get('profile', {}).get('maturity_level', 'Unknown')
        logger.info(f"✓ Profile assessment completed: {maturity_level} maturity level")
        return JSONResponse(content=converted_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile assessment endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Profile assessment error: {str(e)}")

@app.post("/comprehensive-analysis")
async def comprehensive_behavioral_analysis_endpoint(request: dict):
    """
    Complete behavioral analysis combining bias detection, sentiment, and profiling
    
    Request format:
    {
        "conversation_messages": [
            {"role": "user", "content": "I'm really worried about this market volatility"},
            {"role": "user", "content": "But I'm sure my tech picks will outperform"}
        ],
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "period": "1year",
        "user_demographics": {"experience": "intermediate", "age_group": "30-40"},
        "analysis_depth": "comprehensive",
        "use_real_data": true
    }
    """
    try:
        logger.info("Starting comprehensive behavioral analysis")
        
        # Import here to avoid startup issues
        from tools.behavioral_tools_standalone import analyze_conversation_comprehensive
        
        # Validate required fields
        if 'conversation_messages' not in request:
            raise HTTPException(status_code=400, detail="conversation_messages field is required")
        
        conversation_messages = request.get('conversation_messages', [])
        symbols = request.get('symbols', [])
        user_context = {
            'demographics': request.get('user_demographics', {}),
            'analysis_preferences': {
                'depth': request.get('analysis_depth', 'standard'),
                'focus_areas': request.get('focus_areas', [])
            }
        }
        analysis_depth = request.get('analysis_depth', 'standard')
        
        if not conversation_messages:
            raise HTTPException(status_code=400, detail="At least one conversation message is required")
        
        # Call comprehensive analysis function
        comprehensive_result = analyze_conversation_comprehensive(
            conversation_messages=conversation_messages,
            symbols=symbols,
            user_context=user_context,
            analysis_depth=analysis_depth
        )
        
        if not comprehensive_result.get('success', False):
            raise HTTPException(
                status_code=400,
                detail=f"Comprehensive analysis failed: {comprehensive_result.get('error', 'Unknown error')}"
            )
        
        converted_result = convert_numpy_types(comprehensive_result)
        component_count = len([k for k, v in converted_result.get('components', {}).items() if v.get('success')])
        logger.info(f"✓ Comprehensive analysis completed: {component_count} components analyzed")
        return JSONResponse(content=converted_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comprehensive analysis endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis error: {str(e)}")

@app.post("/test-tools")
async def test_behavioral_tools_endpoint():
    """Test behavioral analysis tools functionality with sample data"""
    try:
        logger.info("Testing behavioral analysis tools")
        
        # Import here to avoid startup issues
        from tools.behavioral_tools_standalone import test_behavioral_integration
        
        # Run comprehensive test
        test_result = await test_behavioral_integration(
            test_symbols=["AAPL", "GOOGL", "MSFT"],
            test_messages=[
                {"role": "user", "content": "I'm worried about losing money in this volatile market"},
                {"role": "user", "content": "I definitely think AI stocks will keep going up"},
                {"role": "user", "content": "Everyone is buying tech stocks, I don't want to miss out"}
            ]
        )
        
        converted_result = convert_numpy_types(test_result)
        logger.info("✓ Behavioral tools test completed")
        return JSONResponse(content=converted_result)
        
    except Exception as e:
        logger.error(f"Behavioral tools test failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "tools_status": "error",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/health")
async def health_check():
    """Health check for behavioral analysis service"""
    try:
        # Import here to avoid startup issues
        from tools.behavioral_tools_standalone import get_integration_status
        
        # Get integration status
        integration_status = get_integration_status()
        
        health_status = {
            "status": "healthy",
            "service": "behavioral_analysis",
            "version": "1.0.0",
            "port": 8003,
            "components": {
                "behavioral_tools": integration_status.get("fmp_available", False),
                "fmp_integration": integration_status.get("fmp_available", False),
                "api_server": True
            },
            "available_endpoints": [
                "/analyze-biases",
                "/analyze-sentiment", 
                "/assess-profile",
                "/comprehensive-analysis",
                "/test-tools"
            ],
            "integration_status": integration_status,
            "timestamp": datetime.now().isoformat()
        }
        
        # Determine overall status
        if not integration_status.get("fmp_available", False):
            health_status["status"] = "degraded"
            health_status["issues"] = ["FMP integration not available - limited to conversation analysis"]
        
        logger.info(f"Health check completed: {health_status['status']}")
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "behavioral_analysis",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/capabilities")
async def get_capabilities():
    """Get detailed behavioral analysis capabilities"""
    try:
        capabilities = {
            "service_info": {
                "name": "behavioral_analysis",
                "version": "1.0.0",
                "description": "Comprehensive behavioral finance analysis with FMP market data integration",
                "endpoints_count": 5
            },
            "analysis_types": {
                "bias_detection": {
                    "endpoint": "/analyze-biases",
                    "supported_biases": [
                        "loss_aversion", "overconfidence", "herding_fomo", 
                        "anchoring", "confirmation"
                    ],
                    "fmp_enhancement": "Portfolio context amplifies bias detection accuracy",
                    "output": "Individual bias scores with evidence and mitigation strategies"
                },
                "sentiment_analysis": {
                    "endpoint": "/analyze-sentiment",
                    "capabilities": [
                        "Investor sentiment classification",
                        "Market timing risk assessment", 
                        "Emotional intensity measurement",
                        "Sentiment-market alignment analysis"
                    ],
                    "fmp_enhancement": "Real market data provides sentiment-market alignment analysis",
                    "output": "Sentiment classification with timing risk and recommendations"
                },
                "behavioral_profiling": {
                    "endpoint": "/assess-profile", 
                    "assessments": [
                        "Behavioral maturity level",
                        "Decision-making confidence",
                        "Risk tolerance assessment",
                        "Dominant bias patterns"
                    ],
                    "fmp_enhancement": "Portfolio performance context enhances profile accuracy",
                    "output": "Comprehensive behavioral profile with maturity assessment"
                },
                "comprehensive_analysis": {
                    "endpoint": "/comprehensive-analysis",
                    "components": [
                        "Complete bias detection",
                        "Sentiment analysis", 
                        "Behavioral profiling",
                        "Unified recommendations"
                    ],
                    "output": "Multi-component analysis with unified insights and recommendations"
                }
            },
            "data_integration": {
                "fmp_integration": "Real-time market data for portfolio context",
                "supported_periods": ["1month", "3months", "6months", "1year", "2years"],
                "portfolio_context": "Real market data amplifies behavioral analysis accuracy",
                "fallback_mode": "Conversation-only analysis when FMP unavailable",
                "data_sources": ["FMP market data", "Portfolio performance metrics", "Conversation analysis"]
            },
            "ethical_framework": {
                "focus": "Educational and awareness-building",
                "boundaries": "No psychological manipulation or exploitation",
                "approach": "Evidence-based behavioral finance theory",
                "recommendations": "Constructive mitigation strategies and professional guidance"
            },
            "api_features": {
                "async_processing": True,
                "real_time_analysis": True,
                "numpy_serialization": True,
                "error_handling": "Comprehensive with fallbacks",
                "input_validation": "Required field validation with helpful error messages"
            }
        }
        
        return capabilities
        
    except Exception as e:
        logger.error(f"Failed to get capabilities: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve service capabilities")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("BEHAVIORAL ANALYSIS API - COMPLETE VERSION - PORT 8003")
    logger.info("=" * 60)
    logger.info("Available endpoints:")
    logger.info("  POST /analyze-biases         - Comprehensive bias detection with FMP context")
    logger.info("  POST /analyze-sentiment      - Market sentiment analysis with alignment") 
    logger.info("  POST /assess-profile         - Behavioral maturity and risk assessment")
    logger.info("  POST /comprehensive-analysis - Complete multi-component analysis")
    logger.info("  POST /test-tools             - Test behavioral tools functionality")
    logger.info("  GET  /health                 - Service health check with integration status")
    logger.info("  GET  /capabilities           - Detailed service capabilities")
    logger.info("=" * 60)
    logger.info("Features:")
    logger.info("  ✓ Real FMP market data integration")
    logger.info("  ✓ Ethical behavioral finance analysis")
    logger.info("  ✓ Educational focus with mitigation strategies")
    logger.info("  ✓ Portfolio context enhancement")
    logger.info("  ✓ Comprehensive error handling")
    logger.info("=" * 60)
    
    # Run the API server
    import os
    port = int(os.environ.get("PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port)