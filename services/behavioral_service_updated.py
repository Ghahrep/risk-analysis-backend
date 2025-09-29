# services/behavioral_service_updated.py - Aligned with Centralized Models
"""
Behavioral Analysis Service - Updated for Centralized Models Integration
=======================================================================

Service layer updated to work seamlessly with centralized models/requests.py structure.
Handles both ConversationMessage objects and raw dicts for backward compatibility.
"""

import asyncio
import importlib.util
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

# Import centralized models for type hints and validation
def import_centralized_models():
    """Import centralized models with proper path handling"""
    try:
        import sys
        import os
        
        # Add the project root to Python path
        project_root = os.path.dirname(os.path.dirname(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from models.requests import (
            ConversationMessage, BehavioralAnalysisRequest, 
            BehavioralAnalysisResponse, AnalysisPeriod
        )
        return True, ConversationMessage, BehavioralAnalysisResponse, AnalysisPeriod
    except ImportError as e:
        logger.warning(f"Centralized models not available: {e}")
        return False, None, None, None

# Try to import centralized models
CENTRALIZED_MODELS_AVAILABLE, ConversationMessage, BehavioralAnalysisResponse, AnalysisPeriod = import_centralized_models()

if CENTRALIZED_MODELS_AVAILABLE:
    logger.info("Centralized models imported successfully")
else:
    # Define placeholder types for when centralized models aren't available
    ConversationMessage = None
    BehavioralAnalysisResponse = None
    AnalysisPeriod = None

# DIRECT IMPORT PATTERN (Following proven pattern)
def import_behavioral_tools():
    """Import behavioral tools directly from file"""
    try:
        import sys
        import os
        
        project_root = os.path.dirname(os.path.dirname(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        spec = importlib.util.spec_from_file_location(
            "behavioral_tools_standalone", 
            os.path.join(project_root, "tools", "behavioral_tools_standalone.py")
        )
        behavioral_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(behavioral_module)
        return behavioral_module
    except Exception as e:
        logger.error(f"Could not import behavioral tools: {e}")
        return None

def import_fmp_integration():
    """Import FMP integration directly from file"""
    try:
        import sys
        import os
        
        project_root = os.path.dirname(os.path.dirname(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            
        spec = importlib.util.spec_from_file_location(
            "fmp_integration", 
            os.path.join(project_root, "data", "providers", "fmp_integration.py")
        )
        fmp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fmp_module)
        return fmp_module.PortfolioDataManager, fmp_module.FMPDataProvider
    except Exception as e:
        logger.warning(f"Could not import FMP integration: {e}")
        return None, None

# Initialize imports
behavioral_tools = import_behavioral_tools()
PortfolioDataManager, FMPDataProvider = import_fmp_integration()

# Check availability
BEHAVIORAL_TOOLS_AVAILABLE = behavioral_tools is not None
FMP_AVAILABLE = PortfolioDataManager is not None and FMPDataProvider is not None

# Legacy response classes for backward compatibility
class LegacyBehavioralAnalysisResponse:
    """Legacy response format for backward compatibility"""
    def __init__(self, success: bool, data_source: str, analysis_type: str,
                 bias_count: int, overall_risk_score: float, recommendations: List[str],
                 execution_time: float, fmp_integration_used: bool, 
                 portfolio_context_available: bool, error: Optional[str] = None):
        self.success = success
        self.data_source = data_source
        self.analysis_type = analysis_type
        self.bias_count = bias_count
        self.overall_risk_score = overall_risk_score
        self.recommendations = recommendations
        self.execution_time = execution_time
        self.fmp_integration_used = fmp_integration_used
        self.portfolio_context_available = portfolio_context_available
        self.error = error

class BehavioralAnalysisService:
    """
    Updated Behavioral Analysis Service for Centralized Models Integration
    
    Key Updates:
    - Handles both ConversationMessage objects and raw dicts
    - Returns centralized response models when available
    - Maintains backward compatibility with existing integrations
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 confidence_levels: List[float] = [0.95, 0.99],
                 default_period: str = "1year"):
        """Initialize with centralized model support"""
        self.service_name = "behavioral_analysis_direct"
        self.version = "2.1.0"  # Updated for centralized models
        self.risk_free_rate = risk_free_rate
        self.confidence_levels = confidence_levels
        self.default_period = default_period
        
        # Initialize data manager if FMP available
        self._data_manager = None
        if FMP_AVAILABLE:
            try:
                self._data_manager = PortfolioDataManager()
                logger.info("FMP data manager initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize FMP data manager: {e}")
                self._data_manager = None
        
        # Service state
        self.tools_available = BEHAVIORAL_TOOLS_AVAILABLE
        self.fmp_available = self._data_manager is not None
        self.centralized_models_available = CENTRALIZED_MODELS_AVAILABLE
        
        logger.info(f"Initialized {self.service_name} v{self.version}")
        logger.info(f"Tools: {self.tools_available}, FMP: {self.fmp_available}, Models: {self.centralized_models_available}")

    def _normalize_conversation_messages(
        self, 
        messages: Union[List[Any], List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """
        Convert ConversationMessage objects to dict format for tools layer
        Handles both centralized models and legacy dict format
        """
        if not messages:
            return []
        
        normalized = []
        for msg in messages:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                # ConversationMessage object from centralized models
                normalized.append({
                    "role": msg.role,
                    "content": msg.content
                })
            elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                # Legacy dict format
                normalized.append(msg)
            else:
                logger.warning(f"Invalid message format: {type(msg)}")
                continue
        
        return normalized

    def _normalize_period(self, period: Union[str, Any, None]) -> str:
        """
        Convert AnalysisPeriod enum to string for tools layer
        Handles both centralized models and legacy string format
        """
        if period is None:
            return self.default_period
        
        # Handle enum from centralized models
        if hasattr(period, 'value'):
            return period.value
        
        # Handle string format
        if isinstance(period, str):
            return period
        
        logger.warning(f"Invalid period format: {type(period)}, using default")
        return self.default_period

    def _create_centralized_response(
        self,
        success: bool,
        data_source: str,
        execution_time: float,
        bias_count: int = 0,
        overall_risk_score: float = 50.0,
        detected_biases: List[Dict] = None,
        recommendations: List[str] = None,
        error: Optional[str] = None
    ) -> Union[Any, 'LegacyBehavioralAnalysisResponse']:
        """
        Create response using centralized models if available, fallback to legacy
        """
        if CENTRALIZED_MODELS_AVAILABLE:
            return BehavioralAnalysisResponse(
                success=success,
                message="Behavioral analysis completed" if success else "Analysis failed",
                data_source=data_source,
                execution_time=execution_time,
                timestamp=datetime.now(),
                bias_count=bias_count,
                overall_risk_score=overall_risk_score,
                detected_biases=detected_biases or [],
                recommendations=recommendations or [],
                error=error
            )
        else:
            # Fallback to legacy response
            return LegacyBehavioralAnalysisResponse(
                success=success,
                data_source=data_source,
                analysis_type="behavioral_analysis",
                bias_count=bias_count,
                overall_risk_score=overall_risk_score,
                recommendations=recommendations or [],
                execution_time=execution_time,
                fmp_integration_used="FMP" in data_source,
                portfolio_context_available=bool(detected_biases),
                error=error
            )

    async def comprehensive_behavioral_analysis(
        self,
        conversation_messages: Union[List[Any], List[Dict[str, str]]],
        symbols: List[str],
        period: Union[str, Any, None] = None,
        use_real_data: bool = True
    ) -> Union[Any, 'LegacyBehavioralAnalysisResponse']:
        """
        UPDATED: Comprehensive behavioral analysis with centralized model support
        
        Handles both ConversationMessage objects and legacy dict format
        Returns centralized response models when available
        """
        start_time = datetime.now()
        
        try:
            # Normalize inputs to work with tools layer
            normalized_messages = self._normalize_conversation_messages(conversation_messages)
            normalized_period = self._normalize_period(period)
            
            logger.info(f"Starting behavioral analysis: {len(normalized_messages)} messages, {len(symbols)} symbols")
            
            if not BEHAVIORAL_TOOLS_AVAILABLE:
                execution_time = (datetime.now() - start_time).total_seconds()
                return self._create_centralized_response(
                    success=False,
                    data_source="Error",
                    execution_time=execution_time,
                    error="Behavioral tools not available"
                )
            
            if not normalized_messages:
                execution_time = (datetime.now() - start_time).total_seconds()
                return self._create_centralized_response(
                    success=False,
                    data_source="Error", 
                    execution_time=execution_time,
                    error="No conversation messages provided"
                )
            
            # Use FMP-integrated analysis if available and requested
            if use_real_data and self.fmp_available and symbols:
                logger.info("Using FMP-integrated behavioral analysis")
                
                # Call the async FMP-integrated function
                analysis_result = await behavioral_tools.analyze_behavioral_biases_with_real_data(
                    conversation_messages=normalized_messages,
                    symbols=symbols,
                    period=normalized_period,
                    use_real_data=True
                )
                
                data_source = analysis_result.get('data_source', 'FMP + Conversation Analysis')
                
            else:
                logger.info("Using conversation-only behavioral analysis")
                
                # Use the legacy sync function for conversation-only analysis
                analysis_result = behavioral_tools.analyze_behavioral_biases(
                    conversation_messages=normalized_messages,
                    portfolio_data=None,
                    market_context=None
                )
                
                data_source = "Conversation Analysis Only"
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract key metrics
            success = analysis_result.get('success', False)
            bias_count = analysis_result.get('bias_count', 0)
            overall_risk_score = analysis_result.get('overall_risk_score', 50.0)
            recommendations = analysis_result.get('recommendations', [])
            
            # Extract detected biases for centralized response
            detected_biases = analysis_result.get('detected_biases', [])
            if not detected_biases and bias_count > 0:
                # Create basic bias info if detailed results not available
                detected_biases = [{"type": "general", "confidence": 0.5, "evidence": "Based on conversation analysis"}]
            
            response = self._create_centralized_response(
                success=success,
                data_source=data_source,
                execution_time=execution_time,
                bias_count=bias_count,
                overall_risk_score=round(overall_risk_score, 2),
                detected_biases=detected_biases,
                recommendations=recommendations,
                error=analysis_result.get('error') if not success else None
            )
            
            if success:
                logger.info(f"✓ Analysis completed: {bias_count} biases, {overall_risk_score:.1f} risk score, {execution_time:.2f}s")
            else:
                logger.warning(f"✗ Analysis failed: {analysis_result.get('error', 'Unknown error')}")
            
            return response
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Comprehensive behavioral analysis failed: {e}")
            
            return self._create_centralized_response(
                success=False,
                data_source="Error",
                execution_time=execution_time,
                error=str(e)
            )

    # Updated versions of other key methods with same pattern

    async def sentiment_analysis_with_market_context(
        self,
        conversation_messages: Union[List[Any], List[Dict[str, str]]],
        symbols: Optional[List[str]] = None,
        period: Union[str, Any, None] = None,
        use_real_data: bool = True
    ) -> Dict[str, Any]:
        """
        UPDATED: Sentiment analysis with centralized model support
        """
        start_time = datetime.now()
        
        try:
            # Normalize inputs
            normalized_messages = self._normalize_conversation_messages(conversation_messages)
            normalized_period = self._normalize_period(period)
            
            logger.info("Starting sentiment analysis with market context")
            
            if not BEHAVIORAL_TOOLS_AVAILABLE:
                execution_time = (datetime.now() - start_time).total_seconds()
                return {
                    'success': False,
                    'error': 'Behavioral tools not available',
                    'data_source': 'Error',
                    'sentiment': 'neutral',
                    'confidence': 0.0,
                    'market_timing_risk': 0.5,
                    'recommendations': [],
                    'execution_time': execution_time
                }
            
            # Use FMP-enhanced sentiment analysis if available
            if use_real_data and self.fmp_available and symbols:
                logger.info("Using FMP-enhanced sentiment analysis")
                
                sentiment_result = await behavioral_tools.analyze_market_sentiment_with_real_data(
                    conversation_messages=normalized_messages,
                    symbols=symbols,
                    period=normalized_period,
                    use_real_data=True
                )
                
                data_source = sentiment_result.get('data_source', 'FMP + Conversation Analysis')
            else:
                logger.info("Using conversation-only sentiment analysis")
                
                sentiment_result = behavioral_tools.analyze_market_sentiment(
                    conversation_messages=normalized_messages,
                    market_context=None
                )
                
                data_source = "Conversation Analysis Only"
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Return standardized dict format (not dataclass for flexibility)
            response = {
                'success': sentiment_result.get('success', False),
                'data_source': data_source,
                'sentiment': sentiment_result.get('sentiment', 'neutral'),
                'confidence': round(sentiment_result.get('confidence', 0.5), 3),
                'market_timing_risk': round(sentiment_result.get('market_timing_risk', 0.5), 3),
                'recommendations': sentiment_result.get('recommendations', []),
                'execution_time': round(execution_time, 3),
                'error': sentiment_result.get('error') if not sentiment_result.get('success') else None
            }
            
            if response['success']:
                logger.info(f"✓ Sentiment analysis completed: {response['sentiment']}, {execution_time:.2f}s")
            else:
                logger.warning(f"✗ Sentiment analysis failed: {response['error']}")
            
            return response
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Sentiment analysis failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'data_source': 'Error',
                'sentiment': 'neutral',
                'confidence': 0.0,
                'market_timing_risk': 0.5,
                'recommendations': [],
                'execution_time': round(execution_time, 3)
            }

    async def behavioral_profile_assessment(
        self,
        conversation_messages: Union[List[Any], List[Dict[str, str]]],
        symbols: Optional[List[str]] = None,
        period: Union[str, Any, None] = None,
        user_demographics: Optional[Dict] = None,
        use_real_data: bool = True
    ) -> Dict[str, Any]:
        """
        UPDATED: Behavioral profile assessment with centralized model support
        """
        start_time = datetime.now()
        
        try:
            # Normalize inputs
            normalized_messages = self._normalize_conversation_messages(conversation_messages)
            normalized_period = self._normalize_period(period)
            
            logger.info("Starting behavioral profile assessment")
            
            if not BEHAVIORAL_TOOLS_AVAILABLE:
                execution_time = (datetime.now() - start_time).total_seconds()
                return {
                    'success': False,
                    'error': 'Behavioral tools not available',
                    'data_source': 'Error',
                    'maturity_level': 'Unknown',
                    'overall_risk_score': 50.0,
                    'dominant_biases': [],
                    'recommendations': [],
                    'execution_time': execution_time
                }
            
            # Use FMP-enhanced profile assessment if available
            if use_real_data and self.fmp_available and symbols:
                logger.info("Using FMP-enhanced behavioral profile assessment")
                
                profile_result = await behavioral_tools.assess_behavioral_profile_with_real_data(
                    conversation_messages=normalized_messages,
                    symbols=symbols,
                    period=normalized_period,
                    user_demographics=user_demographics,
                    use_real_data=True
                )
                
                data_source = profile_result.get('data_source', 'FMP + Conversation Analysis')
            else:
                logger.info("Using conversation-only profile assessment")
                
                profile_result = behavioral_tools.assess_behavioral_profile(
                    conversation_messages=normalized_messages,
                    portfolio_data=None,
                    user_demographics=user_demographics
                )
                
                data_source = "Conversation Analysis Only"
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract profile metrics
            success = profile_result.get('success', False)
            profile_data = profile_result.get('profile', {})
            
            response = {
                'success': success,
                'data_source': data_source,
                'maturity_level': profile_data.get('maturity_level', 'Intermediate'),
                'overall_risk_score': round(profile_data.get('overall_risk_score', 50.0), 2),
                'dominant_biases': profile_data.get('dominant_biases', []),
                'recommendations': profile_data.get('recommendations', []),
                'execution_time': round(execution_time, 3),
                'error': profile_result.get('error') if not success else None
            }
            
            if success:
                logger.info(f"✓ Profile assessment completed: {response['maturity_level']}, {execution_time:.2f}s")
            else:
                logger.warning(f"✗ Profile assessment failed: {response['error']}")
            
            return response
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Behavioral profile assessment failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'data_source': 'Error',
                'maturity_level': 'Unknown',
                'overall_risk_score': 50.0,
                'dominant_biases': [],
                'recommendations': [],
                'execution_time': round(execution_time, 3)
            }

    # Keep all other methods (detect_specific_biases, get_portfolio_behavioral_context, etc.)
    # with similar updates...

    async def detect_specific_biases(
        self,
        conversation_messages: Union[List[Any], List[Dict[str, str]]],
        bias_types: List[str],
        symbols: Optional[List[str]] = None,
        period: Union[str, Any, None] = None,
        use_real_data: bool = True
    ) -> Dict[str, Any]:
        """UPDATED: Targeted bias detection with centralized model support"""
        try:
            logger.info(f"Detecting specific biases: {bias_types}")
            
            # Get comprehensive analysis first (handles normalization internally)
            full_analysis = await self.comprehensive_behavioral_analysis(
                conversation_messages, symbols or [], period, use_real_data
            )
            
            # Extract success and error handling for both response types
            if hasattr(full_analysis, 'success'):
                # Centralized response model
                success = full_analysis.success
                error = full_analysis.error
            else:
                # Legacy response
                success = full_analysis.success
                error = full_analysis.error
            
            if not success:
                return {
                    'success': False,
                    'error': error,
                    'targeted_biases': bias_types,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            
            # Return analysis with bias filtering metadata
            return {
                'success': True,
                'targeted_biases': bias_types,
                'data_source': full_analysis.data_source,
                'bias_count': getattr(full_analysis, 'bias_count', 0),
                'overall_risk_score': getattr(full_analysis, 'overall_risk_score', 50.0),
                'fmp_integration_used': 'FMP' in full_analysis.data_source,
                'recommendations': getattr(full_analysis, 'recommendations', []),
                'execution_time': getattr(full_analysis, 'execution_time', 0.0),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Specific bias detection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'targeted_biases': bias_types,
                'analysis_timestamp': datetime.now().isoformat()
            }

    def get_service_status(self) -> Dict[str, Any]:
        """UPDATED: Get comprehensive service status including centralized model support"""
        return {
            'service_name': self.service_name,
            'version': self.version,
            'status': 'operational' if self.tools_available else 'degraded',
            'capabilities': {
                'behavioral_tools_available': self.tools_available,
                'fmp_integration_available': self.fmp_available,
                'centralized_models_support': self.centralized_models_available,
                'async_analysis_supported': True,
                'portfolio_context_analysis': self.fmp_available,
                'real_market_data': self.fmp_available,
                'backward_compatibility': True
            },
            'configuration': {
                'risk_free_rate': self.risk_free_rate,
                'confidence_levels': self.confidence_levels,
                'default_period': self.default_period
            },
            'supported_input_formats': [
                'ConversationMessage objects (centralized models)',
                'Dict messages (legacy format)',
                'AnalysisPeriod enums (centralized models)',
                'String periods (legacy format)'
            ],
            'response_formats': [
                'BehavioralAnalysisResponse (centralized models)' if self.centralized_models_available else 'Not Available',
                'Legacy response dictionaries (backward compatibility)'
            ],
            'data_sources': [
                'Conversation Analysis',
                'FMP Market Data' if self.fmp_available else 'Not Available',
                'Portfolio Performance Context' if self.fmp_available else 'Not Available'
            ],
            'timestamp': datetime.now().isoformat()
        }

    async def health_check(self) -> Dict[str, Any]:
        """UPDATED: Health check testing both legacy and centralized model formats"""
        start_time = datetime.now()
        
        try:
            # Test with legacy dict format
            test_messages_dict = [{"role": "user", "content": "Test message for health check"}]
            test_symbols = ["AAPL"]
            
            # Test basic analysis (legacy format)
            basic_test = await self.comprehensive_behavioral_analysis(
                test_messages_dict, [], use_real_data=False
            )
            
            basic_health = getattr(basic_test, 'success', False)
            
            # Test with centralized models if available
            centralized_health = True
            if CENTRALIZED_MODELS_AVAILABLE:
                try:
                    # Create ConversationMessage objects
                    test_message_objects = [ConversationMessage(role="user", content="Test message for centralized models")]
                    
                    centralized_test = await self.comprehensive_behavioral_analysis(
                        test_message_objects, test_symbols, AnalysisPeriod.SIX_MONTHS, use_real_data=False
                    )
                    centralized_health = getattr(centralized_test, 'success', False)
                except Exception as e:
                    logger.warning(f"Centralized models health check failed: {e}")
                    centralized_health = False
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            overall_health = 'healthy' if (basic_health and centralized_health) else 'unhealthy'
            if basic_health and not centralized_health:
                overall_health = 'degraded'
            
            return {
                'status': overall_health,
                'service': self.service_name,
                'version': self.version,
                'checks': {
                    'basic_analysis': basic_health,
                    'centralized_models_support': centralized_health,
                    'tools_import': self.tools_available,
                    'fmp_integration': self.fmp_available,
                    'data_manager': self.fmp_available
                },
                'execution_time': round(execution_time, 3),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Health check failed: {e}")
            
            return {
                'status': 'unhealthy',
                'service': self.service_name,
                'error': str(e),
                'execution_time': round(execution_time, 3),
                'timestamp': datetime.now().isoformat()
            }

# Testing function updated for centralized models
async def test_centralized_models_integration():
    """Test integration with centralized models"""
    logger.info("=" * 50)
    logger.info("BEHAVIORAL SERVICE CENTRALIZED MODELS TEST")
    logger.info("=" * 50)
    
    service = BehavioralAnalysisService()
    
    # Test 1: Legacy format (backward compatibility)
    logger.info("1. Testing legacy dict format...")
    legacy_messages = [{"role": "user", "content": "I'm worried about market volatility"}]
    
    try:
        legacy_result = await service.comprehensive_behavioral_analysis(
            legacy_messages, ["AAPL"], "6month", use_real_data=False
        )
        logger.info(f"✓ Legacy format test: {getattr(legacy_result, 'success', False)}")
    except Exception as e:
        logger.error(f"✗ Legacy format test failed: {e}")
    
    # Test 2: Centralized models format (if available)
    if CENTRALIZED_MODELS_AVAILABLE and ConversationMessage and AnalysisPeriod:
        logger.info("2. Testing centralized models format...")
        try:
            centralized_messages = [ConversationMessage(role="user", content="I'm worried about market volatility")]
            
            centralized_result = await service.comprehensive_behavioral_analysis(
                centralized_messages, ["AAPL"], AnalysisPeriod.SIX_MONTHS, use_real_data=False
            )
            logger.info(f"✓ Centralized models test: {getattr(centralized_result, 'success', False)}")
        except Exception as e:
            logger.error(f"✗ Centralized models test failed: {e}")
    else:
        logger.warning("2. Centralized models not available - skipping test")
    
    # Test 3: Service status
    logger.info("3. Testing service status...")
    status = service.get_service_status()
    logger.info(f"✓ Service status: {status['status']}")
    logger.info(f"  - Centralized models support: {status['capabilities']['centralized_models_support']}")
    logger.info(f"  - Backward compatibility: {status['capabilities']['backward_compatibility']}")
    
    logger.info("=" * 50)
    logger.info("CENTRALIZED MODELS INTEGRATION TEST COMPLETE")
    logger.info("=" * 50)

if __name__ == "__main__":
    asyncio.run(test_centralized_models_integration())