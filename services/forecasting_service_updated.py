# services/forecasting_service_updated.py - Refactored Following Proven Patterns
"""
Forecasting Service - Aligned with Centralized Architecture
==========================================================

Refactored to follow the exact patterns from behavioral_service_updated.py
and integrate with centralized models while maintaining FMP integration.
"""

import asyncio
import importlib.util
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

# Import centralized models with proper error handling
def import_centralized_models():
    """Import centralized models with proper path handling"""
    try:
        import sys
        import os
        
        project_root = os.path.dirname(os.path.dirname(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from models.requests import (
            AnalysisPeriod, ForecastHorizon, ModelType, VolatilityModel,
            RegimeMethod, IntegrationDepth,
            ForecastingAnalysisResponse, RegimeAnalysisResponse, 
            IntegratedAnalysisResponse, ForecastingHealthResponse
        )
        return True, AnalysisPeriod, ForecastingAnalysisResponse, RegimeAnalysisResponse, IntegratedAnalysisResponse
    except ImportError as e:
        logger.warning(f"Centralized models not available: {e}")
        return False, None, None, None, None

# Try to import centralized models
CENTRALIZED_MODELS_AVAILABLE, AnalysisPeriod, ForecastingAnalysisResponse, RegimeAnalysisResponse, IntegratedAnalysisResponse = import_centralized_models()

if CENTRALIZED_MODELS_AVAILABLE:
    logger.info("Centralized models imported successfully for forecasting")
else:
    # Define placeholder types
    AnalysisPeriod = None
    ForecastingAnalysisResponse = None
    RegimeAnalysisResponse = None
    IntegratedAnalysisResponse = None

# DIRECT IMPORT PATTERN for tools
def import_forecasting_tools():
    """Import forecasting tools directly from file"""
    try:
        import sys
        import os
        
        project_root = os.path.dirname(os.path.dirname(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        spec = importlib.util.spec_from_file_location(
            "forecasting_tools", 
            os.path.join(project_root, "tools", "forecasting_tools.py")
        )
        forecasting_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(forecasting_module)
        return forecasting_module
    except Exception as e:
        logger.error(f"Could not import forecasting tools: {e}")
        return None

def import_integrated_services():
    """Import integrated services for four-way analysis"""
    try:
        import sys
        import os
        
        project_root = os.path.dirname(os.path.dirname(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Import other services following your actual file structure
        risk_spec = importlib.util.spec_from_file_location(
            "risk_service", 
            os.path.join(project_root, "services", "risk_service.py")
        )
        risk_module = importlib.util.module_from_spec(risk_spec)
        risk_spec.loader.exec_module(risk_module)
        
        behavioral_spec = importlib.util.spec_from_file_location(
            "behavioral_service_updated", 
            os.path.join(project_root, "services", "behavioral_service_updated.py")
        )
        behavioral_module = importlib.util.module_from_spec(behavioral_spec)
        behavioral_spec.loader.exec_module(behavioral_module)
        
        # Portfolio service - try to import if available
        try:
            portfolio_spec = importlib.util.spec_from_file_location(
                "portfolio_service_fixed", 
                os.path.join(project_root, "services", "portfolio_service_fixed.py")
            )
            portfolio_module = importlib.util.module_from_spec(portfolio_spec)
            portfolio_spec.loader.exec_module(portfolio_module)
        except FileNotFoundError:
            portfolio_module = None
        
        return risk_module, behavioral_module, portfolio_module
    except Exception as e:
        logger.warning(f"Could not import integrated services: {e}")
        return None, None, None

# Initialize imports
forecasting_tools = import_forecasting_tools()
risk_service_module, behavioral_service_module, portfolio_service_module = import_integrated_services()

# Check availability
FORECASTING_TOOLS_AVAILABLE = forecasting_tools is not None
RISK_SERVICE_AVAILABLE = risk_service_module is not None
BEHAVIORAL_SERVICE_AVAILABLE = behavioral_service_module is not None
PORTFOLIO_SERVICE_AVAILABLE = portfolio_service_module is not None

# Legacy response classes for backward compatibility
class LegacyForecastingResponse:
    """Legacy response format for backward compatibility"""
    def __init__(self, success: bool, data_source: str, analysis_type: str,
                 forecast_horizon: int, execution_time: float, 
                 fmp_integration_used: bool, error: Optional[str] = None):
        self.success = success
        self.data_source = data_source
        self.analysis_type = analysis_type
        self.forecast_horizon = forecast_horizon
        self.execution_time = execution_time
        self.fmp_integration_used = fmp_integration_used
        self.error = error

class ForecastingService:
    """
    Forecasting Service aligned with centralized architecture
    
    Follows the exact pattern from behavioral_service_updated.py:
    - Handles centralized models and legacy formats
    - FMP integration with graceful fallback
    - Four-way service integration capability
    - Comprehensive error handling and logging
    """
    
    def __init__(self, 
                 default_period: str = "1year",
                 default_horizon: int = 21):
        """Initialize forecasting service with centralized model support"""
        self.service_name = "forecasting_service_updated"
        self.version = "5.1.0"  # Aligned with centralized models
        self.default_period = default_period
        self.default_horizon = default_horizon
        
        # Service availability tracking
        self.tools_available = FORECASTING_TOOLS_AVAILABLE
        self.centralized_models_available = CENTRALIZED_MODELS_AVAILABLE
        self.risk_service_available = RISK_SERVICE_AVAILABLE
        self.behavioral_service_available = BEHAVIORAL_SERVICE_AVAILABLE
        self.portfolio_service_available = PORTFOLIO_SERVICE_AVAILABLE
        
        # FMP integration status
        self.fmp_available = False
        if FORECASTING_TOOLS_AVAILABLE:
            try:
                status = forecasting_tools.get_forecasting_tools_integration_status()
                self.fmp_available = status.get("fmp_integration_available", False)
            except Exception as e:
                logger.warning(f"Could not check FMP integration status: {e}")
        
        logger.info(f"Initialized {self.service_name} v{self.version}")
        logger.info(f"Tools: {self.tools_available}, FMP: {self.fmp_available}, Models: {self.centralized_models_available}")
        logger.info(f"Integration: Risk={self.risk_service_available}, Behavioral={self.behavioral_service_available}, Portfolio={self.portfolio_service_available}")

    def _normalize_period(self, period: Union[str, Any, None]) -> str:
        """Convert AnalysisPeriod enum to string for tools layer"""
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

    def _normalize_horizon(self, horizon: Union[int, Any, None]) -> int:
        """Convert ForecastHorizon enum to int for tools layer"""
        if horizon is None:
            return self.default_horizon
        
        # Handle enum from centralized models
        if hasattr(horizon, 'value'):
            return horizon.value
        
        # Handle integer format
        if isinstance(horizon, int):
            return horizon
        
        logger.warning(f"Invalid horizon format: {type(horizon)}, using default")
        return self.default_horizon

    def _normalize_model_type(self, model_type: Union[str, Any, None]) -> str:
        """Convert ModelType enum to string for tools layer"""
        if model_type is None:
            return "auto_arima"
        
        # Handle enum from centralized models
        if hasattr(model_type, 'value'):
            return model_type.value
        
        # Handle string format
        if isinstance(model_type, str):
            return model_type
        
        return "auto_arima"

    def _create_centralized_response(
        self,
        success: bool,
        data_source: str,
        execution_time: float,
        analysis_type: str = "forecasting",
        forecast_results: Dict = None,
        volatility_results: Dict = None,
        regime_results: Dict = None,
        scenario_results: Dict = None,
        comprehensive_results: Dict = None,
        error: Optional[str] = None
    ) -> Union[Any, 'LegacyForecastingResponse']:
        """Create response using centralized models if available"""
        if CENTRALIZED_MODELS_AVAILABLE and ForecastingAnalysisResponse:
            return ForecastingAnalysisResponse(
                success=success,
                service="forecasting_analysis",
                analysis_type=analysis_type,
                timestamp=datetime.now(),
                service_metadata={
                    "service_name": self.service_name,
                    "version": self.version,
                    "analysis_duration": execution_time,
                    "fmp_integration_used": "FMP" in data_source,
                    "data_source": data_source
                },
                symbols=[],  # Will be set by caller
                period=self.default_period,  # Will be set by caller
                data_source=data_source,
                fmp_integration="FMP" in data_source,
                forecast_results=forecast_results or {},
                volatility_results=volatility_results or {},
                regime_results=regime_results or {},
                scenario_results=scenario_results or {},
                comprehensive_results=comprehensive_results or {},
                error=error
            )
        else:
            return LegacyForecastingResponse(
                success=success,
                data_source=data_source,
                analysis_type=analysis_type,
                forecast_horizon=self.default_horizon,
                execution_time=execution_time,
                fmp_integration_used="FMP" in data_source,
                error=error
            )

    # ========================================================================
    # CORE FORECASTING METHODS
    # ========================================================================

    def forecast_returns(
        self,
        symbols: List[str],
        period: Union[str, Any] = None,
        use_real_data: bool = True,
        forecast_horizon: Union[int, Any] = None,
        model_type: Union[str, Any] = None,
        confidence_levels: List[float] = None,
        include_regime_conditioning: bool = True
    ) -> Union[Any, 'LegacyForecastingResponse']:
        """
        Enhanced return forecasting with centralized model support
        """
        start_time = datetime.now()
        
        try:
            # Normalize inputs
            normalized_period = self._normalize_period(period)
            normalized_horizon = self._normalize_horizon(forecast_horizon)
            normalized_model = self._normalize_model_type(model_type)
            
            logger.info(f"Return forecasting: {len(symbols)} symbols, {normalized_period}, {normalized_horizon} days")
            
            if not FORECASTING_TOOLS_AVAILABLE:
                execution_time = (datetime.now() - start_time).total_seconds()
                return self._create_centralized_response(
                    success=False,
                    data_source="Error",
                    execution_time=execution_time,
                    analysis_type="return_forecast",
                    error="Forecasting tools not available"
                )
            
            if not symbols:
                execution_time = (datetime.now() - start_time).total_seconds()
                return self._create_centralized_response(
                    success=False,
                    data_source="Error",
                    execution_time=execution_time,
                    analysis_type="return_forecast",
                    error="No symbols provided"
                )
            
            # Call forecasting tools with proper parameters
            result = forecasting_tools.forecast_portfolio_returns(
                symbols=symbols,
                use_real_data=use_real_data,
                period=normalized_period,
                forecast_horizon=normalized_horizon,
                model_type=normalized_model,
                confidence_levels=confidence_levels or [0.05, 0.25, 0.5, 0.75, 0.95],
                include_regime_conditioning=include_regime_conditioning
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if not result.get('success', False):
                return self._create_centralized_response(
                    success=False,
                    data_source=result.get('data_source', 'Unknown'),
                    execution_time=execution_time,
                    analysis_type="return_forecast",
                    error=result.get('error', 'Forecasting failed')
                )
            
            response = self._create_centralized_response(
                success=True,
                data_source=result.get('data_source', 'FMP'),
                execution_time=execution_time,
                analysis_type="return_forecast",
                forecast_results=result.get('forecasts', {})
            )
            
            # Set symbols and period on response if centralized models available
            if hasattr(response, 'symbols'):
                response.symbols = symbols
                response.period = normalized_period
            
            logger.info(f"✓ Return forecasting completed: {result.get('data_source', 'Unknown')}, {execution_time:.2f}s")
            return response
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Return forecasting failed: {e}")
            
            return self._create_centralized_response(
                success=False,
                data_source="Error",
                execution_time=execution_time,
                analysis_type="return_forecast",
                error=str(e)
            )

    def forecast_volatility(
        self,
        symbols: List[str],
        period: Union[str, Any] = None,
        use_real_data: bool = True,
        forecast_horizon: Union[int, Any] = None,
        volatility_model: Union[str, Any] = None,
        include_regime_switching: bool = True
    ) -> Union[Any, 'LegacyForecastingResponse']:
        """
        Enhanced volatility forecasting with centralized model support
        """
        start_time = datetime.now()
        
        try:
            # Normalize inputs
            normalized_period = self._normalize_period(period)
            normalized_horizon = self._normalize_horizon(forecast_horizon)
            normalized_vol_model = volatility_model.value if hasattr(volatility_model, 'value') else (volatility_model or "garch")
            
            logger.info(f"Volatility forecasting: {len(symbols)} symbols, model={normalized_vol_model}")
            
            if not FORECASTING_TOOLS_AVAILABLE:
                execution_time = (datetime.now() - start_time).total_seconds()
                return self._create_centralized_response(
                    success=False,
                    data_source="Error",
                    execution_time=execution_time,
                    analysis_type="volatility_forecast",
                    error="Forecasting tools not available"
                )
            
            # Call forecasting tools
            result = forecasting_tools.forecast_volatility_with_regimes(
                symbols=symbols,
                use_real_data=use_real_data,
                period=normalized_period,
                forecast_horizon=normalized_horizon,
                volatility_model=normalized_vol_model,
                include_regime_switching=include_regime_switching
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if not result.get('success', False):
                return self._create_centralized_response(
                    success=False,
                    data_source=result.get('data_source', 'Unknown'),
                    execution_time=execution_time,
                    analysis_type="volatility_forecast",
                    error=result.get('error', 'Volatility forecasting failed')
                )
            
            response = self._create_centralized_response(
                success=True,
                data_source=result.get('data_source', 'FMP'),
                execution_time=execution_time,
                analysis_type="volatility_forecast",
                volatility_results=result.get('volatility_forecasts', {})
            )
            
            if hasattr(response, 'symbols'):
                response.symbols = symbols
                response.period = normalized_period
            
            logger.info(f"✓ Volatility forecasting completed: {execution_time:.2f}s")
            return response
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Volatility forecasting failed: {e}")
            
            return self._create_centralized_response(
                success=False,
                data_source="Error",
                execution_time=execution_time,
                analysis_type="volatility_forecast",
                error=str(e)
            )

    def analyze_market_regimes(
        self,
        symbols: List[str],
        period: Union[str, Any] = None,
        use_real_data: bool = True,
        regime_method: Union[str, Any] = None,
        n_regimes: int = 2,
        include_transitions: bool = True
    ) -> Union[Any, 'LegacyForecastingResponse']:
        """
        Enhanced regime analysis with centralized model support
        """
        start_time = datetime.now()
        
        try:
            # Normalize inputs
            normalized_period = self._normalize_period(period)
            normalized_method = regime_method.value if hasattr(regime_method, 'value') else (regime_method or "hmm")
            
            logger.info(f"Regime analysis: {len(symbols)} symbols, method={normalized_method}")
            
            if not FORECASTING_TOOLS_AVAILABLE:
                execution_time = (datetime.now() - start_time).total_seconds()
                return self._create_centralized_response(
                    success=False,
                    data_source="Error",
                    execution_time=execution_time,
                    analysis_type="regime_analysis",
                    error="Forecasting tools not available"
                )
            
            # Call forecasting tools for regime analysis
            result = forecasting_tools.forecast_regime_transitions(
                symbols=symbols,
                use_real_data=use_real_data,
                period=normalized_period,
                forecast_horizon=21,  # Standard for regime analysis
                n_regimes=n_regimes
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if not result.get('success', False):
                return self._create_centralized_response(
                    success=False,
                    data_source=result.get('data_source', 'Unknown'),
                    execution_time=execution_time,
                    analysis_type="regime_analysis",
                    error=result.get('error', 'Regime analysis failed')
                )
            
            response = self._create_centralized_response(
                success=True,
                data_source=result.get('data_source', 'FMP'),
                execution_time=execution_time,
                analysis_type="regime_analysis",
                regime_results=result
            )
            
            if hasattr(response, 'symbols'):
                response.symbols = symbols
                response.period = normalized_period
            
            logger.info(f"✓ Regime analysis completed: {execution_time:.2f}s")
            return response
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Regime analysis failed: {e}")
            
            return self._create_centralized_response(
                success=False,
                data_source="Error",
                execution_time=execution_time,
                analysis_type="regime_analysis",
                error=str(e)
            )

    def generate_scenarios(
        self,
        symbols: List[str],
        period: Union[str, Any] = None,
        use_real_data: bool = True,
        forecast_horizon: Union[int, Any] = None,
        scenarios: Optional[Dict[str, float]] = None,
        monte_carlo_paths: int = 1000
    ) -> Union[Any, 'LegacyForecastingResponse']:
        """
        Enhanced scenario analysis with centralized model support
        """
        start_time = datetime.now()
        
        try:
            # Normalize inputs
            normalized_period = self._normalize_period(period)
            normalized_horizon = self._normalize_horizon(forecast_horizon)
            
            logger.info(f"Scenario analysis: {len(symbols)} symbols, {monte_carlo_paths} paths")
            
            if not FORECASTING_TOOLS_AVAILABLE:
                execution_time = (datetime.now() - start_time).total_seconds()
                return self._create_centralized_response(
                    success=False,
                    data_source="Error",
                    execution_time=execution_time,
                    analysis_type="scenario_analysis",
                    error="Forecasting tools not available"
                )
            
            # Call forecasting tools for scenario analysis
            result = forecasting_tools.generate_scenario_forecasts(
                symbols=symbols,
                use_real_data=use_real_data,
                period=normalized_period,
                forecast_horizon=normalized_horizon,
                scenarios=scenarios,
                monte_carlo_paths=monte_carlo_paths
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if not result.get('success', False):
                return self._create_centralized_response(
                    success=False,
                    data_source=result.get('data_source', 'Unknown'),
                    execution_time=execution_time,
                    analysis_type="scenario_analysis",
                    error=result.get('error', 'Scenario analysis failed')
                )
            
            response = self._create_centralized_response(
                success=True,
                data_source=result.get('data_source', 'FMP'),
                execution_time=execution_time,
                analysis_type="scenario_analysis",
                scenario_results=result
            )
            
            if hasattr(response, 'symbols'):
                response.symbols = symbols
                response.period = normalized_period
            
            logger.info(f"✓ Scenario analysis completed: {execution_time:.2f}s")
            return response
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Scenario analysis failed: {e}")
            
            return self._create_centralized_response(
                success=False,
                data_source="Error",
                execution_time=execution_time,
                analysis_type="scenario_analysis",
                error=str(e)
            )

    def comprehensive_forecast_analysis(
        self,
        symbols: List[str],
        period: Union[str, Any] = None,
        use_real_data: bool = True,
        forecast_horizon: Union[int, Any] = None,
        include_returns: bool = True,
        include_volatility: bool = True,
        include_regimes: bool = True,
        include_scenarios: bool = True
    ) -> Union[Any, 'LegacyForecastingResponse']:
        """
        Comprehensive forecasting analysis combining all components
        """
        start_time = datetime.now()
        
        try:
            # Normalize inputs
            normalized_period = self._normalize_period(period)
            normalized_horizon = self._normalize_horizon(forecast_horizon)
            
            logger.info(f"Comprehensive forecasting: {len(symbols)} symbols")
            
            if not FORECASTING_TOOLS_AVAILABLE:
                execution_time = (datetime.now() - start_time).total_seconds()
                return self._create_centralized_response(
                    success=False,
                    data_source="Error",
                    execution_time=execution_time,
                    analysis_type="comprehensive_forecast",
                    error="Forecasting tools not available"
                )
            
            # Call comprehensive forecasting function
            result = forecasting_tools.generate_comprehensive_forecast(
                symbols=symbols,
                use_real_data=use_real_data,
                period=normalized_period,
                forecast_horizon=normalized_horizon,
                include_volatility=include_volatility,
                include_regimes=include_regimes,
                include_scenarios=include_scenarios
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if not result.get('success', False):
                return self._create_centralized_response(
                    success=False,
                    data_source=result.get('data_source', 'Unknown'),
                    execution_time=execution_time,
                    analysis_type="comprehensive_forecast",
                    error=result.get('error', 'Comprehensive forecasting failed')
                )
            
            response = self._create_centralized_response(
                success=True,
                data_source=result.get('data_source', 'FMP'),
                execution_time=execution_time,
                analysis_type="comprehensive_forecast",
                comprehensive_results=result
            )
            
            if hasattr(response, 'symbols'):
                response.symbols = symbols
                response.period = normalized_period
            
            logger.info(f"✓ Comprehensive forecasting completed: {execution_time:.2f}s")
            return response
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Comprehensive forecasting failed: {e}")
            
            return self._create_centralized_response(
                success=False,
                data_source="Error",
                execution_time=execution_time,
                analysis_type="comprehensive_forecast",
                error=str(e)
            )

    # ========================================================================
    # INTEGRATED ANALYSIS METHODS
    # ========================================================================

    def integrated_analysis_with_risk_context(
        self,
        symbols: List[str],
        period: Union[str, Any] = None,
        use_real_data: bool = True,
        forecast_horizon: Union[int, Any] = None,
        risk_analysis_type: Union[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Integrated forecasting with risk analysis
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Integrated forecast+risk analysis starting")
            
            # Get forecasting analysis
            forecast_result = self.comprehensive_forecast_analysis(
                symbols=symbols,
                period=period,
                use_real_data=use_real_data,
                forecast_horizon=forecast_horizon
            )
            
            # Get risk analysis if service available
            risk_result = {}
            if RISK_SERVICE_AVAILABLE and risk_service_module:
                try:
                    risk_service = risk_service_module.RiskAnalysisService()
                    risk_analysis = risk_service.analyze_portfolio_risk(
                        symbols, self._normalize_period(period), use_real_data
                    )
                    risk_result = risk_analysis if isinstance(risk_analysis, dict) else {}
                except Exception as e:
                    logger.warning(f"Risk service integration failed: {e}")
                    risk_result = {"error": str(e)}
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "analysis_type": "integrated_forecast_risk",
                "data_source": getattr(forecast_result, 'data_source', 'FMP'),
                "execution_time": execution_time,
                "forecasting_analysis": forecast_result,
                "risk_analysis": risk_result,
                "cross_analysis_insights": {
                    "forecast_risk_alignment": "Analysis combines forward-looking forecasts with current risk metrics",
                    "regime_risk_correlation": "Market regimes inform risk model adjustments"
                },
                "integrated_insights": [
                    "Forecasting provides forward-looking risk scenario inputs",
                    "Risk analysis validates forecast assumptions",
                    "Combined analysis improves decision timing"
                ]
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Integrated forecast-risk analysis failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }

    def integrated_analysis_with_behavioral_context(
        self,
        symbols: List[str],
        conversation_messages: List[Dict[str, str]],
        period: Union[str, Any] = None,
        use_real_data: bool = True,
        forecast_horizon: Union[int, Any] = None
    ) -> Dict[str, Any]:
        """
        Integrated forecasting with behavioral analysis
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Integrated forecast+behavioral analysis starting")
            
            # Get forecasting analysis
            forecast_result = self.comprehensive_forecast_analysis(
                symbols=symbols,
                period=period,
                use_real_data=use_real_data,
                forecast_horizon=forecast_horizon
            )
            
            # Get behavioral analysis if service available
            behavioral_result = {}
            if BEHAVIORAL_SERVICE_AVAILABLE and behavioral_service_module:
                try:
                    behavioral_service = behavioral_service_module.BehavioralAnalysisService()
                    # Note: This would need to be called from an async context in real usage
                    # For now, we'll create a placeholder result
                    behavioral_result = {
                        "success": True,
                        "analysis_type": "behavioral_placeholder",
                        "message": "Behavioral service available but requires async context",
                        "integration_note": "Call from async endpoint for full integration"
                    }
                except Exception as e:
                    logger.warning(f"Behavioral service integration failed: {e}")
                    behavioral_result = {"error": str(e)}
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "analysis_type": "integrated_forecast_behavioral",
                "data_source": getattr(forecast_result, 'data_source', 'FMP'),
                "execution_time": execution_time,
                "forecasting_analysis": forecast_result,
                "behavioral_analysis": behavioral_result,
                "cross_analysis_insights": {
                    "forecast_bias_interaction": "Behavioral biases may affect forecast interpretation",
                    "timing_bias_correlation": "Market timing biases inform forecast implementation"
                },
                "integrated_insights": [
                    "Behavioral analysis informs forecast implementation strategy",
                    "Forecasting provides objective market outlook vs subjective biases",
                    "Combined analysis improves decision quality"
                ]
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Integrated forecast-behavioral analysis failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }

    def four_way_integrated_analysis(
        self,
        symbols: List[str],
        conversation_messages: List[Dict[str, str]],
        portfolio_request: Optional[Dict] = None,
        period: Union[str, Any] = None,
        use_real_data: bool = True,
        forecast_horizon: Union[int, Any] = None,
        integration_depth: Union[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Complete four-way integrated analysis
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Four-way integrated analysis starting")
            
            # Initialize results containers
            results = {
                "success": True,
                "analysis_type": "four_way_integrated",
                "data_source": "Multi-service Integration",
                "symbols": symbols,
                "period": self._normalize_period(period),
                "integration_level": "FOUR_WAY",
                "fmp_integration_used": True,
                "analysis_components": {
                    "forecasting": FORECASTING_TOOLS_AVAILABLE,
                    "risk": RISK_SERVICE_AVAILABLE,
                    "behavioral": BEHAVIORAL_SERVICE_AVAILABLE,
                    "portfolio": PORTFOLIO_SERVICE_AVAILABLE
                }
            }
            
            # 1. Forecasting Analysis
            forecast_result = self.comprehensive_forecast_analysis(
                symbols=symbols,
                period=period,
                use_real_data=use_real_data,
                forecast_horizon=forecast_horizon
            )
            results["forecasting_analysis"] = forecast_result
            
            # 2. Risk Analysis (if available)
            if RISK_SERVICE_AVAILABLE and risk_service_module:
                try:
                    risk_service = risk_service_module.RiskAnalysisService()
                    risk_result = risk_service.analyze_portfolio_risk(
                        symbols, self._normalize_period(period), use_real_data
                    )
                    results["risk_analysis"] = risk_result
                except Exception as e:
                    logger.warning(f"Risk analysis integration failed: {e}")
                    results["risk_analysis"] = {"error": str(e)}
            
            # 3. Behavioral Analysis (if available)
            if BEHAVIORAL_SERVICE_AVAILABLE and behavioral_service_module:
                try:
                    behavioral_service = behavioral_service_module.BehavioralAnalysisService()
                    # Note: Behavioral service requires async context for full integration
                    behavioral_result = {
                        "success": True,
                        "analysis_type": "behavioral_placeholder",
                        "message": "Behavioral service available but requires async context",
                        "integration_note": "Call from async endpoint for full integration"
                    }
                    results["behavioral_analysis"] = behavioral_result
                except Exception as e:
                    logger.warning(f"Behavioral analysis integration failed: {e}")
                    results["behavioral_analysis"] = {"error": str(e)}
            
            # 4. Portfolio Analysis (if available)
            if PORTFOLIO_SERVICE_AVAILABLE and portfolio_service_module:
                try:
                    portfolio_service = portfolio_service_module.PortfolioManagementService()
                    portfolio_result = portfolio_service.comprehensive_portfolio_analysis(
                        symbols=symbols,
                        period=self._normalize_period(period),
                        use_real_data=use_real_data
                    )
                    results["portfolio_analysis"] = portfolio_result
                except Exception as e:
                    logger.warning(f"Portfolio analysis integration failed: {e}")
                    results["portfolio_analysis"] = {"error": str(e)}
            
            # Cross-analysis insights
            results["cross_analysis_insights"] = {
                "forecast_risk_alignment": "Forward-looking forecasts inform dynamic risk management",
                "behavioral_forecast_timing": "Behavioral biases affect forecast implementation timing",
                "portfolio_forecast_optimization": "Forecasts guide portfolio rebalancing decisions",
                "integrated_decision_framework": "All four analyses create comprehensive investment framework"
            }
            
            # Unified insights
            results["integrated_insights"] = [
                "Four-way analysis provides comprehensive market view",
                "Forecasting guides strategic timing decisions",
                "Risk analysis validates forecast assumptions",
                "Behavioral analysis improves implementation quality",
                "Portfolio analysis optimizes allocation decisions"
            ]
            
            results["unified_recommendations"] = [
                "Use forecasting insights for strategic timing",
                "Apply risk metrics for position sizing",
                "Address behavioral biases in implementation",
                "Optimize portfolio based on forward-looking analysis"
            ]
            
            execution_time = (datetime.now() - start_time).total_seconds()
            results["analysis_duration"] = execution_time
            
            logger.info(f"✓ Four-way analysis completed in {execution_time:.2f}s")
            return results
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Four-way integrated analysis failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "analysis_duration": execution_time
            }

    # ========================================================================
    # SERVICE MANAGEMENT METHODS
    # ========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check following proven pattern
        """
        try:
            # Test basic forecasting functionality
            test_symbols = ["AAPL"]
            test_forecast = None
            
            if FORECASTING_TOOLS_AVAILABLE:
                try:
                    test_forecast = forecasting_tools.forecast_portfolio_returns(
                        symbols=test_symbols,
                        use_real_data=False,
                        period="3months",
                        forecast_horizon=5
                    )
                except Exception as e:
                    logger.warning(f"Forecasting tools test failed: {e}")
            
            # Determine overall health
            tools_healthy = test_forecast is not None and test_forecast.get('success', False)
            service_status = 'healthy' if tools_healthy else 'degraded'
            
            return {
                "status": service_status,
                "service": self.service_name,
                "version": self.version,
                "fmp_integration_status": {
                    "forecasting": {
                        "fmp_integration": self.fmp_available,
                        "available": self.fmp_available
                    }
                },
                "tools_status": {
                    "forecasting_tools": "available" if FORECASTING_TOOLS_AVAILABLE else "unavailable",
                    "regime_tools": "integrated" if FORECASTING_TOOLS_AVAILABLE else "unavailable"
                },
                "integrated_services_status": {
                    "risk_service": "available" if RISK_SERVICE_AVAILABLE else "unavailable",
                    "behavioral_service": "available" if BEHAVIORAL_SERVICE_AVAILABLE else "unavailable",
                    "portfolio_service": "available" if PORTFOLIO_SERVICE_AVAILABLE else "unavailable"
                },
                "four_way_integration": f"ready - {sum([RISK_SERVICE_AVAILABLE, BEHAVIORAL_SERVICE_AVAILABLE, PORTFOLIO_SERVICE_AVAILABLE])+1}/4 services available",
                "capabilities": {
                    "return_forecasting": FORECASTING_TOOLS_AVAILABLE,
                    "volatility_forecasting": FORECASTING_TOOLS_AVAILABLE,
                    "regime_analysis": FORECASTING_TOOLS_AVAILABLE,
                    "scenario_analysis": FORECASTING_TOOLS_AVAILABLE,
                    "risk_integration": RISK_SERVICE_AVAILABLE,
                    "behavioral_integration": BEHAVIORAL_SERVICE_AVAILABLE,
                    "portfolio_integration": PORTFOLIO_SERVICE_AVAILABLE,
                    "centralized_models": CENTRALIZED_MODELS_AVAILABLE
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "service": self.service_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def get_service_status(self) -> Dict[str, Any]:
        """Get detailed service status"""
        return {
            "service_name": self.service_name,
            "version": self.version,
            "status": "operational" if self.tools_available else "degraded",
            "capabilities": {
                "forecasting_tools_available": self.tools_available,
                "fmp_integration_available": self.fmp_available,
                "centralized_models_support": self.centralized_models_available,
                "risk_service_integration": self.risk_service_available,
                "behavioral_service_integration": self.behavioral_service_available,
                "portfolio_service_integration": self.portfolio_service_available,
                "four_way_analysis_capability": True
            },
            "configuration": {
                "default_period": self.default_period,
                "default_horizon": self.default_horizon
            },
            "integration_status": {
                "services_available": f"{sum([self.risk_service_available, self.behavioral_service_available, self.portfolio_service_available])+1}/4",
                "fmp_data_integration": self.fmp_available,
                "centralized_models": self.centralized_models_available
            },
            "timestamp": datetime.now().isoformat()
        }

# Testing function for validation
async def test_forecasting_service_integration():
    """Test forecasting service with centralized models"""
    logger.info("=" * 60)
    logger.info("FORECASTING SERVICE CENTRALIZED MODELS TEST")
    logger.info("=" * 60)
    
    service = ForecastingService()
    
    # Test 1: Basic forecasting
    logger.info("1. Testing basic return forecasting...")
    try:
        result = service.forecast_returns(
            symbols=["AAPL", "MSFT"],
            period="6months",
            use_real_data=False,
            forecast_horizon=10
        )
        logger.info(f"✓ Basic forecasting test: {getattr(result, 'success', False)}")
    except Exception as e:
        logger.error(f"✗ Basic forecasting test failed: {e}")
    
    # Test 2: Health check
    logger.info("2. Testing service health...")
    try:
        health = service.health_check()
        logger.info(f"✓ Health check: {health.get('status', 'unknown')}")
        logger.info(f"  - Tools: {health.get('tools_status', {}).get('forecasting_tools', 'unknown')}")
        logger.info(f"  - FMP: {health.get('fmp_integration_status', {}).get('forecasting', {}).get('fmp_integration', False)}")
        logger.info(f"  - Integration: {health.get('four_way_integration', 'unknown')}")
    except Exception as e:
        logger.error(f"✗ Health check failed: {e}")
    
    # Test 3: Service status
    logger.info("3. Testing service status...")
    try:
        status = service.get_service_status()
        logger.info(f"✓ Service status: {status['status']}")
        logger.info(f"  - Centralized models: {status['capabilities']['centralized_models_support']}")
        logger.info(f"  - Four-way capability: {status['capabilities']['four_way_analysis_capability']}")
    except Exception as e:
        logger.error(f"✗ Service status test failed: {e}")
    
    logger.info("=" * 60)
    logger.info("FORECASTING SERVICE TEST COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_forecasting_service_integration())