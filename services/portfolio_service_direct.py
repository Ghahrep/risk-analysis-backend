# services/portfolio_service_direct.py - Final Fix
"""
Portfolio Management Service - Final Working Version
===================================================

Fixed to work with your actual centralized models structure from models/requests.py
"""

from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import asyncio
import sys
import os
import importlib.util

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import centralized models (your actual models)
try:
    from models.requests import (
        PortfolioOptimizationRequest,
        RebalancingRequest,
        PortfolioRiskAnalysisRequest,
        PortfolioAnalysisResponse,
        OptimizationMethod,
        AnalysisPeriod
    )
    HAS_CENTRALIZED_MODELS = True
    print("✓ Successfully imported centralized models")
except ImportError as e:
    print(f"✗ Failed to import centralized models: {e}")
    print("Using fallback models...")
    HAS_CENTRALIZED_MODELS = False
    
    # Minimal fallback models that match your structure
    from pydantic import BaseModel, Field
    from enum import Enum
    from typing import Optional, Dict, Any, List
    
    class OptimizationMethod(str, Enum):
        MAX_SHARPE = "max_sharpe"
        MIN_VARIANCE = "min_variance"
        EQUAL_WEIGHT = "equal_weight"
        MAX_RETURN = "max_return"
        RISK_PARITY = "risk_parity"
        MAXIMIZE_DIVERSIFICATION = "maximize_diversification"
    
    class AnalysisPeriod(str, Enum):
        ONE_MONTH = "1month"
        THREE_MONTHS = "3months"
        SIX_MONTHS = "6months"
        ONE_YEAR = "1year"
    
    class PortfolioOptimizationRequest(BaseModel):
        portfolio_id: Optional[int] = None
        symbols: List[str] = Field(..., description="Stock symbols for optimization")
        period: AnalysisPeriod = AnalysisPeriod.ONE_YEAR
        optimization_method: OptimizationMethod = OptimizationMethod.MAX_SHARPE
        target_return: Optional[float] = None
        use_real_data: bool = True
        risk_tolerance: Optional[float] = 0.5
    
    class RebalancingRequest(BaseModel):
        portfolio_id: str
        current_holdings: Dict[str, float]
        target_allocation: Optional[Dict[str, float]] = None
        rebalance_threshold: float = 0.05
        period: AnalysisPeriod = AnalysisPeriod.ONE_YEAR
        use_real_data: bool = True
    
    class PortfolioRiskAnalysisRequest(BaseModel):
        portfolio_id: str
        holdings: Dict[str, float]
        period: AnalysisPeriod = AnalysisPeriod.ONE_YEAR
        use_real_data: bool = True
    
    class PortfolioAnalysisResponse(BaseModel):
        success: bool
        message: str
        data_source: str
        execution_time: float
        timestamp: datetime
        optimization_result: Optional[Dict] = None
        trade_analysis: Optional[Dict] = None
        rebalancing_needed: Optional[bool] = None

logger = logging.getLogger(__name__)

# =============================================================================
# DIRECT TOOL IMPORT
# =============================================================================

def import_portfolio_tools():
    """Import portfolio tools directly to avoid circular dependencies"""
    try:
        project_root = os.path.dirname(os.path.dirname(__file__))
        tools_file = os.path.join(project_root, 'tools', 'standalone_fmp_portfolio_tools.py')
        
        if not os.path.exists(tools_file):
            fallback_paths = [
                os.path.join(os.path.dirname(__file__), '..', 'tools', 'standalone_fmp_portfolio_tools.py'),
                os.path.join(project_root, 'standalone_fmp_portfolio_tools.py'),
            ]
            
            for path in fallback_paths:
                if os.path.exists(path):
                    tools_file = path
                    break
        
        if not os.path.exists(tools_file):
            raise ImportError(f"Portfolio tools not found at: {tools_file}")
        
        spec = importlib.util.spec_from_file_location("portfolio_tools", tools_file)
        tools_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tools_module)
        
        logger.info(f"✓ Portfolio tools imported from: {tools_file}")
        return tools_module, True
        
    except Exception as e:
        logger.error(f"Failed to import portfolio tools: {e}")
        return None, False

# Import tools
PORTFOLIO_TOOLS, HAS_PORTFOLIO_TOOLS = import_portfolio_tools()

if HAS_PORTFOLIO_TOOLS:
    optimize_portfolio = PORTFOLIO_TOOLS.optimize_portfolio
    calculate_portfolio_risk = PORTFOLIO_TOOLS.calculate_portfolio_risk
    OptimizationResult = PORTFOLIO_TOOLS.OptimizationResult
    RiskMetrics = PORTFOLIO_TOOLS.RiskMetrics
    logger.info("✓ Portfolio tools functions loaded successfully")
else:
    logger.warning("⚠ Portfolio tools not available - using fallback implementations")

# =============================================================================
# ATTRIBUTE MAPPING HELPERS
# =============================================================================

def map_optimization_result(optimization_result) -> Dict[str, Any]:
    """Map OptimizationResult to expected service format"""
    try:
        result_dict = {
            'success': getattr(optimization_result, 'success', True),
            'expected_return': getattr(optimization_result, 'expected_return', 0.0),
            'expected_volatility': getattr(optimization_result, 'expected_volatility', 0.0),
            'sharpe_ratio': getattr(optimization_result, 'sharpe_ratio', 0.0),
            'optimal_weights': getattr(optimization_result, 'optimal_weights', {}),
            'error': getattr(optimization_result, 'error', None),
            'data_source': getattr(optimization_result, 'data_source', 'FMP'),
            'analysis_period': getattr(optimization_result, 'analysis_period', '1year'),
            'symbols_analyzed': getattr(optimization_result, 'symbols_analyzed', []),
        }
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Error mapping optimization result: {e}")
        return {
            'success': False,
            'error': f"Result mapping failed: {str(e)}",
            'expected_return': 0.0,
            'expected_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'optimal_weights': {},
            'data_source': 'Unknown',
            'analysis_period': '1year',
            'symbols_analyzed': []
        }

def map_risk_metrics(risk_metrics) -> Dict[str, Any]:
    """Map RiskMetrics to expected service format"""
    try:
        result_dict = {
            'success': getattr(risk_metrics, 'success', True),
            'annual_return': getattr(risk_metrics, 'annual_return', 0.0),
            'annual_volatility': getattr(risk_metrics, 'annual_volatility', 0.0),
            'sharpe_ratio': getattr(risk_metrics, 'sharpe_ratio', 0.0),
            'max_drawdown': getattr(risk_metrics, 'max_drawdown', 0.0),
            'error': getattr(risk_metrics, 'error', None),
            'data_source': getattr(risk_metrics, 'data_source', 'FMP'),
            'analysis_period': getattr(risk_metrics, 'analysis_period', '1year'),
            'symbols_analyzed': getattr(risk_metrics, 'symbols_analyzed', []),
        }
        
        return result_dict
        
    except Exception as e:
        logger.error(f"Error mapping risk metrics: {e}")
        return {
            'success': False,
            'error': f"Risk mapping failed: {str(e)}",
            'annual_return': 0.0,
            'annual_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'data_source': 'Unknown',
            'analysis_period': '1year',
            'symbols_analyzed': []
        }

# =============================================================================
# SERVICE IMPLEMENTATION
# =============================================================================

class PortfolioManagementService:
    """Portfolio Management Service - Final Working Version"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize portfolio service"""
        self.risk_free_rate = risk_free_rate
        self._cache = {}
        self.service_name = "Portfolio Management Service"
        
        # Service health check
        self.integration_status = {
            "tools_available": HAS_PORTFOLIO_TOOLS,
            "centralized_models": HAS_CENTRALIZED_MODELS,
            "fmp_integration": HAS_PORTFOLIO_TOOLS,
            "service_ready": HAS_PORTFOLIO_TOOLS
        }
        
        logger.info(f"{self.service_name} initialized - Tools: {'✓' if HAS_PORTFOLIO_TOOLS else '✗'}")
    
    def get_service_health(self) -> Dict[str, Any]:
        """Get service health status"""
        return {
            "service": self.service_name,
            "status": "healthy" if self.integration_status["service_ready"] else "degraded",
            "integration_status": self.integration_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # =============================================================================
    # PORTFOLIO OPTIMIZATION
    # =============================================================================
    
    async def comprehensive_portfolio_optimization(
        self, 
        request: Union[PortfolioOptimizationRequest, Dict[str, Any]]
    ) -> PortfolioAnalysisResponse:
        """Comprehensive portfolio optimization with your actual model structure"""
        start_time = datetime.utcnow()
        
        try:
            # Handle both request objects and dicts
            if isinstance(request, dict):
                symbols = request.get('symbols', [])
                optimization_method = request.get('optimization_method', 'max_sharpe')
                period = request.get('period', '1year')
                use_real_data = request.get('use_real_data', True)
                portfolio_id = request.get('portfolio_id')
                target_return = request.get('target_return')
                risk_tolerance = request.get('risk_tolerance', 0.5)
            else:
                # Extract from your actual Pydantic model
                symbols = request.symbols or []
                optimization_method = request.optimization_method.value if hasattr(request.optimization_method, 'value') else str(request.optimization_method)
                period = request.period.value if hasattr(request.period, 'value') else str(request.period)
                use_real_data = request.use_real_data
                portfolio_id = getattr(request, 'portfolio_id', None)
                target_return = getattr(request, 'target_return', None)
                risk_tolerance = getattr(request, 'risk_tolerance', 0.5)
            
            # Validation
            if not symbols or len(symbols) < 2:
                return PortfolioAnalysisResponse(
                    success=False,
                    message="At least 2 symbols required for optimization",
                    data_source="Service Validation",
                    execution_time=(datetime.utcnow() - start_time).total_seconds(),
                    timestamp=datetime.utcnow()
                )
            
            if not HAS_PORTFOLIO_TOOLS:
                return PortfolioAnalysisResponse(
                    success=False,
                    message="Portfolio tools not available",
                    data_source="Service Status",
                    execution_time=(datetime.utcnow() - start_time).total_seconds(),
                    timestamp=datetime.utcnow()
                )
            
            # Perform optimization using real tools
            logger.info(f"Starting optimization for {len(symbols)} symbols: {symbols}")
            
            # Call optimization with the parameters your tools expect
            optimization_result = await optimize_portfolio(
                symbols=symbols,
                method=optimization_method,
                period=period,
                risk_free_rate=self.risk_free_rate
            )
            
            # Map result with proper attribute handling
            mapped_result = map_optimization_result(optimization_result)
            
            if not mapped_result['success']:
                return PortfolioAnalysisResponse(
                    success=False,
                    message=f"Optimization failed: {mapped_result['error']}",
                    data_source="Portfolio Tools",
                    execution_time=(datetime.utcnow() - start_time).total_seconds(),
                    timestamp=datetime.utcnow()
                )
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Build comprehensive response
            return PortfolioAnalysisResponse(
                success=True,
                message="Portfolio optimization completed successfully",
                data_source=mapped_result['data_source'],
                execution_time=execution_time,
                timestamp=datetime.utcnow(),
                optimization_result={
                    "expected_return": f"{mapped_result['expected_return']:.2%}",
                    "expected_volatility": f"{mapped_result['expected_volatility']:.2%}",
                    "sharpe_ratio": f"{mapped_result['sharpe_ratio']:.3f}",
                    "optimal_weights": mapped_result['optimal_weights'],
                    "optimization_method": optimization_method,
                    "symbols_optimized": len(symbols)
                }
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Portfolio optimization error: {e}")
            return PortfolioAnalysisResponse(
                success=False,
                message=f"Service error: {str(e)}",
                data_source="Service Error",
                execution_time=execution_time,
                timestamp=datetime.utcnow()
            )
    
    # =============================================================================
    # PORTFOLIO REBALANCING
    # =============================================================================
    
    async def intelligent_rebalancing_analysis(
        self, 
        request: Union[RebalancingRequest, Dict[str, Any]]
    ) -> PortfolioAnalysisResponse:
        """Intelligent portfolio rebalancing analysis"""
        start_time = datetime.utcnow()
        
        try:
            # Handle both request objects and dicts
            if isinstance(request, dict):
                current_holdings = request.get('current_holdings', {})
                target_allocation = request.get('target_allocation')
                rebalance_threshold = request.get('rebalance_threshold', 0.05)
                period = request.get('period', '1year')
                use_real_data = request.get('use_real_data', True)
            else:
                current_holdings = request.current_holdings or {}
                target_allocation = request.target_allocation
                rebalance_threshold = request.rebalance_threshold
                period = request.period.value if hasattr(request.period, 'value') else str(request.period)
                use_real_data = request.use_real_data
            
            # Validation
            if not current_holdings:
                return PortfolioAnalysisResponse(
                    success=False,
                    message="Current holdings required for rebalancing analysis",
                    data_source="Service Validation",
                    execution_time=(datetime.utcnow() - start_time).total_seconds(),
                    timestamp=datetime.utcnow()
                )
            
            # If no target allocation, optimize to get one
            if not target_allocation:
                symbols = list(current_holdings.keys())
                optimization_result = await optimize_portfolio(
                    symbols=symbols,
                    method='max_sharpe',
                    period=period,
                    risk_free_rate=self.risk_free_rate
                )
                
                mapped_result = map_optimization_result(optimization_result)
                if not mapped_result['success']:
                    return PortfolioAnalysisResponse(
                        success=False,
                        message=f"Could not determine target allocation: {mapped_result['error']}",
                        data_source="Portfolio Tools",
                        execution_time=(datetime.utcnow() - start_time).total_seconds(),
                        timestamp=datetime.utcnow()
                    )
                
                target_allocation = mapped_result['optimal_weights']
            
            # Perform rebalancing analysis
            rebalancing_result = await self._perform_rebalancing_analysis(
                current_holdings=current_holdings,
                target_allocation=target_allocation,
                rebalance_threshold=rebalance_threshold
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return PortfolioAnalysisResponse(
                success=True,
                message="Rebalancing analysis completed successfully",
                data_source="FMP + Portfolio Analysis" if use_real_data else "Portfolio Analysis Only",
                execution_time=execution_time,
                timestamp=datetime.utcnow(),
                rebalancing_needed=rebalancing_result["rebalancing_needed"],
                trade_analysis=rebalancing_result
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Portfolio rebalancing error: {e}")
            return PortfolioAnalysisResponse(
                success=False,
                message=f"Service error: {str(e)}",
                data_source="Service Error",
                execution_time=execution_time,
                timestamp=datetime.utcnow()
            )
    
    # =============================================================================
    # PORTFOLIO RISK ANALYSIS  
    # =============================================================================
    
    async def comprehensive_portfolio_risk_analysis(
        self, 
        request: Union[PortfolioRiskAnalysisRequest, Dict[str, Any]]
    ) -> PortfolioAnalysisResponse:
        """Comprehensive portfolio risk analysis"""
        start_time = datetime.utcnow()
        
        try:
            # Handle both request objects and dicts
            if isinstance(request, dict):
                holdings = request.get('holdings', {})
                symbols = request.get('symbols', list(holdings.keys()))
                period = request.get('period', '1year')
                use_real_data = request.get('use_real_data', True)
            else:
                holdings = request.holdings or {}
                symbols = list(holdings.keys()) if holdings else []
                period = request.period.value if hasattr(request.period, 'value') else str(request.period)
                use_real_data = request.use_real_data
            
            # Validation
            if not holdings and not symbols:
                return PortfolioAnalysisResponse(
                    success=False,
                    message="Holdings or symbols required for risk analysis",
                    data_source="Service Validation",
                    execution_time=(datetime.utcnow() - start_time).total_seconds(),
                    timestamp=datetime.utcnow()
                )
            
            if not HAS_PORTFOLIO_TOOLS:
                return PortfolioAnalysisResponse(
                    success=False,
                    message="Portfolio tools not available",
                    data_source="Service Status",
                    execution_time=(datetime.utcnow() - start_time).total_seconds(),
                    timestamp=datetime.utcnow()
                )
            
            # Calculate risk metrics using real tools
            risk_metrics = await calculate_portfolio_risk(
                holdings=holdings,
                period=period,
                risk_free_rate=self.risk_free_rate
            )
            
            # Map result with proper attribute handling
            mapped_result = map_risk_metrics(risk_metrics)
            
            if not mapped_result['success']:
                return PortfolioAnalysisResponse(
                    success=False,
                    message=f"Risk calculation failed: {mapped_result['error']}",
                    data_source="Portfolio Tools",
                    execution_time=(datetime.utcnow() - start_time).total_seconds(),
                    timestamp=datetime.utcnow()
                )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return PortfolioAnalysisResponse(
                success=True,
                message="Portfolio risk analysis completed successfully",
                data_source=mapped_result['data_source'],
                execution_time=execution_time,
                timestamp=datetime.utcnow(),
                optimization_result={
                    "annual_return": f"{mapped_result['annual_return']:.2%}",
                    "annual_volatility": f"{mapped_result['annual_volatility']:.2%}",
                    "sharpe_ratio": f"{mapped_result['sharpe_ratio']:.3f}",
                    "max_drawdown": f"{mapped_result['max_drawdown']:.2%}",
                    "risk_assessment": self._generate_risk_insights(mapped_result, holdings)
                }
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Portfolio risk analysis error: {e}")
            return PortfolioAnalysisResponse(
                success=False,
                message=f"Service error: {str(e)}",
                data_source="Service Error",
                execution_time=execution_time,
                timestamp=datetime.utcnow()
            )
    
    # =============================================================================
    # HELPER METHODS
    # =============================================================================
    
    async def _perform_rebalancing_analysis(
        self, 
        current_holdings: Dict[str, float], 
        target_allocation: Dict[str, float], 
        rebalance_threshold: float
    ) -> Dict[str, Any]:
        """Perform detailed rebalancing analysis"""
        
        total_value = sum(current_holdings.values())
        current_weights = {k: v/total_value for k, v in current_holdings.items()}
        
        rebalancing_needed = False
        trades = []
        
        # Check deviations
        for symbol in target_allocation:
            current_weight = current_weights.get(symbol, 0.0)
            target_weight = target_allocation[symbol]
            deviation = abs(current_weight - target_weight)
            
            if deviation > rebalance_threshold:
                rebalancing_needed = True
                
                current_value = current_holdings.get(symbol, 0.0)
                target_value = target_weight * total_value
                trade_amount = target_value - current_value
                
                if abs(trade_amount) >= total_value * 0.01:
                    trade = {
                        'symbol': symbol,
                        'action': 'BUY' if trade_amount > 0 else 'SELL',
                        'amount': abs(trade_amount),
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'deviation': deviation
                    }
                    trades.append(trade)
        
        return {
            'rebalancing_needed': rebalancing_needed,
            'trades': trades,
            'total_trade_value': sum(trade['amount'] for trade in trades),
            'estimated_costs': sum(trade['amount'] for trade in trades) * 0.001
        }
    
    def _generate_risk_insights(self, risk_metrics: Dict[str, Any], holdings: Dict[str, float]) -> Dict[str, Any]:
        """Generate comprehensive risk insights"""
        return {
            'risk_level': self._categorize_risk_level(risk_metrics['annual_volatility']),
            'return_profile': self._categorize_return_profile(risk_metrics['annual_return']),
            'sharpe_assessment': self._assess_sharpe_performance(risk_metrics['sharpe_ratio']),
            'portfolio_diversification': {
                'holdings_count': len(holdings),
                'diversification': 'Well diversified' if len(holdings) >= 5 else 'Concentrated'
            }
        }
    
    def _categorize_risk_level(self, volatility: float) -> str:
        """Categorize portfolio risk level"""
        if volatility < 0.1:
            return "Conservative"
        elif volatility < 0.2:
            return "Moderate"
        else:
            return "Aggressive"
    
    def _categorize_return_profile(self, annual_return: float) -> str:
        """Categorize return profile"""
        if annual_return > 0.15:
            return "High growth"
        elif annual_return > 0.08:
            return "Growth oriented"
        else:
            return "Income focused"
    
    def _assess_sharpe_performance(self, sharpe_ratio: float) -> str:
        """Assess Sharpe ratio performance"""
        if sharpe_ratio > 2.0:
            return "Exceptional"
        elif sharpe_ratio > 1.5:
            return "Excellent"
        elif sharpe_ratio > 1.0:
            return "Good"
        else:
            return "Below average"


# =============================================================================
# TEST FUNCTION
# =============================================================================

async def test_portfolio_service():
    """Test the final portfolio service"""
    print("Testing Final Portfolio Management Service")
    print("=" * 46)
    
    try:
        # Initialize service
        service = PortfolioManagementService(risk_free_rate=0.025)
        
        # Service health check
        health = service.get_service_health()
        print(f"\nService Health: {health['status']}")
        print(f"Tools Available: {'✓' if health['integration_status']['tools_available'] else '✗'}")
        print(f"Centralized Models: {'✓' if health['integration_status']['centralized_models'] else '✗'}")
        print(f"FMP Integration: {'✓' if health['integration_status']['fmp_integration'] else '✗'}")
        
        if not service.integration_status["service_ready"]:
            print("⚠ Service not ready - skipping functionality tests")
            return False
        
        # Test 1: Portfolio Optimization with your actual model
        print("\n1. Testing Portfolio Optimization...")
        opt_request = PortfolioOptimizationRequest(
            symbols=['AAPL', 'GOOGL', 'MSFT'],
            optimization_method=OptimizationMethod.MAX_SHARPE,
            period=AnalysisPeriod.THREE_MONTHS,
            use_real_data=True,
            portfolio_id=1
        )
        
        opt_response = await service.comprehensive_portfolio_optimization(opt_request)
        
        if opt_response.success:
            print("✓ Portfolio optimization successful")
            print(f"  Execution time: {opt_response.execution_time:.3f}s")
            print(f"  Data source: {opt_response.data_source}")
            print(f"  Message: {opt_response.message}")
            if opt_response.optimization_result:
                result = opt_response.optimization_result
                print(f"  Expected return: {result.get('expected_return', 'N/A')}")
                print(f"  Sharpe ratio: {result.get('sharpe_ratio', 'N/A')}")
        else:
            print(f"✗ Portfolio optimization failed: {opt_response.message}")
            return False
        
        # Test 2: Legacy format compatibility
        print("\n2. Testing Legacy Format Compatibility...")
        legacy_request = {
            'symbols': ['AAPL', 'MSFT'],
            'optimization_method': 'max_sharpe',
            'period': '3months',
            'use_real_data': True
        }
        
        legacy_response = await service.comprehensive_portfolio_optimization(legacy_request)
        
        if legacy_response.success:
            print("✓ Legacy format compatibility working")
            print(f"  Execution time: {legacy_response.execution_time:.3f}s")
            print(f"  Data source: {legacy_response.data_source}")
        else:
            print(f"✗ Legacy format compatibility failed: {legacy_response.message}")
            return False
        
        print(f"\n🎉 All portfolio service tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Portfolio service testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Portfolio Management Service - Final Working Version")
    print("=" * 52)
    
    success = asyncio.run(test_portfolio_service())
    
    if success:
        print("\n" + "=" * 52)
        print("✅ PORTFOLIO SERVICE REFACTORING COMPLETE")
        print("=" * 52)
        print("\nKey Features:")
        print("- Works with your actual centralized models")
        print("- Handles missing attributes gracefully") 
        print("- Real FMP market data integration")
        print("- Backward compatibility with dict format")
        print("- Proper Pydantic response validation")
        print("- Production-ready error handling")
    else:
        print("\n❌ Portfolio service needs attention")