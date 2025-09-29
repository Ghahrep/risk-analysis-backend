# tools/__init__.py
"""
Financial Analysis Tools Package
==============================

Complete toolkit for institutional-grade financial analysis.
Updated with robust import handling and error recovery.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# =============================================================================
# TOOL AVAILABILITY FLAGS
# =============================================================================

RISK_TOOLS_AVAILABLE = False
PORTFOLIO_TOOLS_AVAILABLE = False
BEHAVIORAL_TOOLS_AVAILABLE = False
FORECASTING_TOOLS_AVAILABLE = False
REGIME_TOOLS_AVAILABLE = False
FRACTAL_TOOLS_AVAILABLE = False

# =============================================================================
# RISK ANALYSIS TOOLS (Core - Should be available)
# =============================================================================

risk_tools_functions = []

try:
    from .risk_tools_standalone import (  # CORRECTED: risk_tools_standalone.py
        calculate_risk_metrics,
        calculate_drawdowns,
        fit_garch_forecast,
        calculate_correlation_matrix,
        calculate_beta,
        apply_market_shock,
        run_monte_carlo_stress_test
    )
    RISK_TOOLS_AVAILABLE = True
    risk_tools_functions = [
        "calculate_risk_metrics", "calculate_drawdowns", "fit_garch_forecast",
        "calculate_correlation_matrix", "calculate_beta", "apply_market_shock",
        "run_monte_carlo_stress_test"
    ]
    logger.info("✓ Risk tools loaded successfully")
except ImportError as e:
    logger.warning(f"⚠ Risk tools not available: {e}")

# Try to load additional risk tools if available
try:
    from .risk_tools_standalone import (  # CORRECTED: risk_tools_standalone.py
        calculate_factor_risk_attribution,
        calculate_dynamic_risk_budgets,
        calculate_time_varying_risk,
        generate_risk_sentiment_index,
        calculate_regime_conditional_risk
    )
    risk_tools_functions.extend([
        "calculate_factor_risk_attribution", "calculate_dynamic_risk_budgets",
        "calculate_time_varying_risk", "generate_risk_sentiment_index",
        "calculate_regime_conditional_risk"
    ])
    logger.info("✓ Advanced risk tools loaded")
except ImportError:
    logger.info("ℹ Advanced risk tools not available")

# =============================================================================
# PORTFOLIO ANALYSIS TOOLS
# =============================================================================

portfolio_tools_functions = []

try:
    from .standalone_fmp_portfolio_tools import (  # CORRECTED: standalone_fmp_portfolio_tools.py
        optimize_portfolio,
        calculate_efficient_frontier,
        rebalance_portfolio,
        calculate_portfolio_risk
    )
    PORTFOLIO_TOOLS_AVAILABLE = True
    portfolio_tools_functions = [
        "optimize_portfolio", "calculate_efficient_frontier",
        "rebalance_portfolio", "calculate_portfolio_risk"
    ]
    logger.info("✓ Portfolio tools loaded successfully")
except ImportError as e:
    logger.warning(f"⚠ Portfolio tools not available: {e}")

# Try additional portfolio tools
try:
    from .standalone_fmp_portfolio_tools import (  # CORRECTED: standalone_fmp_portfolio_tools.py
        screen_securities,
        calculate_hedging_analysis,
        generate_portfolio_summary
    )
    portfolio_tools_functions.extend([
        "screen_securities", "calculate_hedging_analysis", "generate_portfolio_summary"
    ])
    logger.info("✓ Advanced portfolio tools loaded")
except ImportError:
    logger.info("ℹ Advanced portfolio tools not available")

# =============================================================================
# BEHAVIORAL ANALYSIS TOOLS
# =============================================================================

behavioral_tools_functions = []

try:
    from .behavioral_tools_standalone import (
        analyze_behavioral_biases,
        analyze_market_sentiment,
        assess_behavioral_profile
    )
    BEHAVIORAL_TOOLS_AVAILABLE = True
    behavioral_tools_functions = [
        "analyze_behavioral_biases", "analyze_market_sentiment", "assess_behavioral_profile"
    ]
    logger.info("✓ Behavioral tools loaded successfully")
except ImportError as e:
    logger.warning(f"⚠ Behavioral tools not available: {e}")

# Try additional behavioral tools
try:
    from .behavioral_tools_standalone import (
        generate_behavioral_recommendations,
        calculate_behavioral_risk_score
    )
    behavioral_tools_functions.extend([
        "generate_behavioral_recommendations", "calculate_behavioral_risk_score"
    ])
    logger.info("✓ Advanced behavioral tools loaded")
except ImportError:
    logger.info("ℹ Advanced behavioral tools not available")

# =============================================================================
# FORECASTING TOOLS
# =============================================================================

forecasting_tools_functions = []

try:
    from .forecasting_tools import (
        forecast_returns_ensemble,
        forecast_volatility_extended
    )
    FORECASTING_TOOLS_AVAILABLE = True
    forecasting_tools_functions = [
        "forecast_returns_ensemble", "forecast_volatility_extended"
    ]
    logger.info("✓ Forecasting tools loaded successfully")
except ImportError as e:
    logger.warning(f"⚠ Forecasting tools not available: {e}")

# Try additional forecasting tools
try:
    from .forecasting_tools import (
        forecast_regime_transitions,
        analyze_scenarios_comprehensive,
        comprehensive_forecasting_analysis
    )
    forecasting_tools_functions.extend([
        "forecast_regime_transitions", "analyze_scenarios_comprehensive",
        "comprehensive_forecasting_analysis"
    ])
    logger.info("✓ Advanced forecasting tools loaded")
except ImportError:
    logger.info("ℹ Advanced forecasting tools not available")

# =============================================================================
# REGIME ANALYSIS TOOLS
# =============================================================================

regime_tools_functions = []

try:
    from .regime_tools_standalone import (
        detect_hmm_regimes,
        detect_volatility_regimes
    )
    REGIME_TOOLS_AVAILABLE = True
    regime_tools_functions = [
        "detect_hmm_regimes", "detect_volatility_regimes"
    ]
    logger.info("✓ Regime tools loaded successfully")
except ImportError as e:
    logger.warning(f"⚠ Regime tools not available: {e}")

# Try additional regime tools
try:
    from .regime_tools_standalone import (
        forecast_regime_transition_probability,
        analyze_regime_conditional_returns,
        detect_regime_shifts,
        comprehensive_regime_analysis
    )
    regime_tools_functions.extend([
        "forecast_regime_transition_probability", "analyze_regime_conditional_returns",
        "detect_regime_shifts", "comprehensive_regime_analysis"
    ])
    logger.info("✓ Advanced regime tools loaded")
except ImportError:
    logger.info("ℹ Advanced regime tools not available")

# =============================================================================
# FRACTAL ANALYSIS TOOLS (Future Phase)
# =============================================================================

fractal_tools_functions = []

try:
    from .fractal_tools import (
        calculate_hurst,
        detect_fractal_patterns
    )
    FRACTAL_TOOLS_AVAILABLE = True
    fractal_tools_functions = ["calculate_hurst", "detect_fractal_patterns"]
    logger.info("✓ Fractal tools loaded successfully")
except ImportError:
    logger.info("ℹ Fractal tools not available (future phase)")

# =============================================================================
# DYNAMIC EXPORTS
# =============================================================================

# Build __all__ dynamically based on what's actually available
__all__ = []
__all__.extend(risk_tools_functions)
__all__.extend(portfolio_tools_functions)
__all__.extend(behavioral_tools_functions)
__all__.extend(forecasting_tools_functions)
__all__.extend(regime_tools_functions)
__all__.extend(fractal_tools_functions)

# =============================================================================
# TOOL AVAILABILITY FUNCTIONS
# =============================================================================

def get_available_tools() -> Dict[str, Any]:
    """Get dictionary of available tool categories with details"""
    return {
        'risk_tools': {
            'available': RISK_TOOLS_AVAILABLE,
            'functions': risk_tools_functions,
            'count': len(risk_tools_functions)
        },
        'portfolio_tools': {
            'available': PORTFOLIO_TOOLS_AVAILABLE,
            'functions': portfolio_tools_functions,
            'count': len(portfolio_tools_functions)
        },
        'behavioral_tools': {
            'available': BEHAVIORAL_TOOLS_AVAILABLE,
            'functions': behavioral_tools_functions,
            'count': len(behavioral_tools_functions)
        },
        'forecasting_tools': {
            'available': FORECASTING_TOOLS_AVAILABLE,
            'functions': forecasting_tools_functions,
            'count': len(forecasting_tools_functions)
        },
        'regime_tools': {
            'available': REGIME_TOOLS_AVAILABLE,
            'functions': regime_tools_functions,
            'count': len(regime_tools_functions)
        },
        'fractal_tools': {
            'available': FRACTAL_TOOLS_AVAILABLE,
            'functions': fractal_tools_functions,
            'count': len(fractal_tools_functions)
        }
    }

def get_tool_summary() -> Dict[str, Any]:
    """Get comprehensive summary of all available tools"""
    available_tools = get_available_tools()
    
    total_categories = len(available_tools)
    available_categories = sum(1 for cat in available_tools.values() if cat['available'])
    total_functions = sum(cat['count'] for cat in available_tools.values())
    
    missing_categories = [
        cat_name for cat_name, cat_info in available_tools.items() 
        if not cat_info['available']
    ]
    
    return {
        'total_categories': total_categories,
        'available_categories': available_categories,
        'completion_percentage': round((available_categories / total_categories) * 100, 1),
        'missing_categories': missing_categories,
        'total_functions': total_functions,
        'available_functions': len(__all__),
        'tools_by_category': available_tools
    }

def check_tool_availability(tool_name: str) -> bool:
    """Check if a specific tool function is available"""
    return tool_name in __all__

def get_missing_tools() -> List[str]:
    """Get list of missing tool categories"""
    available_tools = get_available_tools()
    return [cat_name for cat_name, cat_info in available_tools.items() if not cat_info['available']]

# =============================================================================
# INITIALIZATION SUMMARY
# =============================================================================

def print_tools_status():
    """Print comprehensive tools initialization status"""
    summary = get_tool_summary()
    
    logger.info("=" * 60)
    logger.info("TOOLS INITIALIZATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Available Categories: {summary['available_categories']}/{summary['total_categories']} ({summary['completion_percentage']}%)")
    logger.info(f"Available Functions: {summary['available_functions']}")
    
    for cat_name, cat_info in summary['tools_by_category'].items():
        status = "✓ Available" if cat_info['available'] else "✗ Not Available"
        logger.info(f"{cat_name.replace('_', ' ').title()}: {status} ({cat_info['count']} functions)")
    
    if summary['missing_categories']:
        logger.info(f"Missing: {', '.join(summary['missing_categories'])}")
    
    logger.info("=" * 60)

# Print status on import
print_tools_status()

# Add utility functions to exports
__all__.extend([
    'get_available_tools',
    'get_tool_summary', 
    'check_tool_availability',
    'get_missing_tools',
    'print_tools_status'
])