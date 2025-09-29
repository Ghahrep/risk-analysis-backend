# tools/behavioral_tools_standalone.py - FMP Integrated
"""
Behavioral Finance Analysis Tools - Standalone with FMP Integration
================================================================

Institutional-grade behavioral analysis with real market data integration.
Follows proven integration patterns from risk_tools_standalone.py.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import importlib.util

logger = logging.getLogger(__name__)

# FMP INTEGRATION (Following proven risk_tools pattern)
FMP_AVAILABLE = False
_data_manager = None

def import_fmp_tools():
    """Import FMP integration directly from file"""
    try:
        spec = importlib.util.spec_from_file_location(
            "fmp_integration", 
            "data/providers/fmp_integration.py"
        )
        fmp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fmp_module)
        return fmp_module.PortfolioDataManager, fmp_module.FMPDataProvider
    except Exception as e:
        logger.warning(f"Could not import FMP integration: {e}")
        return None, None

# Initialize FMP integration
PortfolioDataManager, FMPDataProvider = import_fmp_tools()
if PortfolioDataManager and FMPDataProvider:
    FMP_AVAILABLE = True
    logger.info("FMP integration available for behavioral analysis")

def get_data_manager():
    """Get singleton data manager instance"""
    global _data_manager
    if _data_manager is None and FMP_AVAILABLE:
        _data_manager = PortfolioDataManager()
    return _data_manager

# STANDARDIZED DATA MODELS (keeping your existing models)

@dataclass
class BiasDetectionResult:
    """Individual bias detection result"""
    bias_type: str
    confidence: float
    severity: str  # 'Low', 'Medium', 'High'
    evidence: List[str]
    risk_impact: float
    mitigation_strategies: List[str]
    
@dataclass
class BehavioralProfile:
    """Comprehensive behavioral assessment profile"""
    overall_risk_score: float
    dominant_biases: List[str]
    emotional_state: str
    decision_confidence: float
    maturity_level: str
    recommendations: List[str]

@dataclass
class SentimentAnalysis:
    """Market sentiment analysis result"""
    sentiment: str  # 'positive', 'negative', 'neutral', 'uncertain'
    confidence: float
    emotional_intensity: str
    risk_indicators: Dict[str, int]
    market_timing_risk: float

# MAIN FMP-INTEGRATED FUNCTIONS

async def analyze_behavioral_biases_with_real_data(
    conversation_messages: List[Dict[str, str]], 
    symbols: List[str],
    period: str = "1year",
    use_real_data: bool = True
) -> Dict[str, Any]:
    """
    MAIN: Enhanced behavioral bias analysis with real FMP market data
    
    This is the new FMP-integrated version of your analysis function
    """
    try:
        logger.info(f"Starting behavioral analysis for {len(symbols)} symbols with real data: {use_real_data}")
        
        # Get real market data if available
        portfolio_data = None
        data_source = "Conversation Analysis Only"
        
        if use_real_data and FMP_AVAILABLE and symbols:
            data_manager = get_data_manager()
            if data_manager:
                logger.info(f"Getting enhanced portfolio context with FMP data")
                portfolio_data = await get_enhanced_portfolio_context_with_fmp(
                    symbols, data_manager, period
                )
                if portfolio_data and not portfolio_data.get('error'):
                    data_source = "Data from FMPDataProvider"
                    logger.info(f"✓ FMP portfolio context retrieved successfully")
        
        # Extract user messages
        user_messages = [
            msg['content'].lower() 
            for msg in conversation_messages 
            if msg.get('role') == 'user' and msg.get('content')
        ]
        
        if not user_messages:
            return _empty_bias_analysis_with_source(data_source)
        
        # Run enhanced bias detectors with FMP data
        detected_biases = []
        
        # Enhanced loss aversion detection
        loss_aversion = _detect_loss_aversion_enhanced(user_messages, portfolio_data)
        if loss_aversion.confidence > 0.4:
            detected_biases.append(loss_aversion)
        
        # Enhanced overconfidence detection  
        overconfidence = _detect_overconfidence_bias_enhanced(user_messages, portfolio_data)
        if overconfidence.confidence > 0.4:
            detected_biases.append(overconfidence)
        
        # Original bias detectors
        herding = _detect_herding_bias(user_messages)
        if herding.confidence > 0.4:
            detected_biases.append(herding)
            
        anchoring = _detect_anchoring_bias(user_messages)
        if anchoring.confidence > 0.4:
            detected_biases.append(anchoring)
            
        confirmation = _detect_confirmation_bias(user_messages)
        if confirmation.confidence > 0.4:
            detected_biases.append(confirmation)
        
        # Enhanced risk calculation with portfolio context
        overall_risk = _calculate_enhanced_bias_risk_impact(detected_biases, portfolio_data)
        
        # Enhanced recommendations
        recommendations = _generate_enhanced_bias_mitigation_plan(detected_biases, overall_risk, portfolio_data)
        
        result = {
            'success': True,
            'analysis_type': 'enhanced_behavioral_bias_detection',
            'data_source': data_source,
            'message_count': len(user_messages),
            'symbols_analyzed': symbols if use_real_data else [],
            'period': period,
            'portfolio_context_available': portfolio_data is not None,
            'detected_biases': [
                {
                    'bias_type': bias.bias_type,
                    'confidence': round(bias.confidence, 3),
                    'severity': bias.severity,
                    'evidence': bias.evidence,
                    'risk_impact': round(bias.risk_impact, 2),
                    'mitigation_strategies': bias.mitigation_strategies
                }
                for bias in detected_biases
            ],
            'overall_risk_score': round(overall_risk, 2),
            'bias_count': len(detected_biases),
            'recommendations': recommendations,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Add portfolio insights if available
        if portfolio_data and not portfolio_data.get('error'):
            result['portfolio_insights'] = {
                'symbols_analyzed': portfolio_data.get('symbols', []),
                'behavioral_risk_factors': portfolio_data.get('behavioral_risk_factors', {}),
                'key_concerns': _extract_key_portfolio_concerns(portfolio_data)
            }
            
        logger.info(f"✓ Behavioral analysis completed: {len(detected_biases)} biases detected, risk score: {overall_risk:.1f}")
        return result
        
    except Exception as e:
        logger.error(f"Enhanced behavioral analysis failed: {e}")
        return _error_bias_analysis_with_source(str(e), "Error")

async def assess_behavioral_profile_with_real_data(
    conversation_messages: List[Dict[str, str]],
    symbols: List[str],
    period: str = "1year",
    user_demographics: Optional[Dict] = None,
    use_real_data: bool = True
) -> Dict[str, Any]:
    """
    MAIN: Comprehensive behavioral profile with real FMP market data
    """
    try:
        logger.info(f"Starting behavioral profile assessment for {len(symbols)} symbols")
        
        if not conversation_messages:
            return _default_behavioral_profile_with_source("No conversation data")
        
        # Get enhanced bias analysis with real data
        bias_analysis = await analyze_behavioral_biases_with_real_data(
            conversation_messages, symbols, period, use_real_data
        )
        
        # Get sentiment analysis  
        sentiment_analysis = await analyze_market_sentiment_with_real_data(
            conversation_messages, symbols, period, use_real_data
        )
        
        # Calculate behavioral maturity
        maturity_assessment = _assess_behavioral_maturity(
            bias_analysis, sentiment_analysis, user_demographics
        )
        
        # Generate comprehensive profile
        profile = BehavioralProfile(
            overall_risk_score=_calculate_overall_behavioral_risk(bias_analysis, sentiment_analysis),
            dominant_biases=[bias['bias_type'] for bias in bias_analysis.get('detected_biases', [])[:3]],
            emotional_state=sentiment_analysis.get('sentiment', 'neutral'),
            decision_confidence=_calculate_decision_confidence(bias_analysis, sentiment_analysis),
            maturity_level=maturity_assessment['level'],
            recommendations=_generate_comprehensive_recommendations(bias_analysis, sentiment_analysis, maturity_assessment)
        )
        
        data_source = bias_analysis.get('data_source', 'Conversation Analysis Only')
        
        result = {
            'success': True,
            'analysis_type': 'behavioral_profile_with_fmp',
            'data_source': data_source,
            'symbols_analyzed': symbols if use_real_data else [],
            'period': period,
            'profile': {
                'overall_risk_score': round(profile.overall_risk_score, 2),
                'dominant_biases': profile.dominant_biases,
                'emotional_state': profile.emotional_state,
                'decision_confidence': round(profile.decision_confidence, 3),
                'maturity_level': profile.maturity_level,
                'recommendations': profile.recommendations
            },
            'supporting_analysis': {
                'bias_analysis': bias_analysis,
                'sentiment_analysis': sentiment_analysis,
                'maturity_assessment': maturity_assessment
            },
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✓ Behavioral profile completed: {profile.maturity_level} level, {profile.overall_risk_score:.1f} risk score")
        return result
        
    except Exception as e:
        logger.error(f"Behavioral profile assessment failed: {e}")
        return _error_behavioral_profile_with_source(str(e), "Error")

async def analyze_market_sentiment_with_real_data(
    conversation_messages: List[Dict[str, str]],
    symbols: Optional[List[str]] = None,
    period: str = "1year",
    use_real_data: bool = True
) -> Dict[str, Any]:
    """
    MAIN: Market sentiment analysis with optional FMP market context
    """
    try:
        logger.info(f"Starting sentiment analysis with real data: {use_real_data}")
        
        # Get market context if symbols provided
        market_context = None
        data_source = "Conversation Analysis Only"
        
        if use_real_data and FMP_AVAILABLE and symbols:
            data_manager = get_data_manager()
            if data_manager:
                market_context = await _get_market_context_for_sentiment(symbols, data_manager, period)
                if market_context and not market_context.get('error'):
                    data_source = "Data from FMPDataProvider"
        
        if not conversation_messages:
            return _neutral_sentiment_analysis_with_source(data_source)
        
        user_messages = [
            msg['content'].lower() 
            for msg in conversation_messages 
            if msg.get('role') == 'user' and msg.get('content')
        ]
        
        if not user_messages:
            return _neutral_sentiment_analysis_with_source(data_source)
        
        # Analyze sentiment indicators
        sentiment_result = _analyze_sentiment_indicators(user_messages)
        
        # Calculate market timing risk with real market context
        timing_risk = _calculate_market_timing_risk_enhanced(sentiment_result, market_context)
        
        # Generate sentiment-based recommendations
        recommendations = _generate_sentiment_recommendations_enhanced(sentiment_result, timing_risk, market_context)
        
        result = {
            'success': True,
            'analysis_type': 'market_sentiment_with_fmp',
            'data_source': data_source,
            'message_count': len(user_messages),
            'symbols_analyzed': symbols if use_real_data and symbols else [],
            'sentiment': sentiment_result.sentiment,
            'confidence': round(sentiment_result.confidence, 3),
            'emotional_intensity': sentiment_result.emotional_intensity,
            'risk_indicators': sentiment_result.risk_indicators,
            'market_timing_risk': round(timing_risk, 3),
            'recommendations': recommendations,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Add market context insights if available
        if market_context and not market_context.get('error'):
            result['market_context_insights'] = {
                'current_volatility': market_context.get('avg_volatility', 0),
                'market_trend': market_context.get('market_trend', 'neutral'),
                'sentiment_market_alignment': _assess_sentiment_market_alignment(sentiment_result, market_context)
            }
        
        logger.info(f"✓ Sentiment analysis completed: {sentiment_result.sentiment} sentiment, {timing_risk:.2f} timing risk")
        return result
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return _error_sentiment_analysis_with_source(str(e), "Error")

# FMP INTEGRATION FUNCTIONS

async def get_enhanced_portfolio_context_with_fmp(
    symbols: List[str],
    data_manager,
    period: str = "1year"
) -> Dict[str, Any]:
    """
    Get enhanced portfolio context using FMP data for behavioral analysis
    
    This replaces the placeholder function in your original code
    """
    try:
        logger.info(f"Getting FMP data for {len(symbols)} symbols, period: {period}")
        
        # Get returns data using proven pattern from risk_tools
        returns_data, data_source = await data_manager.get_returns_data(symbols, period)
        
        if returns_data is None:
            logger.warning("No returns data available from FMP")
            return {'error': 'No returns data available', 'symbols': symbols}
        
        logger.info(f"✓ Retrieved returns data: {data_source}")
        
        # Get additional portfolio data
        fundamentals_data = {}
        try:
            fundamentals_data = await data_manager.get_portfolio_fundamentals(symbols)
            logger.info(f"✓ Retrieved fundamentals for {len(fundamentals_data)} symbols")
        except Exception as e:
            logger.warning(f"Fundamentals data not available: {e}")
        
        # Calculate portfolio metrics for behavioral analysis
        portfolio_context = {
            'symbols': symbols,
            'total_symbols': len(symbols),
            'data_period': period,
            'data_source': data_source,
            'performance_metrics': {},
            'sector_analysis': {},
            'concentration_metrics': {},
            'behavioral_risk_factors': {},
            'valuation_context': {}
        }
        
        # Performance metrics calculation
        if isinstance(returns_data, pd.DataFrame) and not returns_data.empty:
            # Portfolio performance (equal weighted)
            portfolio_returns = returns_data.mean(axis=1)
            recent_performance = portfolio_returns.tail(30).mean() * 252
            total_volatility = portfolio_returns.std() * np.sqrt(252)
            
            # Drawdown analysis
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            current_drawdown = drawdown.iloc[-1]
            
            portfolio_context['performance_metrics'] = {
                'recent_annual_return': recent_performance,
                'annual_volatility': total_volatility,
                'max_drawdown': max_drawdown,
                'current_drawdown': current_drawdown,
                'daily_change': f"{portfolio_returns.iloc[-1]:.2%}" if len(portfolio_returns) > 0 else "0.00%",
                'trailing_30_day_return': portfolio_returns.tail(30).sum(),
                'positive_days_ratio': (portfolio_returns > 0).mean()
            }
            
            # Individual stock performance for behavioral insights
            individual_performance = {}
            for symbol in returns_data.columns:
                stock_returns = returns_data[symbol]
                if not stock_returns.empty:
                    individual_performance[symbol] = {
                        'recent_return': stock_returns.tail(30).mean() * 252,
                        'volatility': stock_returns.std() * np.sqrt(252),
                        'max_drawdown': _calculate_stock_drawdown(stock_returns),
                        'recent_trend': 'up' if stock_returns.tail(5).mean() > 0 else 'down'
                    }
            
            portfolio_context['individual_performance'] = individual_performance
        
        # Sector analysis from fundamentals
        if fundamentals_data:
            sector_counts = {}
            valuation_metrics = {}
            
            for symbol, data in fundamentals_data.items():
                if isinstance(data, dict):
                    # Extract sector information
                    sector = data.get('sector', 'Unknown')
                    if sector:
                        sector_counts[sector] = sector_counts.get(sector, 0) + 1
                    
                    # Extract valuation metrics
                    valuation_metrics[symbol] = {
                        'sector': sector,
                        'beta': data.get('beta', 1.0),
                        'pe_ratio': data.get('peRatio', None)
                    }
            
            # Calculate sector concentration
            total_stocks = len(symbols)
            max_sector_concentration = max(sector_counts.values()) / total_stocks if sector_counts else 0
            
            portfolio_context['sector_analysis'] = {
                'sector_counts': sector_counts,
                'total_sectors': len(sector_counts),
                'max_sector_concentration': max_sector_concentration,
                'diversification_score': len(sector_counts) / total_stocks if total_stocks > 0 else 0
            }
            
            portfolio_context['valuation_context'] = valuation_metrics
        
        # Calculate behavioral risk factors
        portfolio_context['behavioral_risk_factors'] = _calculate_behavioral_risk_factors_fmp(
            portfolio_context
        )
        
        logger.info(f"✓ Enhanced portfolio context generated with {data_source}")
        return portfolio_context
        
    except Exception as e:
        logger.error(f"Failed to get enhanced portfolio context: {e}")
        return {
            'symbols': symbols,
            'error': str(e),
            'behavioral_risk_factors': {'analysis_available': False}
        }

async def _get_market_context_for_sentiment(
    symbols: List[str],
    data_manager,
    period: str = "1year"
) -> Dict[str, Any]:
    """Get market context data for sentiment analysis"""
    try:
        # Get returns data for volatility analysis
        returns_data, data_source = await data_manager.get_returns_data(symbols, period)
        
        if returns_data is None:
            return {'error': 'No market data available'}
        
        # Calculate market volatility metrics
        portfolio_returns = returns_data.mean(axis=1)
        current_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Determine market trend
        recent_returns = portfolio_returns.tail(30).mean()
        market_trend = 'bullish' if recent_returns > 0.001 else 'bearish' if recent_returns < -0.001 else 'neutral'
        
        return {
            'data_source': data_source,
            'avg_volatility': current_volatility,
            'market_trend': market_trend,
            'recent_performance': recent_returns * 252,
            'symbols_count': len(symbols)
        }
        
    except Exception as e:
        logger.error(f"Failed to get market context: {e}")
        return {'error': str(e)}

def _calculate_behavioral_risk_factors_fmp(portfolio_context: Dict) -> Dict:
    """Calculate behavioral risk factors using FMP portfolio data"""
    risk_factors = {}
    
    # Concentration risk (overconfidence indicator)
    sector_analysis = portfolio_context.get('sector_analysis', {})
    max_concentration = sector_analysis.get('max_sector_concentration', 0)
    risk_factors['concentration_risk'] = {
        'level': 'high' if max_concentration > 0.4 else 'medium' if max_concentration > 0.25 else 'low',
        'value': max_concentration,
        'behavioral_implication': 'overconfidence' if max_concentration > 0.4 else None
    }
    
    # Recent loss exposure (loss aversion trigger)
    performance = portfolio_context.get('performance_metrics', {})
    current_drawdown = performance.get('current_drawdown', 0)
    max_drawdown = performance.get('max_drawdown', 0)
    
    risk_factors['loss_exposure'] = {
        'current_drawdown': current_drawdown,
        'max_drawdown': max_drawdown,
        'loss_recency': 'recent' if current_drawdown < -0.05 else 'moderate' if current_drawdown < -0.02 else 'low',
        'behavioral_implication': 'loss_aversion' if current_drawdown < -0.1 else None
    }
    
    # Volatility exposure (anxiety trigger)
    annual_volatility = performance.get('annual_volatility', 0.15)
    risk_factors['volatility_exposure'] = {
        'level': 'high' if annual_volatility > 0.25 else 'medium' if annual_volatility > 0.18 else 'low',
        'value': annual_volatility,
        'behavioral_implication': 'anxiety_driven_decisions' if annual_volatility > 0.3 else None
    }
    
    # High beta exposure check
    valuation_context = portfolio_context.get('valuation_context', {})
    risk_factors['beta_exposure'] = {
        'high_beta_stocks': _check_high_beta_exposure_fmp(valuation_context),
        'avg_beta': _calculate_average_beta(valuation_context)
    }
    
    return risk_factors

def _check_high_beta_exposure_fmp(valuation_metrics: Dict) -> bool:
    """Check if portfolio has high beta exposure using FMP data"""
    betas = []
    for symbol, metrics in valuation_metrics.items():
        if isinstance(metrics, dict) and 'beta' in metrics:
            beta_val = metrics['beta']
            if isinstance(beta_val, (int, float)) and not np.isnan(beta_val):
                betas.append(beta_val)
    
    if not betas:
        return False
    
    avg_beta = np.mean(betas)
    return avg_beta > 1.3

def _calculate_average_beta(valuation_metrics: Dict) -> float:
    """Calculate average portfolio beta"""
    betas = []
    for symbol, metrics in valuation_metrics.items():
        if isinstance(metrics, dict) and 'beta' in metrics:
            beta_val = metrics['beta']
            if isinstance(beta_val, (int, float)) and not np.isnan(beta_val):
                betas.append(beta_val)
    
    return np.mean(betas) if betas else 1.0

# ENHANCED HELPER FUNCTIONS

def _calculate_market_timing_risk_enhanced(sentiment_result: SentimentAnalysis, market_context: Optional[Dict] = None) -> float:
    """Enhanced market timing risk calculation with FMP market context"""
    base_risk = sentiment_result.market_timing_risk
    
    # Amplify risk based on real market volatility from FMP
    if market_context and not market_context.get('error'):
        market_volatility = market_context.get('avg_volatility', 0.15)
        if market_volatility > 0.25:  # High volatility
            base_risk *= 1.4
        elif market_volatility < 0.10:  # Low volatility
            base_risk *= 0.8
        
        # Sentiment-market alignment risk
        market_trend = market_context.get('market_trend', 'neutral')
        if sentiment_result.sentiment == 'positive' and market_trend == 'bearish':
            base_risk *= 1.3  # Contrarian sentiment increases timing risk
        elif sentiment_result.sentiment == 'negative' and market_trend == 'bullish':
            base_risk *= 1.3
    
    return min(1.0, base_risk)

def _generate_sentiment_recommendations_enhanced(sentiment_result: SentimentAnalysis, timing_risk: float, market_context: Optional[Dict] = None) -> List[str]:
    """Generate enhanced sentiment recommendations with market context"""
    recommendations = []
    
    # Base sentiment recommendations
    if sentiment_result.sentiment == 'negative' and sentiment_result.confidence > 0.6:
        recommendations.extend([
            "Avoid major investment changes during high anxiety periods",
            "Focus on long-term investment objectives rather than short-term fears"
        ])
    elif sentiment_result.sentiment == 'positive' and sentiment_result.confidence > 0.7:
        recommendations.extend([
            "Guard against overconfidence in current market conditions",
            "Maintain disciplined position sizing and profit-taking rules"
        ])
    
    # Market context specific recommendations
    if market_context and not market_context.get('error'):
        market_volatility = market_context.get('avg_volatility', 0.15)
        market_trend = market_context.get('market_trend', 'neutral')
        
        if market_volatility > 0.25:
            recommendations.append("High market volatility detected - reduce position sizes during uncertainty")
        
        # Sentiment-market misalignment warnings
        if sentiment_result.sentiment == 'positive' and market_trend == 'bearish':
            recommendations.append("Caution: Positive sentiment conflicts with bearish market trend")
        elif sentiment_result.sentiment == 'negative' and market_trend == 'bullish':
            recommendations.append("Consider: Negative sentiment may be excessive given bullish market conditions")
    
    # Timing risk recommendations
    if timing_risk > 0.7:
        recommendations.append("High market timing risk detected - implement systematic investment approach")
    
    return recommendations[:5]

def _assess_sentiment_market_alignment(sentiment_result: SentimentAnalysis, market_context: Dict) -> str:
    """Assess alignment between investor sentiment and market conditions"""
    if market_context.get('error'):
        return 'unknown'
    
    market_trend = market_context.get('market_trend', 'neutral')
    sentiment = sentiment_result.sentiment
    
    if sentiment == 'positive' and market_trend == 'bullish':
        return 'aligned_bullish'
    elif sentiment == 'negative' and market_trend == 'bearish':
        return 'aligned_bearish'
    elif sentiment == 'positive' and market_trend == 'bearish':
        return 'contrarian_bullish'
    elif sentiment == 'negative' and market_trend == 'bullish':
        return 'contrarian_bearish'
    else:
        return 'neutral'

# ENHANCED ERROR/DEFAULT FUNCTIONS WITH DATA SOURCE TRACKING

def _empty_bias_analysis_with_source(data_source: str) -> Dict[str, Any]:
    """Default response when no messages provided"""
    return {
        'success': True,
        'analysis_type': 'behavioral_bias_detection',
        'data_source': data_source,
        'message_count': 0,
        'detected_biases': [],
        'overall_risk_score': 25.0,
        'bias_count': 0,
        'recommendations': ['Insufficient conversation data for comprehensive bias analysis'],
        'analysis_timestamp': datetime.now().isoformat()
    }

def _error_bias_analysis_with_source(error: str, data_source: str) -> Dict[str, Any]:
    """Error response for bias analysis"""
    return {
        'success': False,
        'error': error,
        'analysis_type': 'behavioral_bias_detection',
        'data_source': data_source,
        'detected_biases': [],
        'overall_risk_score': 50.0,
        'analysis_timestamp': datetime.now().isoformat()
    }

def _neutral_sentiment_analysis_with_source(data_source: str) -> Dict[str, Any]:
    """Default neutral sentiment analysis"""
    return {
        'success': True,
        'analysis_type': 'market_sentiment',
        'data_source': data_source,
        'message_count': 0,
        'sentiment': 'neutral',
        'confidence': 0.5,
        'emotional_intensity': 'Low',
        'risk_indicators': {'positive': 0, 'negative': 0, 'uncertain': 0},
        'market_timing_risk': 0.3,
        'recommendations': ['Insufficient data for sentiment analysis'],
        'analysis_timestamp': datetime.now().isoformat()
    }

def _error_sentiment_analysis_with_source(error: str, data_source: str) -> Dict[str, Any]:
    """Error response for sentiment analysis"""
    return {
        'success': False,
        'error': error,
        'analysis_type': 'market_sentiment',
        'data_source': data_source,
        'sentiment': 'neutral',
        'confidence': 0.0,
        'analysis_timestamp': datetime.now().isoformat()
    }

def _default_behavioral_profile_with_source(data_source: str) -> Dict[str, Any]:
    """Default behavioral profile when insufficient data"""
    return {
        'success': True,
        'analysis_type': 'behavioral_profile_with_fmp',
        'data_source': data_source,
        'profile': {
            'overall_risk_score': 50.0,
            'dominant_biases': [],
            'emotional_state': 'neutral',
            'decision_confidence': 0.5,
            'maturity_level': 'Intermediate',
            'recommendations': ['Insufficient data for comprehensive profile assessment']
        },
        'supporting_analysis': {},
        'analysis_timestamp': datetime.now().isoformat()
    }

def _error_behavioral_profile_with_source(error: str, data_source: str) -> Dict[str, Any]:
    """Error response for behavioral profile"""
    return {
        'success': False,
        'error': error,
        'analysis_type': 'behavioral_profile_with_fmp',
        'data_source': data_source,
        'profile': {},
        'analysis_timestamp': datetime.now().isoformat()
    }

# KEEP ALL YOUR EXISTING HELPER FUNCTIONS
# (All the individual bias detectors, risk calculations, etc.)

def _detect_loss_aversion(messages: List[str], portfolio_data: Optional[Dict] = None) -> BiasDetectionResult:
    """Detect loss aversion bias patterns"""
    loss_indicators = [
        'losing money', 'can\'t afford to lose', 'terrible loss', 'devastating',
        'cutting losses', 'stop loss', 'protect capital', 'preserve', 'down'
    ]
    
    gain_indicators = [
        'making money', 'profits', 'gains', 'returns', 'upside',
        'growth', 'appreciation', 'winning', 'up'
    ]
    
    loss_count = sum(1 for msg in messages for indicator in loss_indicators if indicator in msg)
    gain_count = sum(1 for msg in messages for indicator in gain_indicators if indicator in msg)
    
    # Calculate bias strength
    total_mentions = loss_count + gain_count
    if total_mentions == 0:
        confidence = 0.0
    else:
        loss_ratio = loss_count / total_mentions
        confidence = max(0, (loss_ratio - 0.5) * 2)  # Scale to 0-1
    
    # Amplify if portfolio shows recent losses
    if portfolio_data and portfolio_data.get('performance_metrics', {}).get('daily_change', '0%').startswith('-'):
        confidence = min(1.0, confidence * 1.3)
    
    # Determine severity
    if confidence > 0.7:
        severity = 'High'
    elif confidence > 0.4:
        severity = 'Medium'
    else:
        severity = 'Low'
    
    evidence = []
    if loss_count > gain_count * 2:
        evidence.append(f"Disproportionate focus on losses ({loss_count} vs {gain_count} mentions)")
    
    return BiasDetectionResult(
        bias_type='loss_aversion',
        confidence=confidence,
        severity=severity,
        evidence=evidence,
        risk_impact=confidence * 30,  # 0-30 scale
        mitigation_strategies=[
            'Focus on long-term expected outcomes',
            'Pre-commit to stop-loss and profit-taking rules',
            'Use percentage-based position sizing'
        ] if confidence > 0.5 else []
    )

def _detect_overconfidence_bias(messages: List[str], portfolio_data: Optional[Dict] = None) -> BiasDetectionResult:
    """Detect overconfidence bias patterns"""
    overconfidence_indicators = [
        'definitely', 'certainly', 'guaranteed', 'sure thing', 'can\'t go wrong',
        'easy money', 'obvious', 'clearly', 'no doubt', 'of course'
    ]
    
    uncertainty_indicators = [
        'might', 'maybe', 'perhaps', 'could be', 'not sure', 'uncertain'
    ]
    
    confident_count = sum(1 for msg in messages for indicator in overconfidence_indicators if indicator in msg)
    uncertain_count = sum(1 for msg in messages for indicator in uncertainty_indicators if indicator in msg)
    
    # Calculate overconfidence ratio
    total_indicators = confident_count + uncertain_count
    if total_indicators == 0:
        confidence = 0.0
    else:
        overconfident_ratio = confident_count / total_indicators
        confidence = max(0, (overconfident_ratio - 0.3) * 1.4)  # Scale to 0-1
    
    # Check for frequent trading mentions (overconfidence indicator)
    trading_frequency_indicators = ['buy', 'sell', 'trade', 'rebalance']
    trading_mentions = sum(1 for msg in messages for indicator in trading_frequency_indicators if indicator in msg)
    
    if trading_mentions > len(messages) * 0.3:  # More than 30% of messages mention trading
        confidence = min(1.0, confidence * 1.2)
    
    severity = 'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low'
    
    evidence = []
    if confident_count > uncertain_count * 3:
        evidence.append(f"High confidence language frequency ({confident_count} confident vs {uncertain_count} uncertain)")
    if trading_mentions > len(messages) * 0.3:
        evidence.append(f"Frequent trading discussion ({trading_mentions}/{len(messages)} messages)")
    
    return BiasDetectionResult(
        bias_type='overconfidence',
        confidence=confidence,
        severity=severity,
        evidence=evidence,
        risk_impact=confidence * 25,
        mitigation_strategies=[
            'Track prediction accuracy systematically',
            'Seek external validation of decisions',
            'Implement mandatory cooling-off periods',
            'Use position sizing limits'
        ] if confidence > 0.5 else []
    )

# [Include all your other existing helper functions here - _detect_herding_bias, _detect_anchoring_bias, etc.]

# ADD THE ENHANCED VERSIONS OF YOUR EXISTING FUNCTIONS

def _detect_loss_aversion_enhanced(messages: List[str], portfolio_data: Optional[Dict] = None) -> BiasDetectionResult:
    """Enhanced loss aversion detection with FMP portfolio context"""
    
    # Use your existing text-based detection
    base_result = _detect_loss_aversion(messages, portfolio_data)
    
    # ENHANCED: FMP portfolio context amplification
    if portfolio_data and 'behavioral_risk_factors' in portfolio_data and not portfolio_data.get('error'):
        risk_factors = portfolio_data['behavioral_risk_factors']
        
        # Loss exposure amplification
        loss_exposure = risk_factors.get('loss_exposure', {})
        current_drawdown = loss_exposure.get('current_drawdown', 0)
        
        portfolio_amplifier = 1.0
        additional_evidence = []
        
        if current_drawdown < -0.1:  # More than 10% drawdown
            portfolio_amplifier = 1.5
            additional_evidence.append(f"Current portfolio drawdown of {current_drawdown:.1%} amplifies loss aversion")
        elif current_drawdown < -0.05:  # More than 5% drawdown
            portfolio_amplifier = 1.2
            additional_evidence.append(f"Recent portfolio losses ({current_drawdown:.1%}) may trigger loss aversion")
        
        # Individual stock losses
        individual_performance = portfolio_data.get('individual_performance', {})
        if individual_performance:
            losing_stocks = [
                symbol for symbol, perf in individual_performance.items()
                if perf.get('recent_trend') == 'down'
            ]
            
            if len(losing_stocks) > len(individual_performance) / 2:
                portfolio_amplifier *= 1.3
                additional_evidence.append(f"Majority of holdings trending down ({len(losing_stocks)} of {len(individual_performance)})")
        
        # Apply enhancement
        enhanced_confidence = min(1.0, base_result.confidence * portfolio_amplifier)
        enhanced_evidence = base_result.evidence + additional_evidence
        enhanced_risk_impact = enhanced_confidence * 35
        
        return BiasDetectionResult(
            bias_type=base_result.bias_type,
            confidence=enhanced_confidence,
            severity='High' if enhanced_confidence > 0.7 else 'Medium' if enhanced_confidence > 0.4 else 'Low',
            evidence=enhanced_evidence,
            risk_impact=enhanced_risk_impact,
            mitigation_strategies=base_result.mitigation_strategies + ([
                'Consider reducing portfolio volatility through diversification'
            ] if portfolio_amplifier > 1.2 else [])
        )
    
    return base_result

def _detect_overconfidence_bias_enhanced(messages: List[str], portfolio_data: Optional[Dict] = None) -> BiasDetectionResult:
    """Enhanced overconfidence detection with FMP portfolio context"""
    
    # Use existing detection as base
    base_result = _detect_overconfidence_bias(messages, portfolio_data)
    
    # ENHANCED: Portfolio-based overconfidence indicators
    if portfolio_data and 'behavioral_risk_factors' in portfolio_data and not portfolio_data.get('error'):
        risk_factors = portfolio_data['behavioral_risk_factors']
        
        portfolio_amplifier = 1.0
        additional_evidence = []
        
        # Concentration risk as overconfidence indicator
        concentration = risk_factors.get('concentration_risk', {})
        concentration_level = concentration.get('level', 'low')
        max_concentration = concentration.get('value', 0)
        
        if concentration_level == 'high':
            portfolio_amplifier = 1.4
            additional_evidence.append(f"High sector concentration ({max_concentration:.1%}) indicates overconfidence")
        elif concentration_level == 'medium':
            portfolio_amplifier = 1.2
            additional_evidence.append(f"Moderate concentration suggests some overconfidence")
        
        # High beta exposure
        beta_exposure = risk_factors.get('beta_exposure', {})
        if beta_exposure.get('high_beta_stocks', False):
            portfolio_amplifier *= 1.3
            additional_evidence.append("High-beta stock exposure suggests overconfidence in risk tolerance")
        
        # Apply enhancement
        enhanced_confidence = min(1.0, base_result.confidence * portfolio_amplifier)
        enhanced_evidence = base_result.evidence + additional_evidence
        
        return BiasDetectionResult(
            bias_type=base_result.bias_type,
            confidence=enhanced_confidence,
            severity='High' if enhanced_confidence > 0.7 else 'Medium' if enhanced_confidence > 0.4 else 'Low',
            evidence=enhanced_evidence,
            risk_impact=enhanced_confidence * 30,
            mitigation_strategies=base_result.mitigation_strategies + ([
                'Implement maximum position sizes to reduce concentration risk'
            ] if concentration_level == 'high' else [])
        )
    
    return base_result

# [Continue with all your other existing helper functions...]

def _detect_herding_bias(messages: List[str]) -> BiasDetectionResult:
    """Detect herding/FOMO bias patterns"""
    herding_indicators = [
        'everyone is buying', 'hot stock', 'get in on', 'don\'t want to miss out', 'fomo',
        'trending', 'popular', 'everyone talking about', 'all over social media'
    ]
    
    independent_indicators = [
        'my research', 'my analysis', 'i think', 'contrarian', 'different approach'
    ]
    
    herding_count = sum(1 for msg in messages for indicator in herding_indicators if indicator in msg)
    independent_count = sum(1 for msg in messages for indicator in independent_indicators if indicator in msg)
    
    total_indicators = herding_count + independent_count
    if total_indicators == 0:
        confidence = 0.0
    else:
        herding_ratio = herding_count / total_indicators
        confidence = herding_ratio
    
    severity = 'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low'
    
    evidence = []
    if herding_count > 0:
        evidence.append(f"References to following crowd trends ({herding_count} mentions)")
    
    return BiasDetectionResult(
        bias_type='herding_fomo',
        confidence=confidence,
        severity=severity,
        evidence=evidence,
        risk_impact=confidence * 20,
        mitigation_strategies=[
            'Base decisions on independent research',
            'Avoid social media investment advice',
            'Set predetermined investment criteria',
            'Practice contrarian thinking exercises'
        ] if confidence > 0.5 else []
    )

def _detect_anchoring_bias(messages: List[str]) -> BiasDetectionResult:
    """Detect anchoring bias patterns"""
    anchoring_indicators = [
        'bought at', 'paid', 'was worth', 'used to be', 'originally',
        'first time i saw', 'initially', 'started at'
    ]
    
    anchoring_count = sum(1 for msg in messages for indicator in anchoring_indicators if indicator in msg)
    confidence = min(1.0, anchoring_count / max(1, len(messages)) * 3)  # Scale based on message frequency
    
    severity = 'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low'
    
    evidence = []
    if anchoring_count >= 2:
        evidence.append(f"Multiple references to past prices or initial values ({anchoring_count} mentions)")
    
    return BiasDetectionResult(
        bias_type='anchoring',
        confidence=confidence,
        severity=severity,
        evidence=evidence,
        risk_impact=confidence * 15,
        mitigation_strategies=[
            'Use multiple valuation methods',
            'Focus on future prospects rather than past prices',
            'Set predetermined decision criteria',
            'Regularly update investment thesis'
        ] if confidence > 0.5 else []
    )

def _detect_confirmation_bias(messages: List[str], market_context: Optional[Dict] = None) -> BiasDetectionResult:
    """Detect confirmation bias patterns"""
    confirmation_indicators = [
        'of course', 'obviously', 'clearly', 'definitely', 'i knew',
        'as expected', 'proves my point', 'confirms', 'validates', 'exactly what i thought'
    ]
    
    disconfirmation_indicators = [
        'wrong about', 'unexpected', 'surprised', 'didn\'t see coming',
        'contrary to', 'despite', 'however', 'but', 'actually'
    ]
    
    confirm_count = sum(1 for msg in messages for indicator in confirmation_indicators if indicator in msg)
    disconfirm_count = sum(1 for msg in messages for indicator in disconfirmation_indicators if indicator in msg)
    
    total_indicators = confirm_count + disconfirm_count
    if total_indicators == 0:
        confidence = 0.0
    else:
        confirmation_ratio = confirm_count / total_indicators
        confidence = max(0, (confirmation_ratio - 0.5) * 2)  # Scale to 0-1
    
    severity = 'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low'
    
    evidence = []
    if confirm_count > disconfirm_count * 2:
        evidence.append(f"High confirmation language frequency ({confirm_count} vs {disconfirm_count})")
    
    return BiasDetectionResult(
        bias_type='confirmation',
        confidence=confidence,
        severity=severity,
        evidence=evidence,
        risk_impact=confidence * 25,
        mitigation_strategies=[
            'Actively seek contradictory evidence',
            'Set up devil\'s advocate scenarios',
            'Use systematic decision checklists',
            'Diversify information sources'
        ] if confidence > 0.5 else []
    )

# [Include all your other existing helper functions...]

def _analyze_sentiment_indicators(messages: List[str]) -> SentimentAnalysis:
    """Analyze sentiment indicators from messages"""
    # Define sentiment categories
    positive_indicators = ['bullish', 'optimistic', 'confident', 'excited', 'great opportunity']
    negative_indicators = ['bearish', 'worried', 'scared', 'anxious', 'concerned', 'panic']
    uncertain_indicators = ['unsure', 'confused', 'don\'t know', 'uncertain', 'mixed feelings']
    
    # Count indicators
    positive_count = sum(1 for msg in messages for indicator in positive_indicators if indicator in msg)
    negative_count = sum(1 for msg in messages for indicator in negative_indicators if indicator in msg)
    uncertain_count = sum(1 for msg in messages for indicator in uncertain_indicators if indicator in msg)
    
    total_indicators = positive_count + negative_count + uncertain_count
    
    # Determine dominant sentiment
    if total_indicators == 0:
        sentiment = 'neutral'
        confidence = 0.5
    else:
        if positive_count > negative_count and positive_count > uncertain_count:
            sentiment = 'positive'
            confidence = positive_count / total_indicators
        elif negative_count > positive_count and negative_count > uncertain_count:
            sentiment = 'negative'
            confidence = negative_count / total_indicators
        elif uncertain_count > 0:
            sentiment = 'uncertain'
            confidence = uncertain_count / total_indicators
        else:
            sentiment = 'mixed'
            confidence = 0.5
    
    # Determine emotional intensity
    if confidence > 0.8:
        emotional_intensity = 'High'
    elif confidence > 0.5:
        emotional_intensity = 'Medium'
    else:
        emotional_intensity = 'Low'
    
    return SentimentAnalysis(
        sentiment=sentiment,
        confidence=confidence,
        emotional_intensity=emotional_intensity,
        risk_indicators={
            'positive': positive_count,
            'negative': negative_count,
            'uncertain': uncertain_count
        },
        market_timing_risk=_calculate_sentiment_timing_risk(sentiment, confidence)
    )

def _calculate_sentiment_timing_risk(sentiment: str, confidence: float) -> float:
    """Calculate market timing risk from sentiment"""
    risk_multipliers = {
        'positive': 0.6,  # Moderate risk from overoptimism
        'negative': 0.8,  # High risk from panic decisions
        'uncertain': 0.7,  # High risk from decision paralysis
        'mixed': 0.5,     # Moderate risk from inconsistency
        'neutral': 0.3    # Low risk
    }
    
    base_risk = risk_multipliers.get(sentiment, 0.5)
    return min(1.0, base_risk * confidence)

def _calculate_stock_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown for individual stock"""
    if returns.empty:
        return 0.0
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def _calculate_enhanced_bias_risk_impact(biases: List[BiasDetectionResult], portfolio_data: Optional[Dict] = None) -> float:
    """Enhanced risk calculation with portfolio context"""
    if not biases:
        return 25.0
    
    # Base risk from biases
    base_risk = sum(bias.risk_impact for bias in biases)
    
    # Portfolio amplification factors
    if portfolio_data and 'behavioral_risk_factors' in portfolio_data and not portfolio_data.get('error'):
        risk_factors = portfolio_data['behavioral_risk_factors']
        
        # Volatility amplification
        volatility_level = risk_factors.get('volatility_exposure', {}).get('level', 'medium')
        if volatility_level == 'high':
            base_risk *= 1.2
        
        # Concentration amplification
        concentration_level = risk_factors.get('concentration_risk', {}).get('level', 'medium')
        if concentration_level == 'high':
            base_risk *= 1.3
        
        # Loss exposure amplification
        loss_recency = risk_factors.get('loss_exposure', {}).get('loss_recency', 'low')
        if loss_recency == 'recent':
            base_risk *= 1.4
        elif loss_recency == 'moderate':
            base_risk *= 1.1
    
    return min(100.0, base_risk)

def _generate_enhanced_bias_mitigation_plan(biases: List[BiasDetectionResult], overall_risk: float, portfolio_data: Optional[Dict] = None) -> List[str]:
    """Generate enhanced mitigation recommendations with portfolio context"""
    recommendations = []
    
    # Risk level recommendations
    if overall_risk > 70:
        recommendations.extend([
            "Immediate behavioral intervention recommended - high risk portfolio detected",
            "Consider professional behavioral coaching with portfolio review"
        ])
    elif overall_risk > 50:
        recommendations.extend([
            "Implement systematic bias monitoring with portfolio tracking",
            "Develop structured decision framework with quantitative rules"
        ])
    
    # Portfolio-specific recommendations
    if portfolio_data and 'behavioral_risk_factors' in portfolio_data and not portfolio_data.get('error'):
        risk_factors = portfolio_data['behavioral_risk_factors']
        
        concentration_level = risk_factors.get('concentration_risk', {}).get('level', 'medium')
        if concentration_level == 'high':
            recommendations.append("Reduce portfolio concentration to mitigate overconfidence bias")
        
        volatility_level = risk_factors.get('volatility_exposure', {}).get('level', 'medium')
        if volatility_level == 'high':
            recommendations.append("Consider volatility reduction strategies to minimize emotional decision-making")
        
        loss_recency = risk_factors.get('loss_exposure', {}).get('loss_recency', 'low')
        if loss_recency == 'recent':
            recommendations.append("Avoid major portfolio changes during recent loss periods")
    
    # Specific bias recommendations
    for bias in biases:
        recommendations.extend(bias.mitigation_strategies[:2])  # Top 2 from each bias
    
    # Remove duplicates and limit
    unique_recommendations = []
    for rec in recommendations:
        if rec not in unique_recommendations:
            unique_recommendations.append(rec)
    
    return unique_recommendations[:6]

def _extract_key_portfolio_concerns(portfolio_data: Dict) -> List[str]:
    """Extract key portfolio concerns for behavioral analysis"""
    concerns = []
    
    if portfolio_data.get('error'):
        return ["Portfolio analysis not available"]
    
    risk_factors = portfolio_data.get('behavioral_risk_factors', {})
    
    # Concentration concerns
    concentration = risk_factors.get('concentration_risk', {})
    if concentration.get('level') == 'high':
        concerns.append(f"High sector concentration ({concentration.get('value', 0):.1%}) may amplify overconfidence")
    
    # Loss concerns
    loss_exposure = risk_factors.get('loss_exposure', {})
    current_drawdown = loss_exposure.get('current_drawdown', 0)
    if current_drawdown < -0.1:
        concerns.append(f"Significant current losses ({current_drawdown:.1%}) may trigger loss aversion")
    
    # Volatility concerns
    volatility = risk_factors.get('volatility_exposure', {})
    if volatility.get('level') == 'high':
        concerns.append(f"High portfolio volatility may increase emotional decision-making")
    
    return concerns

# [Include all your other existing helper functions from the original file...]

def _assess_behavioral_maturity(bias_analysis: Dict, sentiment_analysis: Dict, demographics: Optional[Dict] = None) -> Dict:
    """Assess behavioral maturity level"""
    bias_count = bias_analysis.get('bias_count', 0)
    overall_risk = bias_analysis.get('overall_risk_score', 50)
    sentiment_stability = 1.0 - sentiment_analysis.get('market_timing_risk', 0.5)
    
    # Calculate maturity score
    maturity_score = (sentiment_stability * 0.4) + ((100 - overall_risk) / 100 * 0.6)
    
    if maturity_score > 0.8:
        level = 'Advanced'
    elif maturity_score > 0.6:
        level = 'Intermediate' 
    elif maturity_score > 0.4:
        level = 'Developing'
    else:
        level = 'Beginner'
    
    return {
        'level': level,
        'score': maturity_score,
        'factors': {
            'sentiment_stability': sentiment_stability,
            'bias_control': (100 - overall_risk) / 100
        }
    }

def _calculate_overall_behavioral_risk(bias_analysis: Dict, sentiment_analysis: Dict) -> float:
    """Calculate overall behavioral risk score"""
    bias_risk = bias_analysis.get('overall_risk_score', 50)
    sentiment_risk = sentiment_analysis.get('market_timing_risk', 0.5) * 100
    
    # Weighted average (bias risk weighted more heavily)
    return (bias_risk * 0.7) + (sentiment_risk * 0.3)

def _calculate_decision_confidence(bias_analysis: Dict, sentiment_analysis: Dict) -> float:
    """Calculate decision-making confidence score"""
    bias_count = bias_analysis.get('bias_count', 0)
    sentiment_confidence = sentiment_analysis.get('confidence', 0.5)
    
    # Lower bias count and stable sentiment = higher decision confidence
    bias_factor = max(0.3, 1.0 - (bias_count * 0.1))
    sentiment_factor = 1.0 - abs(sentiment_confidence - 0.5) * 2  # Penalty for extreme sentiments
    
    return (bias_factor + sentiment_factor) / 2

def _generate_comprehensive_recommendations(bias_analysis: Dict, sentiment_analysis: Dict, maturity: Dict) -> List[str]:
    """Generate comprehensive behavioral recommendations"""
    recommendations = []
    
    # Base on maturity level
    maturity_level = maturity['level']
    if maturity_level == 'Beginner':
        recommendations.extend([
            "Focus on building fundamental behavioral awareness",
            "Start with simple decision-making frameworks",
            "Practice regular self-assessment of emotional states"
        ])
    elif maturity_level == 'Developing':
        recommendations.extend([
            "Implement systematic bias monitoring protocols",
            "Develop structured investment decision processes",
            "Begin advanced behavioral training programs"
        ])
    elif maturity_level == 'Intermediate':
        recommendations.extend([
            "Refine existing behavioral management systems",
            "Focus on advanced bias mitigation techniques",
            "Consider peer review of investment decisions"
        ])
    else:  # Advanced
        recommendations.extend([
            "Maintain high standards of behavioral discipline",
            "Share knowledge with other investors",
            "Focus on edge case bias management"
        ])
    
    # Add specific bias recommendations
    bias_recs = bias_analysis.get('recommendations', [])
    recommendations.extend(bias_recs[:2])  # Add top 2 bias recommendations
    
    return recommendations[:5]  # Limit to 5 total

# INTEGRATION STATUS AND TESTING FUNCTIONS

def get_integration_status() -> Dict[str, Any]:
    """Get current FMP integration status for behavioral tools"""
    status = {
        'fmp_available': FMP_AVAILABLE,
        'data_manager_initialized': _data_manager is not None,
        'main_functions_available': [
            'analyze_behavioral_biases_with_real_data',
            'assess_behavioral_profile_with_real_data', 
            'analyze_market_sentiment_with_real_data'
        ],
        'integration_features': [
            'Real portfolio context analysis',
            'Enhanced bias detection with market data',
            'Market sentiment alignment analysis',
            'Portfolio-specific risk amplification',
            'Data source tracking'
        ]
    }
    
    if FMP_AVAILABLE:
        status['provider'] = 'FMPDataProvider'
        status['capabilities'] = 'Full integration with real market data'
    else:
        status['provider'] = 'None'
        status['capabilities'] = 'Limited to conversation analysis only'
    
    return status

async def test_behavioral_integration(
    test_symbols: List[str] = ["AAPL", "GOOGL", "MSFT"],
    test_messages: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """Test the FMP integration for behavioral analysis"""
    
    if test_messages is None:
        test_messages = [
            {"role": "user", "content": "I'm worried about my portfolio losing money in this market"},
            {"role": "user", "content": "I definitely think tech stocks will keep going up"},
            {"role": "user", "content": "Everyone is buying AI stocks, I don't want to miss out"}
        ]
    
    test_results = {
        'integration_status': get_integration_status(),
        'test_timestamp': datetime.now().isoformat(),
        'test_results': {}
    }
    
    # Test 1: Bias Analysis with Real Data
    try:
        logger.info("Testing bias analysis with real data...")
        bias_result = await analyze_behavioral_biases_with_real_data(
            test_messages, test_symbols, period="6month", use_real_data=True
        )
        test_results['test_results']['bias_analysis'] = {
            'success': bias_result.get('success', False),
            'data_source': bias_result.get('data_source', 'Unknown'),
            'biases_detected': bias_result.get('bias_count', 0),
            'portfolio_context': bias_result.get('portfolio_context_available', False)
        }
        logger.info(f"✓ Bias analysis: {bias_result.get('data_source', 'Unknown')}")
    except Exception as e:
        test_results['test_results']['bias_analysis'] = {'error': str(e)}
        logger.error(f"✗ Bias analysis failed: {e}")
    
    # Test 2: Behavioral Profile  
    try:
        logger.info("Testing behavioral profile with real data...")
        profile_result = await assess_behavioral_profile_with_real_data(
            test_messages, test_symbols, period="6month", use_real_data=True
        )
        test_results['test_results']['behavioral_profile'] = {
            'success': profile_result.get('success', False),
            'data_source': profile_result.get('data_source', 'Unknown'),
            'maturity_level': profile_result.get('profile', {}).get('maturity_level', 'Unknown'),
            'risk_score': profile_result.get('profile', {}).get('overall_risk_score', 0)
        }
        logger.info(f"✓ Behavioral profile: {profile_result.get('data_source', 'Unknown')}")
    except Exception as e:
        test_results['test_results']['behavioral_profile'] = {'error': str(e)}
        logger.error(f"✗ Behavioral profile failed: {e}")
    
    # Test 3: Sentiment Analysis
    try:
        logger.info("Testing sentiment analysis with real data...")
        sentiment_result = await analyze_market_sentiment_with_real_data(
            test_messages, test_symbols, period="6month", use_real_data=True
        )
        test_results['test_results']['sentiment_analysis'] = {
            'success': sentiment_result.get('success', False),
            'data_source': sentiment_result.get('data_source', 'Unknown'),
            'sentiment': sentiment_result.get('sentiment', 'Unknown'),
            'timing_risk': sentiment_result.get('market_timing_risk', 0)
        }
        logger.info(f"✓ Sentiment analysis: {sentiment_result.get('data_source', 'Unknown')}")
    except Exception as e:
        test_results['test_results']['sentiment_analysis'] = {'error': str(e)}
        logger.error(f"✗ Sentiment analysis failed: {e}")
    
    return test_results

# BACKWARDS COMPATIBILITY FUNCTIONS
# (Keep your original function names for existing API compatibility)

def analyze_behavioral_biases(
    conversation_messages: List[Dict[str, str]], 
    portfolio_data: Optional[Dict] = None,
    market_context: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    LEGACY: Original function maintained for backwards compatibility
    
    For new implementations, use analyze_behavioral_biases_with_real_data()
    """
    try:
        if not conversation_messages:
            return _empty_bias_analysis_with_source("Legacy Function")
        
        # Extract user messages
        user_messages = [
            msg['content'].lower() 
            for msg in conversation_messages 
            if msg.get('role') == 'user' and msg.get('content')
        ]
        
        if not user_messages:
            return _empty_bias_analysis_with_source("Legacy Function")
        
        # Run individual bias detectors (original versions)
        detected_biases = []
        
        loss_aversion = _detect_loss_aversion(user_messages, portfolio_data)
        if loss_aversion.confidence > 0.4:
            detected_biases.append(loss_aversion)
        
        overconfidence = _detect_overconfidence_bias(user_messages, portfolio_data)
        if overconfidence.confidence > 0.4:
            detected_biases.append(overconfidence)
        
        herding = _detect_herding_bias(user_messages)
        if herding.confidence > 0.4:
            detected_biases.append(herding)
        
        anchoring = _detect_anchoring_bias(user_messages)
        if anchoring.confidence > 0.4:
            detected_biases.append(anchoring)
        
        confirmation = _detect_confirmation_bias(user_messages, market_context)
        if confirmation.confidence > 0.4:
            detected_biases.append(confirmation)
        
        # Calculate overall risk impact
        overall_risk = sum(bias.risk_impact for bias in detected_biases) if detected_biases else 25.0
        
        # Generate recommendations
        recommendations = []
        for bias in detected_biases:
            recommendations.extend(bias.mitigation_strategies[:2])
        
        return {
            'success': True,
            'analysis_type': 'behavioral_bias_detection',
            'data_source': 'Legacy Function - Conversation Only',
            'message_count': len(user_messages),
            'detected_biases': [
                {
                    'bias_type': bias.bias_type,
                    'confidence': bias.confidence,
                    'severity': bias.severity,
                    'evidence': bias.evidence,
                    'risk_impact': bias.risk_impact,
                    'mitigation_strategies': bias.mitigation_strategies
                }
                for bias in detected_biases
            ],
            'overall_risk_score': min(100.0, overall_risk),
            'bias_count': len(detected_biases),
            'recommendations': recommendations[:6],
            'analysis_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Legacy bias analysis failed: {e}")
        return _error_bias_analysis_with_source(str(e), "Legacy Function")

def analyze_market_sentiment(
    conversation_messages: List[Dict[str, str]],
    market_context: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    LEGACY: Original function maintained for backwards compatibility
    
    For new implementations, use analyze_market_sentiment_with_real_data()
    """
    try:
        if not conversation_messages:
            return _neutral_sentiment_analysis_with_source("Legacy Function")
        
        user_messages = [
            msg['content'].lower() 
            for msg in conversation_messages 
            if msg.get('role') == 'user' and msg.get('content')
        ]
        
        if not user_messages:
            return _neutral_sentiment_analysis_with_source("Legacy Function")
        
        # Analyze sentiment indicators
        sentiment_result = _analyze_sentiment_indicators(user_messages)
        
        # Calculate market timing risk
        timing_risk = _calculate_sentiment_timing_risk(sentiment_result.sentiment, sentiment_result.confidence)
        
        # Generate sentiment-based recommendations
        recommendations = _generate_sentiment_recommendations_legacy(sentiment_result, timing_risk)
        
        return {
            'success': True,
            'analysis_type': 'market_sentiment',
            'data_source': 'Legacy Function - Conversation Only',
            'message_count': len(user_messages),
            'sentiment': sentiment_result.sentiment,
            'confidence': sentiment_result.confidence,
            'emotional_intensity': sentiment_result.emotional_intensity,
            'risk_indicators': sentiment_result.risk_indicators,
            'market_timing_risk': timing_risk,
            'recommendations': recommendations,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Legacy sentiment analysis failed: {e}")
        return _error_sentiment_analysis_with_source(str(e), "Legacy Function")

def assess_behavioral_profile(
    conversation_messages: List[Dict[str, str]],
    portfolio_data: Optional[Dict] = None,
    user_demographics: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    LEGACY: Original function maintained for backwards compatibility
    
    For new implementations, use assess_behavioral_profile_with_real_data()
    """
    try:
        if not conversation_messages:
            return _default_behavioral_profile_with_source("Legacy Function")
        
        # Get bias analysis (legacy version)
        bias_analysis = analyze_behavioral_biases(conversation_messages, portfolio_data)
        
        # Get sentiment analysis (legacy version)
        sentiment_analysis = analyze_market_sentiment(conversation_messages)
        
        # Calculate behavioral maturity
        maturity_assessment = _assess_behavioral_maturity(
            bias_analysis, sentiment_analysis, user_demographics
        )
        
        # Generate comprehensive profile
        profile = BehavioralProfile(
            overall_risk_score=_calculate_overall_behavioral_risk(bias_analysis, sentiment_analysis),
            dominant_biases=[bias['bias_type'] for bias in bias_analysis.get('detected_biases', [])[:3]],
            emotional_state=sentiment_analysis.get('sentiment', 'neutral'),
            decision_confidence=_calculate_decision_confidence(bias_analysis, sentiment_analysis),
            maturity_level=maturity_assessment['level'],
            recommendations=_generate_comprehensive_recommendations(bias_analysis, sentiment_analysis, maturity_assessment)
        )
        
        return {
            'success': True,
            'analysis_type': 'behavioral_profile',
            'data_source': 'Legacy Function - Conversation Only',
            'profile': {
                'overall_risk_score': profile.overall_risk_score,
                'dominant_biases': profile.dominant_biases,
                'emotional_state': profile.emotional_state,
                'decision_confidence': profile.decision_confidence,
                'maturity_level': profile.maturity_level,
                'recommendations': profile.recommendations
            },
            'supporting_analysis': {
                'bias_analysis': bias_analysis,
                'sentiment_analysis': sentiment_analysis,
                'maturity_assessment': maturity_assessment
            },
            'analysis_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Legacy behavioral profile assessment failed: {e}")
        return _error_behavioral_profile_with_source(str(e), "Legacy Function")

def _generate_sentiment_recommendations_legacy(sentiment_result: SentimentAnalysis, timing_risk: float) -> List[str]:
    """Generate basic sentiment recommendations for legacy function"""
    recommendations = []
    
    if sentiment_result.sentiment == 'negative' and sentiment_result.confidence > 0.6:
        recommendations.extend([
            "Avoid making major investment changes during high anxiety periods",
            "Focus on long-term investment objectives rather than short-term fears",
            "Consider systematic value averaging to reduce timing risk"
        ])
    elif sentiment_result.sentiment == 'positive' and sentiment_result.confidence > 0.7:
        recommendations.extend([
            "Guard against overconfidence in current market conditions",
            "Maintain disciplined position sizing and profit-taking rules",
            "Prepare for potential market corrections"
        ])
    elif sentiment_result.sentiment == 'uncertain':
        recommendations.extend([
            "Focus on high-conviction investment opportunities only",
            "Reduce position sizes until market clarity improves",
            "Seek additional research and analysis before major decisions"
        ])
    else:
        recommendations.extend([
            "Continue systematic investment approach",
            "Monitor sentiment changes for tactical adjustments",
            "Maintain emotional discipline in decision-making"
        ])
    
    if timing_risk > 0.7:
        recommendations.append("High market timing risk detected - avoid frequent trading")
    
    return recommendations[:4]

# MISSING FUNCTIONS FROM ORIGINAL FILE - ADDED FOR COMPLETENESS

def get_enhanced_portfolio_context(
    symbols: List[str],
    market_data_manager,
    period: str = "1year"
) -> Dict[str, Any]:
    """
    LEGACY: Original function signature for backwards compatibility
    
    This wraps the new async FMP version for existing code
    """
    import asyncio
    
    try:
        # Run the async version
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, return placeholder
            logger.warning("Cannot run async function in existing event loop - returning basic context")
            return {
                'symbols': symbols,
                'total_symbols': len(symbols),
                'data_period': period,
                'behavioral_risk_factors': {'analysis_available': False},
                'note': 'Use get_enhanced_portfolio_context_with_fmp() in async context'
            }
        else:
            return loop.run_until_complete(
                get_enhanced_portfolio_context_with_fmp(symbols, market_data_manager, period)
            )
    except Exception as e:
        logger.error(f"Enhanced portfolio context failed: {e}")
        return {
            'symbols': symbols,
            'error': str(e),
            'behavioral_risk_factors': {'analysis_available': False}
        }

def analyze_behavioral_biases_with_fmp(
    conversation_messages: List[Dict[str, str]],
    symbols: Optional[List[str]] = None,
    market_data_manager = None,
    period: str = "1year"
) -> Dict[str, Any]:
    """
    LEGACY: Sync wrapper for your original enhanced function
    
    For new code, use analyze_behavioral_biases_with_real_data() which is fully async
    """
    import asyncio
    
    if not symbols or not market_data_manager:
        # Fall back to conversation-only analysis
        return analyze_behavioral_biases(conversation_messages)
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            logger.warning("Cannot run async function in existing event loop - using conversation-only analysis")
            return analyze_behavioral_biases(conversation_messages)
        else:
            return loop.run_until_complete(
                analyze_behavioral_biases_with_real_data(
                    conversation_messages, symbols, period, use_real_data=True
                )
            )
    except Exception as e:
        logger.error(f"FMP behavioral analysis failed, falling back: {e}")
        return analyze_behavioral_biases(conversation_messages)

def analyze_conversation_comprehensive(
    conversation_messages: List[Dict[str, str]],
    symbols: Optional[List[str]] = None,
    user_context: Optional[Dict] = None,
    analysis_depth: str = "standard"
) -> Dict[str, Any]:
    """
    Comprehensive conversation analysis combining all behavioral tools
    
    This function provides a complete behavioral assessment
    """
    try:
        logger.info(f"Starting comprehensive analysis with depth: {analysis_depth}")
        
        if not conversation_messages:
            return {
                'success': False,
                'error': 'No conversation messages provided',
                'analysis_timestamp': datetime.now().isoformat()
            }
        
        # Initialize results
        comprehensive_results = {
            'success': True,
            'analysis_type': 'comprehensive_behavioral_analysis',
            'analysis_depth': analysis_depth,
            'message_count': len(conversation_messages),
            'symbols_analyzed': symbols or [],
            'components': {},
            'summary': {},
            'recommendations': [],
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Component 1: Bias Analysis
        try:
            if symbols and len(symbols) > 0:
                # Use FMP-enhanced analysis if symbols provided
                bias_analysis = analyze_behavioral_biases_with_fmp(
                    conversation_messages, symbols, None, "1year"
                )
            else:
                # Use conversation-only analysis
                bias_analysis = analyze_behavioral_biases(conversation_messages)
            
            comprehensive_results['components']['bias_analysis'] = bias_analysis
            logger.info("✓ Bias analysis completed")
        except Exception as e:
            logger.error(f"Bias analysis failed: {e}")
            comprehensive_results['components']['bias_analysis'] = {'error': str(e)}
        
        # Component 2: Sentiment Analysis
        try:
            sentiment_analysis = analyze_market_sentiment(conversation_messages)
            comprehensive_results['components']['sentiment_analysis'] = sentiment_analysis
            logger.info("✓ Sentiment analysis completed")
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            comprehensive_results['components']['sentiment_analysis'] = {'error': str(e)}
        
        # Component 3: Behavioral Profile
        try:
            profile_analysis = assess_behavioral_profile(
                conversation_messages, None, user_context
            )
            comprehensive_results['components']['behavioral_profile'] = profile_analysis
            logger.info("✓ Behavioral profile completed")
        except Exception as e:
            logger.error(f"Behavioral profile failed: {e}")
            comprehensive_results['components']['behavioral_profile'] = {'error': str(e)}
        
        # Generate comprehensive summary
        comprehensive_results['summary'] = _generate_comprehensive_summary(
            comprehensive_results['components'], analysis_depth
        )
        
        # Generate unified recommendations
        comprehensive_results['recommendations'] = _generate_unified_recommendations(
            comprehensive_results['components'], comprehensive_results['summary']
        )
        
        logger.info("✓ Comprehensive behavioral analysis completed")
        return comprehensive_results
        
    except Exception as e:
        logger.error(f"Comprehensive analysis failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'analysis_type': 'comprehensive_behavioral_analysis',
            'analysis_timestamp': datetime.now().isoformat()
        }

def _generate_comprehensive_summary(components: Dict, analysis_depth: str) -> Dict[str, Any]:
    """Generate comprehensive summary from all analysis components"""
    summary = {
        'overall_assessment': 'Unknown',
        'key_findings': [],
        'risk_level': 'Medium',
        'confidence_score': 0.5,
        'primary_concerns': [],
        'behavioral_strengths': []
    }
    
    try:
        # Extract key metrics from components
        bias_analysis = components.get('bias_analysis', {})
        sentiment_analysis = components.get('sentiment_analysis', {})
        profile_analysis = components.get('behavioral_profile', {})
        
        # Calculate overall risk
        risk_scores = []
        if bias_analysis.get('success') and 'overall_risk_score' in bias_analysis:
            risk_scores.append(bias_analysis['overall_risk_score'])
        
        if sentiment_analysis.get('success') and 'market_timing_risk' in sentiment_analysis:
            risk_scores.append(sentiment_analysis['market_timing_risk'] * 100)
        
        if profile_analysis.get('success'):
            profile_data = profile_analysis.get('profile', {})
            if 'overall_risk_score' in profile_data:
                risk_scores.append(profile_data['overall_risk_score'])
        
        # Calculate average risk
        if risk_scores:
            avg_risk = sum(risk_scores) / len(risk_scores)
            summary['risk_level'] = 'High' if avg_risk > 70 else 'Medium' if avg_risk > 40 else 'Low'
            summary['confidence_score'] = min(1.0, len(risk_scores) / 3.0)  # Based on successful components
        
        # Extract key findings
        findings = []
        if bias_analysis.get('success'):
            bias_count = bias_analysis.get('bias_count', 0)
            if bias_count > 0:
                findings.append(f"{bias_count} behavioral biases detected")
                dominant_biases = [b['bias_type'] for b in bias_analysis.get('detected_biases', [])[:2]]
                if dominant_biases:
                    findings.append(f"Primary biases: {', '.join(dominant_biases)}")
        
        if sentiment_analysis.get('success'):
            sentiment = sentiment_analysis.get('sentiment', 'neutral')
            confidence = sentiment_analysis.get('confidence', 0)
            if confidence > 0.6:
                findings.append(f"Strong {sentiment} market sentiment detected")
        
        if profile_analysis.get('success'):
            profile_data = profile_analysis.get('profile', {})
            maturity_level = profile_data.get('maturity_level', 'Unknown')
            if maturity_level != 'Unknown':
                findings.append(f"Behavioral maturity: {maturity_level}")
        
        summary['key_findings'] = findings
        
        # Overall assessment
        if summary['risk_level'] == 'Low' and summary['confidence_score'] > 0.7:
            summary['overall_assessment'] = 'Strong behavioral discipline with low risk indicators'
        elif summary['risk_level'] == 'High':
            summary['overall_assessment'] = 'Multiple behavioral risks detected - intervention recommended'
        else:
            summary['overall_assessment'] = 'Moderate behavioral risks - monitoring and improvement recommended'
            
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        summary['error'] = str(e)
    
    return summary

def _generate_unified_recommendations(components: Dict, summary: Dict) -> List[str]:
    """Generate unified recommendations from all analysis components"""
    unified_recs = []
    
    try:
        # Priority recommendations based on overall risk
        risk_level = summary.get('risk_level', 'Medium')
        
        if risk_level == 'High':
            unified_recs.extend([
                "Immediate behavioral intervention recommended",
                "Implement systematic decision-making frameworks",
                "Consider professional behavioral coaching"
            ])
        elif risk_level == 'Medium':
            unified_recs.extend([
                "Develop structured bias awareness protocols",
                "Implement regular behavioral self-assessment"
            ])
        
        # Collect recommendations from components
        all_component_recs = []
        
        # From bias analysis
        bias_analysis = components.get('bias_analysis', {})
        if bias_analysis.get('success'):
            all_component_recs.extend(bias_analysis.get('recommendations', []))
        
        # From sentiment analysis
        sentiment_analysis = components.get('sentiment_analysis', {})
        if sentiment_analysis.get('success'):
            all_component_recs.extend(sentiment_analysis.get('recommendations', []))
        
        # From behavioral profile
        profile_analysis = components.get('behavioral_profile', {})
        if profile_analysis.get('success'):
            profile_data = profile_analysis.get('profile', {})
            all_component_recs.extend(profile_data.get('recommendations', []))
        
        # Deduplicate and add best component recommendations
        unique_recs = []
        for rec in all_component_recs:
            if rec not in unique_recs and rec not in unified_recs:
                unique_recs.append(rec)
        
        # Add top component recommendations
        unified_recs.extend(unique_recs[:4])
        
        # General recommendations
        unified_recs.extend([
            "Maintain decision journal to track behavioral patterns",
            "Regular portfolio review with predetermined criteria"
        ])
        
    except Exception as e:
        logger.error(f"Unified recommendations generation failed: {e}")
        unified_recs = ["Unable to generate specific recommendations due to analysis error"]
    
    return unified_recs[:8]  # Limit to 8 total recommendations

# ADDITIONAL HELPER FUNCTIONS THAT MIGHT BE MISSING

def validate_conversation_format(conversation_messages: List[Dict]) -> bool:
    """Validate conversation message format"""
    if not isinstance(conversation_messages, list):
        return False
    
    for msg in conversation_messages:
        if not isinstance(msg, dict):
            return False
        if 'role' not in msg or 'content' not in msg:
            return False
        if msg['role'] not in ['user', 'assistant', 'system']:
            return False
        if not isinstance(msg['content'], str):
            return False
    
    return True

def extract_user_messages(conversation_messages: List[Dict]) -> List[str]:
    """Extract and clean user messages from conversation"""
    user_messages = []
    
    for msg in conversation_messages:
        if (msg.get('role') == 'user' and 
            msg.get('content') and 
            isinstance(msg['content'], str)):
            # Clean and normalize the message
            cleaned = msg['content'].lower().strip()
            if cleaned and len(cleaned) > 3:  # Minimum message length
                user_messages.append(cleaned)
    
    return user_messages

def calculate_message_statistics(conversation_messages: List[Dict]) -> Dict[str, Any]:
    """Calculate statistics about the conversation"""
    stats = {
        'total_messages': len(conversation_messages),
        'user_messages': 0,
        'assistant_messages': 0,
        'total_words': 0,
        'avg_message_length': 0,
        'conversation_span': 'Unknown'
    }
    
    try:
        user_msgs = []
        assistant_msgs = []
        all_words = []
        
        for msg in conversation_messages:
            content = msg.get('content', '')
            role = msg.get('role', '')
            
            if role == 'user':
                user_msgs.append(content)
            elif role == 'assistant':
                assistant_msgs.append(content)
            
            words = content.split() if isinstance(content, str) else []
            all_words.extend(words)
        
        stats['user_messages'] = len(user_msgs)
        stats['assistant_messages'] = len(assistant_msgs)
        stats['total_words'] = len(all_words)
        
        if len(conversation_messages) > 0:
            total_chars = sum(len(msg.get('content', '')) for msg in conversation_messages)
            stats['avg_message_length'] = total_chars / len(conversation_messages)
        
        # Determine conversation span
        if stats['user_messages'] > 20:
            stats['conversation_span'] = 'Extended'
        elif stats['user_messages'] > 10:
            stats['conversation_span'] = 'Moderate'
        elif stats['user_messages'] > 3:
            stats['conversation_span'] = 'Brief'
        else:
            stats['conversation_span'] = 'Minimal'
    
    except Exception as e:
        logger.error(f"Statistics calculation failed: {e}")
        stats['error'] = str(e)
    
    return stats