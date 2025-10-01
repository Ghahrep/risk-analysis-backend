"""
Metric Benchmarks and Contextual Explanations
Provides industry-standard benchmarks and visual indicators for all metrics
"""

import streamlit as st

# Benchmark definitions
BENCHMARKS = {
    'sharpe_ratio': {
        'excellent': 2.0,
        'good': 1.0,
        'fair': 0.5,
        'poor': 0,
        'description': 'Risk-adjusted return measure',
        'explanation': """
        **What it means:** Return earned per unit of risk taken.
        
        **Your score: {value:.2f}**
        - Below 0.5: Poor risk-adjusted returns
        - 0.5-1.0: Acceptable performance
        - 1.0-2.0: Good risk-adjusted returns
        - Above 2.0: Excellent performance
        
        **Benchmark:** S&P 500 typically 0.8-1.2
        """,
        'higher_is_better': True
    },
    'sortino_ratio': {
        'excellent': 2.0,
        'good': 1.5,
        'fair': 1.0,
        'poor': 0,
        'description': 'Downside risk-adjusted return',
        'explanation': """
        **What it means:** Like Sharpe, but only penalizes downside volatility.
        
        **Your score: {value:.2f}**
        - Below 1.0: Poor downside protection
        - 1.0-1.5: Acceptable downside management
        - 1.5-2.0: Good downside control
        - Above 2.0: Excellent downside protection
        
        **Why it matters:** Better for asymmetric return profiles
        """,
        'higher_is_better': True
    },
    'annual_volatility': {
        'excellent': 0.15,
        'good': 0.20,
        'fair': 0.30,
        'poor': 0.40,
        'description': 'Annualized standard deviation',
        'explanation': """
        **What it means:** How much your portfolio value swings.
        
        **Your score: {value:.1%}**
        - Below 15%: Very stable (bond-like)
        - 15-20%: Moderate (balanced)
        - 20-30%: High (equity-heavy)
        - Above 30%: Very high (aggressive)
        
        **Benchmark:** S&P 500 averages 18-20%
        """,
        'higher_is_better': False
    },
    'max_drawdown': {
        'excellent': -10,
        'good': -20,
        'fair': -30,
        'poor': -40,
        'description': 'Largest peak-to-trough decline',
        'explanation': """
        **What it means:** Worst loss from a peak.
        
        **Your score: {value:.1f}%**
        - Less than 10%: Excellent resilience
        - 10-20%: Good protection
        - 20-30%: Moderate drawdowns
        - Above 30%: Severe losses possible
        
        **Historical:** S&P 500 max drawdown was -55% (2008-09)
        """,
        'higher_is_better': False
    },
    'var_95': {
        'excellent': -2,
        'good': -5,
        'fair': -8,
        'poor': -12,
        'description': 'Value at Risk (95% confidence)',
        'explanation': """
        **What it means:** Maximum loss expected on 95% of days.
        
        **Your score: {value:.2%}**
        - Better than -2%: Low daily risk
        - -2% to -5%: Moderate risk
        - -5% to -8%: High risk
        - Worse than -8%: Very high risk
        
        **Translation:** On 19 out of 20 days, you won't lose more than this.
        """,
        'higher_is_better': False
    },
    'correlation': {
        'excellent': 0.3,
        'good': 0.5,
        'fair': 0.7,
        'poor': 0.9,
        'description': 'Average pairwise correlation',
        'explanation': """
        **What it means:** How similarly your holdings move.
        
        **Your score: {value:.2f}**
        - Below 0.3: Excellent diversification
        - 0.3-0.5: Good diversification
        - 0.5-0.7: Moderate correlation
        - Above 0.7: High correlation (limited diversification)
        
        **Ideal:** Below 0.5 for true diversification benefits
        """,
        'higher_is_better': False
    },
    'beta': {
        'excellent': 0.8,
        'good': 1.0,
        'fair': 1.2,
        'poor': 1.5,
        'description': 'Market sensitivity',
        'explanation': """
        **What it means:** How much your portfolio moves with the market.
        
        **Your score: {value:.2f}**
        - Below 0.8: Defensive (less volatile than market)
        - 0.8-1.2: Market-like volatility
        - Above 1.2: Aggressive (more volatile than market)
        
        **Example:** Beta of 1.5 means portfolio moves 50% more than market
        """,
        'higher_is_better': None  # Depends on investor preference
    }
}

def get_rating(metric_name, value):
    """Get rating (excellent/good/fair/poor) for a metric value"""
    if metric_name not in BENCHMARKS:
        return 'unknown', '#808495'
    
    # FIX: Handle None values
    if value is None:
        return 'unknown', '#808495'
    
    benchmark = BENCHMARKS[metric_name]
    higher_is_better = benchmark['higher_is_better']
    
    if higher_is_better is None:  # Beta case - neutral
        return 'neutral', '#667eea'
    
    if higher_is_better:
        if value >= benchmark['excellent']:
            return 'excellent', '#28a745'
        elif value >= benchmark['good']:
            return 'good', '#17a2b8'
        elif value >= benchmark['fair']:
            return 'fair', '#ffc107'
        else:
            return 'poor', '#dc3545'
    else:
        if value <= benchmark['excellent']:
            return 'excellent', '#28a745'
        elif value <= benchmark['good']:
            return 'good', '#17a2b8'
        elif value <= benchmark['fair']:
            return 'fair', '#ffc107'
        else:
            return 'poor', '#dc3545'

def get_star_rating(rating):
    """Convert rating to star display"""
    stars = {
        'excellent': '⭐⭐⭐',
        'good': '⭐⭐',
        'fair': '⭐',
        'poor': '❌',
        'neutral': '━'
    }
    return stars.get(rating, '?')

def display_metric_with_benchmark(metric_name, value, show_explanation=True):
    """
    Display a metric with visual benchmark and optional explanation
    
    Args:
        metric_name: Key from BENCHMARKS dict
        value: The metric value
        show_explanation: Whether to show expandable explanation
    """
    if metric_name not in BENCHMARKS:
        st.metric(metric_name, f"{value:.2f}")
        return
    
    benchmark = BENCHMARKS[metric_name]
    rating, color = get_rating(metric_name, value)
    stars = get_star_rating(rating)
    
    # Format value based on metric type
    if 'volatility' in metric_name or 'var' in metric_name or 'correlation' in metric_name:
        if abs(value) < 1:
            value_display = f"{value*100:.1f}%"
        else:
            value_display = f"{value:.2f}"
    elif 'drawdown' in metric_name:
        value_display = f"{abs(value):.1f}%"
    else:
        value_display = f"{value:.2f}"
    
    # Main metric display
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.metric(
            benchmark['description'].title(),
            value_display,
            delta=rating.capitalize() if rating != 'neutral' else None,
            delta_color="normal" if rating in ['excellent', 'good'] else "inverse" if rating != 'neutral' else "off"
        )
    
    with col2:
        st.markdown(f"""
        <div style='text-align: center; padding-top: 1rem;'>
            <div style='font-size: 1.5rem;'>{stars}</div>
            <div style='color: {color}; font-weight: 600; font-size: 0.8rem;'>
                {rating.upper() if rating != 'neutral' else 'NEUTRAL'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Expandable explanation
    if show_explanation:
        with st.expander(f"ℹ️ What does this mean?"):
            explanation = benchmark['explanation'].format(value=value)
            st.markdown(explanation)