"""
Portfolio Insights Dashboard
Synthesized recommendations and actionable insights from all analytics
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.portfolio_manager import get_portfolio, set_portfolio, normalize_weights, initialize_portfolio
from utils.portfolio_presets import list_presets, get_preset
from utils.api_client import get_risk_api_client

st.set_page_config(page_title="Portfolio Insights", page_icon="💡", layout="wide")

# Initialize portfolio
initialize_portfolio()

api_client = get_risk_api_client()

def calculate_health_score(metrics):
    """Calculate composite portfolio health score (0-100)"""
    scores = {}
    
    # Sharpe ratio score (0-30 points)
    sharpe = metrics.get('sharpe_ratio', 0)
    if sharpe > 2.0:
        scores['risk_return'] = 30
    elif sharpe > 1.0:
        scores['risk_return'] = 15 + (sharpe - 1.0) * 15
    elif sharpe > 0:
        scores['risk_return'] = sharpe * 15
    else:
        scores['risk_return'] = 0
    
    # Diversification score (0-25 points)
    correlation = metrics.get('avg_correlation', 0.5)
    scores['diversification'] = max(0, (1 - correlation) * 25)
    
    # Stress resilience score (0-25 points)
    resilience = metrics.get('resilience_score', 50)
    scores['stress_resilience'] = resilience * 0.25
    
    # Volatility score (0-20 points)
    volatility = metrics.get('annual_volatility', 0.25)
    if volatility < 0.15:
        scores['volatility'] = 20
    elif volatility < 0.25:
        scores['volatility'] = 20 - ((volatility - 0.15) * 100)
    else:
        scores['volatility'] = max(0, 10 - ((volatility - 0.25) * 40))
    
    total = sum(scores.values())
    return min(100, max(0, total)), scores

def generate_action_items(symbols, weights, metrics, period="1year"):
    """Generate prioritized action items based on analytics"""
    actions = []
    
    # Check correlation during crises
    crisis_multiplier = metrics.get('crisis_multiplier', 1)
    if crisis_multiplier > 1.8:
        actions.append({
            'priority': 1,
            'title': 'Critical: Reduce Crisis Correlation',
            'issue': f'Correlations increase {crisis_multiplier:.1f}x during market crashes',
            'impact': 'Your diversification vanishes exactly when you need it most',
            'recommendation': 'Add negatively-correlated assets: Treasury bonds, gold, or inverse ETFs',
            'improvement': f'Target crisis multiplier below 1.5x (currently {crisis_multiplier:.1f}x)',
            'urgency': 'High'
        })
    elif crisis_multiplier > 1.5:
        actions.append({
            'priority': 2,
            'title': 'Moderate: Improve Crisis Protection',
            'issue': f'Correlations increase {crisis_multiplier:.1f}x during downturns',
            'impact': 'Diversification benefits decline in stressed markets',
            'recommendation': 'Consider adding 10-15% defensive positions',
            'improvement': f'Target crisis multiplier below 1.3x',
            'urgency': 'Medium'
        })
    
    # Check stress testing results
    worst_loss = metrics.get('worst_case_loss', 0)
    if worst_loss < -20:
        portfolio_value = 100000
        actions.append({
            'priority': 1,
            'title': 'Critical: High Tail Risk',
            'issue': f'Worst-case scenario shows {abs(worst_loss):.1f}% loss',
            'impact': f'Potential ${abs(worst_loss * portfolio_value / 100):,.0f} loss in severe market stress',
            'recommendation': 'Implement protective puts or reduce position concentration',
            'improvement': 'Target worst-case loss below 15%',
            'urgency': 'High'
        })
    elif worst_loss < -15:
        actions.append({
            'priority': 2,
            'title': 'Elevated Stress Risk',
            'issue': f'Worst-case loss of {abs(worst_loss):.1f}%',
            'impact': 'Portfolio vulnerable to severe market shocks',
            'recommendation': 'Consider adding hedges or rebalancing to more defensive allocation',
            'improvement': 'Target worst-case loss below 12%',
            'urgency': 'Medium'
        })
    
    # Check Sharpe ratio optimization
    current_sharpe = metrics.get('current_sharpe', 0)
    optimal_sharpe = metrics.get('optimal_sharpe', 0)
    if optimal_sharpe > 0 and current_sharpe < optimal_sharpe * 0.85:
        improvement_pct = ((optimal_sharpe - current_sharpe) / current_sharpe * 100) if current_sharpe > 0 else 0
        actions.append({
            'priority': 2,
            'title': 'Suboptimal Risk-Return Profile',
            'issue': f'Current Sharpe {current_sharpe:.2f} vs optimal {optimal_sharpe:.2f}',
            'impact': f'Missing {improvement_pct:.0f}% potential risk-adjusted returns',
            'recommendation': 'Rebalance to optimized weights shown in Portfolio Analysis',
            'improvement': f'Increase Sharpe ratio by {optimal_sharpe - current_sharpe:.2f}',
            'urgency': 'Medium'
        })
    
    # Check concentration
    max_position = max(weights) if weights else 0
    if max_position > 0.4:
        actions.append({
            'priority': 2,
            'title': 'High Concentration Risk',
            'issue': f'Single position represents {max_position*100:.0f}% of portfolio',
            'impact': 'Excessive single-stock risk',
            'recommendation': 'Reduce largest position to below 30% and diversify',
            'improvement': 'Spread risk across more holdings',
            'urgency': 'Medium'
        })
    
    # Check volatility
    volatility = metrics.get('annual_volatility', 0)
    if volatility > 0.30:
        actions.append({
            'priority': 3,
            'title': 'High Portfolio Volatility',
            'issue': f'Annual volatility of {volatility*100:.1f}%',
            'impact': 'Large portfolio swings may trigger emotional decision-making',
            'recommendation': 'Add low-volatility assets or reduce equity allocation',
            'improvement': 'Target volatility below 25%',
            'urgency': 'Low'
        })
    
    return sorted(actions, key=lambda x: (x['priority'], x['urgency'] == 'High'), reverse=False)

def identify_strengths(metrics):
    """Identify what the portfolio is doing well"""
    strengths = []
    
    sharpe = metrics.get('sharpe_ratio', 0)
    if sharpe > 1.5:
        strengths.append(f"Excellent risk-adjusted returns (Sharpe: {sharpe:.2f})")
    elif sharpe > 1.0:
        strengths.append(f"Strong risk-adjusted returns (Sharpe: {sharpe:.2f})")
    
    correlation = metrics.get('avg_correlation', 0.5)
    if correlation < 0.4:
        strengths.append(f"Well-diversified holdings (avg correlation: {correlation:.2f})")
    
    resilience = metrics.get('resilience_score', 0)
    if resilience > 85:
        strengths.append(f"Excellent stress resilience (score: {resilience:.0f}/100)")
    
    volatility = metrics.get('annual_volatility', 0)
    if volatility < 0.18:
        strengths.append(f"Low volatility profile ({volatility*100:.1f}% annual)")
    
    crisis_multiplier = metrics.get('crisis_multiplier', 1)
    if crisis_multiplier < 1.3:
        strengths.append(f"Correlations remain stable during crises (multiplier: {crisis_multiplier:.1f}x)")
    
    if not strengths:
        strengths.append("Portfolio meets basic diversification requirements")
    
    return strengths

def create_health_gauge(score):
    """Create a gauge chart for health score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightcoral"},
                {'range': [40, 70], 'color': "lightyellow"},
                {'range': [70, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
    return fig

def create_score_breakdown(scores):
    """Create breakdown of health score components"""
    df = pd.DataFrame([
        {'Category': 'Risk-Return', 'Score': scores.get('risk_return', 0), 'Max': 30},
        {'Category': 'Diversification', 'Score': scores.get('diversification', 0), 'Max': 25},
        {'Category': 'Stress Resilience', 'Score': scores.get('stress_resilience', 0), 'Max': 25},
        {'Category': 'Volatility Control', 'Score': scores.get('volatility', 0), 'Max': 20}
    ])
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Current',
        x=df['Category'],
        y=df['Score'],
        marker_color='steelblue'
    ))
    fig.add_trace(go.Bar(
        name='Maximum',
        x=df['Category'],
        y=df['Max'] - df['Score'],
        marker_color='lightgray'
    ))
    
    fig.update_layout(
        barmode='stack',
        title='Health Score Breakdown',
        yaxis_title='Points',
        height=300,
        showlegend=False
    )
    
    return fig

def main():
    st.title("💡 Portfolio Insights & Recommendations")
    st.markdown("**Actionable recommendations synthesized from comprehensive analytics**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Portfolio Configuration")
        
        # Preset selector
        st.subheader("Quick Load")
        preset_options = ["Custom"] + list(list_presets().values())
        preset_name = st.selectbox(
            "Select Preset Portfolio",
            options=preset_options
        )
        
        if preset_name != "Custom":
            preset_key = [k for k, v in list_presets().items() if v == preset_name][0]
            preset = get_preset(preset_key)
            
            st.caption(f"*{preset['description']}*")
            
            if st.button("📥 Load Preset", width='stretch'):
                set_portfolio(preset["symbols"], preset["weights"])
                st.success(f"{preset['name']} loaded!")
                st.rerun()
        
        st.markdown("---")
        
        # Get current portfolio
        current_symbols, current_weights = get_portfolio()
        
        symbols_input = st.text_area(
            "Stock Symbols (one per line)",
            value='\n'.join(current_symbols),
            height=120
        )
        symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]
        
        st.subheader("Portfolio Weights")
        weights = []
        
        for i, symbol in enumerate(symbols):
            default_weight = current_weights[i] if i < len(current_weights) else 1.0/len(symbols)
            weight = st.slider(
                f"{symbol}",
                0.0, 1.0, default_weight, 0.01,
                key=f"insight_weight_{symbol}"
            )
            weights.append(weight)
        
        # Normalize
        total = sum(weights)
        if total > 0:
            weights = [w/total for w in weights]
        
        st.write(f"**Total Weight:** {sum(weights):.2%}")
        
        # Save button
        if st.button("💾 Save Portfolio", width='stretch'):
            set_portfolio(symbols, weights)
            st.success("Portfolio saved!")
        
        period = st.selectbox(
            "Analysis Period",
            ["1month", "3months", "6months", "1year", "2years"],
            index=3
        )
        
        analyze_button = st.button("🔍 Generate Insights", type="primary", width='stretch')
    
    if not symbols:
        st.info("👈 Select a preset or enter stock symbols in the sidebar to begin analysis")
        return
    
    # Run analysis
    if analyze_button:
        with st.spinner("Analyzing portfolio across all dimensions..."):
            # Gather all metrics
            risk_result = api_client.analyze_risk(symbols, weights, period)
            opt_result = api_client.optimize_portfolio(symbols, "max_sharpe", period)
            stress_result = api_client.stress_test(symbols, weights)
            corr_result = api_client.regime_correlations(symbols, "volatility", period)
            
            # Consolidate metrics
            metrics = {}
            if risk_result:
                risk_metrics = risk_result.get('metrics', {})
                metrics['sharpe_ratio'] = risk_metrics.get('sharpe_ratio', 0)
                metrics['current_sharpe'] = risk_metrics.get('sharpe_ratio', 0)
                metrics['annual_volatility'] = risk_metrics.get('annualized_volatility', 0)
                metrics['max_drawdown'] = risk_metrics.get('max_drawdown_pct', 0)
            
            if opt_result:
                opt_data = opt_result.get('optimization_results', {})
                metrics['optimal_sharpe'] = opt_data.get('sharpe_ratio', 0)
            
            if stress_result:
                stress_scenarios = stress_result.get('stress_scenarios', {})
                if stress_scenarios:
                    metrics['worst_case_loss'] = min(
                        scenario.get('total_loss_pct', 0)
                        for scenario in stress_scenarios.values()
                    )
                metrics['resilience_score'] = stress_result.get('resilience_score', 50)
            
            if corr_result:
                regime_data = corr_result.get('regime_correlations', {})
                regime_sens = regime_data.get('regime_sensitivity', {})
                metrics['crisis_multiplier'] = regime_sens.get('crisis_correlation_multiplier', 1)
                
                market_regimes = regime_data.get('market_regime_correlations', {})
                bull_corr = market_regimes.get('bull', {}).get('avg_correlation', 0.5)
                metrics['avg_correlation'] = bull_corr
            
            st.session_state['portfolio_metrics'] = metrics
            st.session_state['insights_generated'] = True
    
    # Display insights
    if st.session_state.get('insights_generated'):
        metrics = st.session_state.get('portfolio_metrics', {})
        
        # Section 1: Portfolio Health Score
        st.header("Portfolio Health Assessment")
        
        health_score, score_breakdown = calculate_health_score(metrics)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            fig = create_health_gauge(health_score)
            st.plotly_chart(fig, width='stretch')
            
            if health_score >= 80:
                st.success("**Excellent** - Portfolio is well-optimized")
            elif health_score >= 65:
                st.info("**Good** - Minor improvements recommended")
            elif health_score >= 50:
                st.warning("**Fair** - Several improvements needed")
            else:
                st.error("**Poor** - Significant changes recommended")
        
        with col2:
            fig = create_score_breakdown(score_breakdown)
            st.plotly_chart(fig, width='stretch')
        
        st.markdown("---")
        
        # Section 2: Priority Actions
        st.header("🎯 Priority Actions")
        
        actions = generate_action_items(symbols, weights, metrics, period)
        
        if not actions:
            st.success("No critical issues identified - portfolio is well-structured")
        else:
            for i, action in enumerate(actions[:5], 1):
                urgency_color = {
                    'High': '🔴',
                    'Medium': '🟡',
                    'Low': '🟢'
                }.get(action['urgency'], '⚪')
                
                with st.expander(f"{urgency_color} **Action {i}:** {action['title']}", expanded=(i<=2)):
                    st.markdown(f"**Problem:** {action['issue']}")
                    st.markdown(f"**Impact:** {action['impact']}")
                    st.markdown(f"**Recommendation:** {action['recommendation']}")
                    st.markdown(f"**Expected Result:** {action['improvement']}")
        
        st.markdown("---")
        
        # Section 3: Portfolio Strengths
        st.header("✅ Portfolio Strengths")
        
        strengths = identify_strengths(metrics)
        
        cols = st.columns(2)
        for i, strength in enumerate(strengths):
            with cols[i % 2]:
                st.success(strength)
        
        st.markdown("---")
        
        # Section 4: Key Metrics Summary
        st.header("📊 Key Metrics Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sharpe = metrics.get('sharpe_ratio', 0)
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.2f}",
                delta="Good" if sharpe > 1.0 else "Needs Work",
                delta_color="normal" if sharpe > 1.0 else "inverse"
            )
        
        with col2:
            volatility = metrics.get('annual_volatility', 0)
            st.metric(
                "Volatility",
                f"{volatility*100:.1f}%",
                delta="Low" if volatility < 0.20 else "High",
                delta_color="normal" if volatility < 0.20 else "inverse"
            )
        
        with col3:
            resilience = metrics.get('resilience_score', 0)
            st.metric(
                "Stress Resilience",
                f"{resilience:.0f}/100",
                delta="Strong" if resilience > 80 else "Weak",
                delta_color="normal" if resilience > 80 else "inverse"
            )
        
        with col4:
            crisis_mult = metrics.get('crisis_multiplier', 1)
            st.metric(
                "Crisis Protection",
                f"{crisis_mult:.1f}x",
                delta="Stable" if crisis_mult < 1.5 else "Vulnerable",
                delta_color="normal" if crisis_mult < 1.5 else "inverse"
            )
        
        st.markdown("---")
        
        # Section 5: Next Steps
        st.header("🚀 Recommended Next Steps")
        
        st.markdown("""
        1. **Review Priority Actions** - Address high-urgency items first
        2. **Explore Portfolio Analysis** - See optimized weight recommendations
        3. **Check Correlation Analytics** - Understand diversification structure
        4. **Run Stress Testing** - Validate changes under extreme scenarios
        5. **Monitor Regularly** - Re-run insights monthly or after major trades
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📈 Go to Portfolio Optimization", width='stretch'):
                st.switch_page("pages/1_Portfolio_Analysis.py")
        with col2:
            if st.button("⚠️ View Detailed Risk Analysis", width='stretch'):
                st.switch_page("pages/2_Risk_Analytics.py")
    
    else:
        st.info("Click '🔍 Generate Insights' in the sidebar to analyze your portfolio")
        
        # Show sample insights
        st.subheader("What You'll Get:")
        st.markdown("""
        - **Portfolio Health Score** - Overall assessment from 0-100
        - **Priority Action Items** - Ranked recommendations for improvement
        - **Strength Analysis** - What your portfolio does well
        - **Risk Summary** - Key vulnerability areas
        - **Clear Next Steps** - Specific actions to take
        """)

if __name__ == "__main__":
    main()