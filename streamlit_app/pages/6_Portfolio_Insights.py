"""
Portfolio Insights Dashboard - Enhanced UX for User Testing
Key Improvements: Clear action bridges, prioritized recommendations, simplified flow
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.portfolio_manager import get_portfolio, set_portfolio, normalize_weights, initialize_portfolio
from utils.portfolio_presets import list_presets, get_preset
from utils.api_client import get_risk_api_client
from utils.error_handler import validate_portfolio, safe_api_call
from utils.request_logger import request_logger
from utils.styling import (
    inject_custom_css, 
    add_page_header, 
    add_sidebar_branding,
    show_empty_state,
    add_footer_tip
)

st.set_page_config(page_title="Portfolio Insights", page_icon="💡", layout="wide")

initialize_portfolio()
api_client = get_risk_api_client()

def load_example_portfolio(portfolio_type):
    """Load example portfolios"""
    examples = {
        "needs_work": {
            "symbols": ["TSLA", "ARKK", "COIN", "MSTR"],
            "weights": [0.40, 0.30, 0.20, 0.10],
            "name": "Needs Improvement"
        },
        "moderate": {
            "symbols": ["VTI", "QQQ", "VEA", "AGG"],
            "weights": [0.35, 0.30, 0.20, 0.15],
            "name": "Moderate"
        },
        "optimized": {
            "symbols": ["VTI", "BND", "VEA", "VWO", "GLD", "VNQ"],
            "weights": [0.30, 0.25, 0.15, 0.10, 0.10, 0.10],
            "name": "Well-Optimized"
        }
    }
    
    if portfolio_type in examples:
        portfolio = examples[portfolio_type]
        set_portfolio(portfolio["symbols"], portfolio["weights"])
        st.success(f"✓ {portfolio['name']} portfolio loaded!")
        time.sleep(0.5)
        st.rerun()

def calculate_health_score(metrics):
    """Calculate health score 0-100"""
    scores = {}
    
    # Risk-return (30 points)
    sharpe = metrics.get('sharpe_ratio', 0)
    if sharpe > 2.0:
        scores['risk_return'] = 30
    elif sharpe > 1.0:
        scores['risk_return'] = 15 + (sharpe - 1.0) * 15
    elif sharpe > 0:
        scores['risk_return'] = sharpe * 15
    else:
        scores['risk_return'] = 0
    
    # Diversification (25 points)
    correlation = metrics.get('avg_correlation', 0.5)
    scores['diversification'] = max(0, (1 - correlation) * 25)
    
    # Stress resilience (25 points)
    resilience = metrics.get('resilience_score', 50)
    scores['stress_resilience'] = resilience * 0.25
    
    # Volatility control (20 points)
    volatility = metrics.get('annual_volatility', 0.25)
    if volatility < 0.15:
        scores['volatility'] = 20
    elif volatility < 0.25:
        scores['volatility'] = 20 - ((volatility - 0.15) * 100)
    else:
        scores['volatility'] = max(0, 10 - ((volatility - 0.25) * 40))
    
    total = sum(scores.values())
    return min(100, max(0, total)), scores

def generate_action_items(symbols, weights, metrics):
    """Generate prioritized action items"""
    actions = []
    
    # Crisis correlation
    crisis_multiplier = metrics.get('crisis_multiplier', 1)
    if crisis_multiplier > 1.8:
        actions.append({
            'title': 'Fix: Crisis Correlation Spike',
            'problem': f'Your diversification breaks down during crashes ({crisis_multiplier:.1f}x increase)',
            'why_it_matters': 'When you need diversification most - during crises - it vanishes. All holdings drop together.',
            'what_to_do': 'Add defensive assets that move differently: Treasury bonds (TLT), gold (GLD), or managed futures',
            'tool': 'Correlation Analysis',
            'page': 'pages/3_Correlation_Analysis.py',
            'urgency': 'High'
        })
    
    # Worst case loss
    worst_loss = metrics.get('worst_case_loss', 0)
    if worst_loss < -25:
        actions.append({
            'title': 'Fix: Extreme Downside Risk',
            'problem': f'Portfolio could lose {abs(worst_loss):.0f}% in severe stress (like 2008)',
            'why_it_matters': 'A $100k portfolio could drop to ${100000 + (worst_loss * 1000):.0f}. Can you handle that emotionally?',
            'what_to_do': 'Reduce equity concentration or add protective puts on largest holdings',
            'tool': 'Stress Testing',
            'page': 'pages/2_Risk_Analytics.py',
            'urgency': 'High'
        })
    
    # Suboptimal Sharpe
    current_sharpe = metrics.get('current_sharpe', 0)
    optimal_sharpe = metrics.get('optimal_sharpe', 0)
    if optimal_sharpe > 0 and current_sharpe < optimal_sharpe * 0.85:
        improvement = ((optimal_sharpe - current_sharpe) / current_sharpe * 100) if current_sharpe > 0 else 0
        actions.append({
            'title': 'Improve: Risk-Adjusted Returns',
            'problem': f'Missing {improvement:.0f}% potential return for same risk level',
            'why_it_matters': f'Current Sharpe {current_sharpe:.2f} vs optimal {optimal_sharpe:.2f} - you can do better',
            'what_to_do': 'Rebalance to optimized weights shown in Portfolio Analysis',
            'tool': 'Portfolio Optimization',
            'page': 'pages/1_Portfolio_Analysis.py',
            'urgency': 'Medium'
        })
    
    # Concentration
    max_position = max(weights) if weights else 0
    if max_position > 0.4:
        actions.append({
            'title': 'Fix: Over-Concentration',
            'problem': f'Single position is {max_position*100:.0f}% of portfolio',
            'why_it_matters': 'One stock determines your fate. If it crashes, your portfolio crashes.',
            'what_to_do': f'Reduce to max 30% per holding. Add 2-3 uncorrelated positions.',
            'tool': 'Portfolio Optimization',
            'page': 'pages/1_Portfolio_Analysis.py',
            'urgency': 'High'
        })
    
    # Poor diversification
    correlation = metrics.get('avg_correlation', 0.5)
    if correlation > 0.7:
        actions.append({
            'title': 'Fix: Inadequate Diversification',
            'problem': f'Holdings move together {correlation:.0%} of the time',
            'why_it_matters': 'Your "diversified" portfolio acts like one big bet. Defeats the purpose.',
            'what_to_do': 'Add assets from different sectors, countries, or asset classes (bonds, international)',
            'tool': 'Correlation Analysis',
            'page': 'pages/3_Correlation_Analysis.py',
            'urgency': 'Medium'
        })
    
    return sorted(actions, key=lambda x: (0 if x['urgency'] == 'High' else 1, x['title']))

def identify_strengths(metrics):
    """Identify strengths"""
    strengths = []
    
    sharpe = metrics.get('sharpe_ratio', 0)
    if sharpe > 1.5:
        strengths.append(('🎯', 'Excellent Returns', f'Sharpe ratio of {sharpe:.2f} shows strong risk-adjusted performance'))
    
    correlation = metrics.get('avg_correlation', 0.5)
    if correlation < 0.45:
        strengths.append(('🌐', 'Well Diversified', f'Low correlation ({correlation:.2f}) means holdings move independently'))
    
    resilience = metrics.get('resilience_score', 0)
    if resilience > 80:
        strengths.append(('💪', 'Crisis Resilient', f'Score of {resilience:.0f}/100 shows strong stress protection'))
    
    crisis_mult = metrics.get('crisis_multiplier', 1)
    if crisis_mult < 1.4:
        strengths.append(('🛡️', 'Stable Correlations', f'Diversification holds up during crashes ({crisis_mult:.1f}x multiplier)'))
    
    volatility = metrics.get('annual_volatility', 0)
    if volatility < 0.20:
        strengths.append(('📊', 'Low Volatility', f'Annual volatility of {volatility*100:.0f}% provides smooth ride'))
    
    if not strengths:
        strengths.append(('✓', 'Baseline Met', 'Portfolio meets basic requirements'))
    
    return strengths

def create_health_gauge(score):
    """Create health gauge"""
    if score >= 80:
        color, status = "#28a745", "Excellent"
    elif score >= 65:
        color, status = "#17a2b8", "Good"
    elif score >= 50:
        color, status = "#ffc107", "Fair"
    else:
        color, status = "#dc3545", "Needs Work"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{status}</b>", 'font': {'size': 24}},
        number={'suffix': "/100", 'font': {'size': 48}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color, 'thickness': 0.75},
            'steps': [
                {'range': [0, 50], 'color': "#ffebee"},
                {'range': [50, 80], 'color': "#fff9c4"},
                {'range': [80, 100], 'color': "#e8f5e9"}
            ],
            'threshold': {'line': {'color': "red", 'width': 3}, 'thickness': 0.75, 'value': 60}
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def main():
    inject_custom_css()
    
    add_page_header(
        "Portfolio Health Report",
        "Your complete portfolio assessment with prioritized actions",
        "💡"
    )
    
    # Simplified sidebar
    with st.sidebar:
        add_sidebar_branding()
        
        st.markdown("### Quick Load")
        preset_options = ["Current Portfolio"] + list(list_presets().values())
        preset_name = st.selectbox("Portfolio", preset_options, label_visibility="collapsed")
        
        if preset_name != "Current Portfolio":
            preset_key = [k for k, v in list_presets().items() if v == preset_name][0]
            preset = get_preset(preset_key)
            
            if st.button("📥 Load", key="load_preset", use_container_width=True):
                set_portfolio(preset["symbols"], preset["weights"])
                st.success("✓ Loaded!")
                time.sleep(0.5)
                st.rerun()
        
        st.markdown("---")
        st.markdown("### Analysis Period")
        period = st.selectbox(
            "Period",
            ["1year", "2years"],
            format_func=lambda x: x.replace("year", " Year").replace("s", "s").title(),
            label_visibility="collapsed"
        )
    
    symbols, weights = get_portfolio()
    
    # IMPROVED EMPTY STATE
    if not symbols:
        show_empty_state(
            icon="💡",
            title="Portfolio Health Report Ready",
            message="Load a portfolio to get your complete health assessment"
        )
        
        st.markdown("### 🎯 See Different Health Scores")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 12px; border: 2px solid rgba(220, 53, 69, 0.3);'>
                <h4 style='color: #dc3545; margin-top: 0;'>❌ Poor (30-40/100)</h4>
                <p style='color: #808495; font-size: 0.9rem;'>High concentration, weak diversification</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("See Example", key="poor", use_container_width=True):
                load_example_portfolio("needs_work")
        
        with col2:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 12px; border: 2px solid rgba(255, 193, 7, 0.3);'>
                <h4 style='color: #ffc107; margin-top: 0;'>⚡ Good (65-75/100)</h4>
                <p style='color: #808495; font-size: 0.9rem;'>Solid foundation, room to optimize</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("See Example", key="moderate", use_container_width=True):
                load_example_portfolio("moderate")
        
        with col3:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 12px; border: 2px solid rgba(40, 167, 69, 0.3);'>
                <h4 style='color: #28a745; margin-top: 0;'>✓ Excellent (85+/100)</h4>
                <p style='color: #808495; font-size: 0.9rem;'>Well-diversified, low correlation</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("See Example", key="excellent", use_container_width=True):
                load_example_portfolio("optimized")
        
        return
    
    # Validate
    is_valid, error_msg = validate_portfolio(symbols, weights)
    if not is_valid:
        st.error(f"⚠️ {error_msg}")
        return
    
    # Generate insights button
    if st.button("🔍 Generate Health Report", type="primary", use_container_width=False):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Generating comprehensive health report..."):
            # Gather metrics
            status_text.text("⚠️ Analyzing risk... (1/4)")
            progress_bar.progress(25)
            
            risk_result = safe_api_call(
                lambda: api_client.analyze_risk(symbols, weights, period),
                error_context="risk analysis"
            )
            
            status_text.text("🎯 Optimizing... (2/4)")
            progress_bar.progress(50)
            
            opt_result = safe_api_call(
                lambda: api_client.optimize_portfolio(symbols, "max_sharpe", period),
                error_context="optimization"
            )
            
            status_text.text("🔥 Stress testing... (3/4)")
            progress_bar.progress(75)
            
            stress_result = safe_api_call(
                lambda: api_client.stress_test(symbols, weights),
                error_context="stress test"
            )
            
            status_text.text("🔗 Checking correlations... (4/4)")
            progress_bar.progress(90)
            
            corr_result = safe_api_call(
                lambda: api_client.regime_correlations(symbols, "volatility", period),
                error_context="correlations"
            )
            
            progress_bar.progress(100)
            
            # Consolidate metrics
            metrics = {}
            if risk_result:
                risk_metrics = risk_result.get('metrics', {})
                metrics['sharpe_ratio'] = risk_metrics.get('sharpe_ratio', 0)
                metrics['current_sharpe'] = risk_metrics.get('sharpe_ratio', 0)
                metrics['annual_volatility'] = risk_metrics.get('annualized_volatility', 0)
            
            if opt_result:
                metrics['optimal_sharpe'] = opt_result.get('optimization_results', {}).get('sharpe_ratio', 0)
            
            if stress_result:
                scenarios = stress_result.get('stress_scenarios', {})
                if scenarios:
                    try:
                        metrics['worst_case_loss'] = min(s.get('total_loss_pct', 0) for s in scenarios.values())
                    except:
                        metrics['worst_case_loss'] = 0
                metrics['resilience_score'] = stress_result.get('resilience_score', 50)
            
            if corr_result:
                regime_data = corr_result.get('regime_correlations', {})
                metrics['crisis_multiplier'] = regime_data.get('regime_sensitivity', {}).get('crisis_correlation_multiplier', 1)
                bull_corr = regime_data.get('market_regime_correlations', {}).get('bull', {}).get('avg_correlation', 0.5)
                metrics['avg_correlation'] = bull_corr
            
            status_text.empty()
            progress_bar.empty()
            
            if metrics:
                st.session_state['portfolio_metrics'] = metrics
                st.session_state['insights_generated'] = True
                st.success("✓ Report ready!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Unable to generate report. Verify symbols are valid US stocks with 1+ year history.")
    
    # Display insights
    if st.session_state.get('insights_generated'):
        metrics = st.session_state.get('portfolio_metrics', {})
        
        if not metrics:
            st.warning("No data available. Generate report again.")
            return
        
        # Health Score
        st.markdown("## Your Portfolio Health")
        
        health_score, score_breakdown = calculate_health_score(metrics)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            fig = create_health_gauge(health_score)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # What this means
            if health_score >= 80:
                st.markdown("""
                <div style='background: rgba(40, 167, 69, 0.1); padding: 1.5rem; border-radius: 12px;'>
                    <h3 style='margin-top: 0; color: #28a745;'>Excellent Portfolio</h3>
                    <p>Your portfolio is well-constructed with strong diversification, good risk-adjusted 
                    returns, and solid stress protection. Minor tweaks may help, but fundamentals are sound.</p>
                    <p style='margin-bottom: 0;'><strong>Recommendation:</strong> Monitor regularly and maintain 
                    this structure. Review quarterly for rebalancing opportunities.</p>
                </div>
                """, unsafe_allow_html=True)
            elif health_score >= 65:
                st.markdown("""
                <div style='background: rgba(23, 162, 184, 0.1); padding: 1.5rem; border-radius: 12px;'>
                    <h3 style='margin-top: 0; color: #17a2b8;'>Good Portfolio</h3>
                    <p>Solid foundation with room for improvement. You're avoiding major mistakes, but some 
                    optimizations could enhance performance and reduce risk.</p>
                    <p style='margin-bottom: 0;'><strong>Recommendation:</strong> Review priority actions below. 
                    Focus on 1-2 high-impact improvements first.</p>
                </div>
                """, unsafe_allow_html=True)
            elif health_score >= 50:
                st.markdown("""
                <div style='background: rgba(255, 193, 7, 0.1); padding: 1.5rem; border-radius: 12px;'>
                    <h3 style='margin-top: 0; color: #ffc107;'>Needs Improvement</h3>
                    <p>Your portfolio has meaningful issues that could hurt performance or expose you to 
                    unnecessary risk. Several improvements are needed.</p>
                    <p style='margin-bottom: 0;'><strong>Recommendation:</strong> Address high-priority actions 
                    immediately. Consider consulting a financial advisor if making major changes.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: rgba(220, 53, 69, 0.1); padding: 1.5rem; border-radius: 12px;'>
                    <h3 style='margin-top: 0; color: #dc3545;'>Significant Issues</h3>
                    <p>Your portfolio has structural problems that need immediate attention. Current construction 
                    exposes you to excessive risk or delivers poor risk-adjusted returns.</p>
                    <p style='margin-bottom: 0;'><strong>Recommendation:</strong> Take action on priority items 
                    now. Consider professional guidance before making major trades.</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Priority Actions
        st.markdown("## 🎯 What You Should Do")
        st.caption("Prioritized by impact")
        
        actions = generate_action_items(symbols, weights, metrics)
        
        if not actions:
            st.success("✓ No critical issues - portfolio structure is solid!")
        else:
            for i, action in enumerate(actions[:3], 1):  # Top 3 only
                urgency_icon = "🔴" if action['urgency'] == 'High' else "🟡"
                
                with st.expander(f"{urgency_icon} **Priority {i}:** {action['title']}", expanded=(i==1)):
                    st.markdown(f"**The problem:** {action['problem']}")
                    st.markdown(f"**Why it matters:** {action['why_it_matters']}")
                    st.markdown(f"**What to do:** {action['what_to_do']}")
                    
                    # Action button
                    if st.button(f"🔧 Fix with {action['tool']}", key=f"action_{i}", use_container_width=True):
                        st.switch_page(action['page'])
        
        st.markdown("---")
        
        # Strengths
        st.markdown("## ✅ What's Working Well")
        
        strengths = identify_strengths(metrics)
        
        for emoji, title, description in strengths:
            st.markdown(f"""
            <div style='background: rgba(40, 167, 69, 0.05); padding: 1rem; border-radius: 8px; 
                        border-left: 3px solid #28a745; margin-bottom: 0.5rem;'>
                <strong>{emoji} {title}</strong><br>
                <span style='color: #808495;'>{description}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Next Steps
        st.markdown("## 📋 Your Action Plan")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 1. Address Priorities")
            st.caption("Fix high-urgency items first")
            if actions:
                if st.button(f"🔧 Fix: {actions[0]['title'][:20]}...", use_container_width=True):
                    st.switch_page(actions[0]['page'])
            else:
                st.button("✓ No action needed", disabled=True, use_container_width=True)
        
        with col2:
            st.markdown("#### 2. Optimize Further")
            st.caption("Improve risk-return profile")
            if st.button("📈 Optimize Portfolio", use_container_width=True):
                st.switch_page("pages/1_Portfolio_Analysis.py")
        
        with col3:
            st.markdown("#### 3. Monitor & Adjust")
            st.caption("Track changes over time")
            if st.button("📊 View Analytics", use_container_width=True):
                st.switch_page("pages/4_Advanced_Analytics.py")
    
    # Footer
    st.markdown("---")
    add_footer_tip("💡 Re-run health report monthly or after major portfolio changes to stay informed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("An error occurred. Please try again.")
        request_logger.logger.exception("Unhandled exception in Portfolio Insights")
        with st.expander("🔍 Error Details"):
            st.code(str(e))