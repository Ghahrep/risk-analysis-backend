"""
Portfolio Dashboard - Enhanced UX for User Testing
Key Improvements: Goal-based navigation, better onboarding, action bridges
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
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
    show_weight_summary
)
from utils.metric_benchmarks import display_metric_with_benchmark

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

initialize_portfolio()
api_client = get_risk_api_client()

@st.cache_data(ttl=600)
def get_cached_risk_analysis(symbols_tuple, weights_tuple, period):
    """Cache analysis results to avoid unnecessary API calls"""
    symbols_list = list(symbols_tuple)
    weights_list = list(weights_tuple)
    return api_client.analyze_risk(symbols_list, weights_list, period)

def format_time_ago(timestamp):
    """Format timestamp as 'X hours/days ago'"""
    if not timestamp:
        return "Never"
    
    diff = datetime.now() - timestamp
    
    if diff < timedelta(minutes=1):
        return "Just now"
    elif diff < timedelta(hours=1):
        mins = int(diff.total_seconds() / 60)
        return f"{mins} min{'s' if mins != 1 else ''} ago"
    elif diff < timedelta(days=1):
        hours = int(diff.total_seconds() / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = diff.days
        return f"{days} day{'s' if days != 1 else ''} ago"

def create_mini_health_gauge(score):
    """Create compact health gauge for dashboard"""
    if score >= 80:
        color = "#28a745"
    elif score >= 65:
        color = "#17a2b8"
    elif score >= 50:
        color = "#ffc107"
    else:
        color = "#dc3545"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': "#ffc1c1"},
                {'range': [40, 70], 'color': "#fff3cd"},
                {'range': [70, 100], 'color': "#c3f7c3"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        },
        title={'text': "Health Score", 'font': {'size': 16}}
    ))
    fig.update_layout(
        height=200, 
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#333", 'family': "Arial"}
    )
    return fig

def load_example_portfolio(portfolio_type):
    """Load example portfolios"""
    examples = {
        "balanced": {
            "symbols": ["VTI", "BND", "VEA", "GLD"],
            "weights": [0.40, 0.30, 0.20, 0.10],
            "name": "Balanced"
        },
        "aggressive": {
            "symbols": ["QQQ", "ARKK", "VUG", "TSLA"],
            "weights": [0.35, 0.25, 0.25, 0.15],
            "name": "Aggressive Growth"
        },
        "conservative": {
            "symbols": ["BND", "AGG", "VTI", "GLD"],
            "weights": [0.40, 0.30, 0.20, 0.10],
            "name": "Conservative"
        }
    }
    
    if portfolio_type in examples:
        portfolio = examples[portfolio_type]
        set_portfolio(portfolio["symbols"], portfolio["weights"])
        st.success(f"✓ {portfolio['name']} portfolio loaded!")
        time.sleep(0.5)
        st.rerun()

def show_metric_interpretation(metric_name, value):
    """NEW: Add AI-style interpretation for metrics"""
    interpretations = {
        'sharpe_ratio': {
            'high': (1.5, "Your risk-adjusted returns are strong. This portfolio delivers good returns relative to volatility."),
            'medium': (0.5, "Decent risk-adjusted returns. There's room to improve the return/risk trade-off."),
            'low': (-float('inf'), "Poor risk-adjusted returns. Consider optimization to improve your return per unit of risk.")
        },
        'annual_volatility': {
            'high': (0.25, "High volatility means larger swings in portfolio value. This is typical for aggressive, equity-heavy portfolios."),
            'medium': (0.15, "Moderate volatility. Your portfolio has reasonable price fluctuation for a balanced approach."),
            'low': (-float('inf'), "Low volatility indicates stable returns. Good for conservative investors prioritizing capital preservation.")
        },
        'max_drawdown': {
            'high': (-0.20, "Large drawdowns mean significant unrealized losses during market stress. Consider adding defensive holdings."),
            'medium': (-0.10, "Moderate drawdowns. Your portfolio has experienced manageable declines during downturns."),
            'low': (-float('inf'), "Small drawdowns indicate strong downside protection. Your portfolio has weathered volatility well.")
        }
    }
    
    if metric_name in interpretations:
        thresholds = interpretations[metric_name]
        for level, (threshold, message) in thresholds.items():
            if value >= threshold:
                return message
    
    return None

def main():
    inject_custom_css()
    
    # Header
    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <h1 style='margin-bottom: 0.5rem;'>📊 Portfolio Dashboard</h1>
        <p style='color: #808495; font-size: 1.1rem; margin: 0;'>
            Your central hub for portfolio health and risk metrics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Staleness warning at top
    if 'last_analysis_time' in st.session_state:
        last_time = st.session_state['last_analysis_time']
        time_diff = datetime.now() - last_time
        
        if time_diff > timedelta(days=3):
            st.warning(f"⚠️ Dashboard data is {time_diff.days} days old. Consider running fresh analysis below.")
        elif time_diff > timedelta(days=1):
            st.info(f"ℹ️ Data is {time_diff.days} day{'s' if time_diff.days > 1 else ''} old. Last refreshed: {format_time_ago(last_time)}")
    
    # IMPROVED SIDEBAR: Goal-based navigation
    with st.sidebar:
        add_sidebar_branding()
        
        # NEW: What do you want to do? (Goal-based navigation)
        st.markdown("### 🎯 What do you want to do?")
        st.caption("Navigate by your goals, not features")
        
        goal_nav = {
            "Understand my risk": ("pages/2_Risk_Analytics.py", "See VaR, stress tests, volatility"),
            "Check my diversification": ("pages/3_Correlation_Analysis.py", "Analyze holdings relationships"),
            "Improve my returns": ("pages/1_Portfolio_Analysis.py", "Optimize portfolio weights"),
            "See overall health": ("pages/6_Portfolio_Insights.py", "Get health score & actions"),
            "Get AI guidance": ("pages/5_Behavioral_Analysis.py", "Behavioral insights")
        }
        
        for goal, (page, description) in goal_nav.items():
            if st.button(goal, use_container_width=True, key=f"nav_{goal}"):
                st.switch_page(page)
            st.caption(description)
        
        st.markdown("---")
        
        # Quick preset loader
        st.markdown("### Quick Load")
        preset_options = ["Select..."] + list(list_presets().values())
        preset_name = st.selectbox("Load Preset", preset_options, label_visibility="collapsed")
        
        if preset_name != "Select...":
            preset_key = [k for k, v in list_presets().items() if v == preset_name][0]
            preset = get_preset(preset_key)
            
            if st.button("📥 Load", key="load_preset_dash", use_container_width=True):
                with st.spinner("Loading..."):
                    time.sleep(0.3)
                    set_portfolio(preset["symbols"], preset["weights"])
                st.success("✓ Loaded!")
                time.sleep(1)
                st.rerun()
    
    # Get current portfolio
    symbols, weights = get_portfolio()
    
    # IMPROVED ONBOARDING - Streamlined empty state
    if not symbols:
        st.markdown("""
        <div style='text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%); border-radius: 16px; margin: 2rem 0;'>
            <div style='font-size: 4rem; margin-bottom: 1rem;'>📊</div>
            <h2 style='color: #667eea; margin-bottom: 1rem;'>Welcome to Portfolio Intelligence</h2>
            <p style='color: #808495; font-size: 1.1rem; max-width: 600px; margin: 0 auto;'>
                Get started in seconds with an example, or enter your holdings
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simplified: Just two clear options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Try an Example")
            st.caption("See the platform in action (recommended)")
            
            example_type = st.radio(
                "Choose portfolio type:",
                ["Balanced (60/40)", "Aggressive Growth", "Conservative"],
                label_visibility="collapsed"
            )
            
            mapping = {
                "Balanced (60/40)": "balanced",
                "Aggressive Growth": "aggressive", 
                "Conservative": "conservative"
            }
            
            if st.button("Load Example Portfolio", type="primary", use_container_width=True):
                load_example_portfolio(mapping[example_type])
        
        with col2:
            st.markdown("### ✍️ Enter Your Holdings")
            st.caption("Input your own portfolio")
            
            symbols_input = st.text_area(
                "Stock Symbols (one per line)", 
                height=150,
                placeholder="AAPL\nMSFT\nGOOGL\nAMZN",
                help="Enter 2-10 stock tickers"
            )
            
            if symbols_input:
                input_symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]
                
                if len(input_symbols) >= 2:
                    st.success(f"✓ {len(input_symbols)} symbols entered")
                    
                    # Equal weight by default
                    equal_weight = 1.0 / len(input_symbols)
                    input_weights = [equal_weight] * len(input_symbols)
                    
                    if st.button("💾 Create Portfolio", type="primary", use_container_width=True):
                        with st.spinner("Creating portfolio..."):
                            time.sleep(0.3)
                            set_portfolio(input_symbols, input_weights)
                        st.success("✓ Portfolio created! (Equal weights)")
                        st.info("💡 Adjust weights in Portfolio Analysis")
                        time.sleep(1.5)
                        st.rerun()
                else:
                    st.warning("⚠️ Enter at least 2 symbols")
        
        return  # Exit early if no portfolio
    
    # Portfolio loaded - show dashboard
    st.markdown("## Portfolio Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Holdings", len(symbols))
    
    with col2:
        largest = max(zip(symbols, weights), key=lambda x: x[1])
        st.metric("Largest Position", f"{largest[0]}", f"{largest[1]:.1%}")
    
    with col3:
        max_weight = max(weights)
        concentration = "High" if max_weight > 0.4 else "Balanced"
        st.metric("Concentration", f"{max_weight:.1%}", concentration,
                 delta_color="inverse" if max_weight > 0.4 else "off")
    
    with col4:
        if 'last_analysis_time' in st.session_state:
            last_time = st.session_state['last_analysis_time']
            time_ago = format_time_ago(last_time)
            st.metric("Last Updated", time_ago, 
                     "Current" if datetime.now() - last_time < timedelta(hours=1) else "Stale",
                     delta_color="normal" if datetime.now() - last_time < timedelta(hours=1) else "inverse")
        else:
            st.metric("Last Updated", "Never", "Run Analysis")
    
    # Portfolio allocation chart
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.pie(values=weights, names=symbols, 
                    title="Current Allocation", hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Holdings")
        holdings_df = pd.DataFrame({
            'Symbol': symbols,
            'Weight': [f"{w:.1%}" for w in weights]
        })
        st.dataframe(holdings_df, hide_index=True, use_container_width=True, height=300)
    
    st.markdown("---")
    
    # Key Risk Metrics WITH INTERPRETATIONS
    st.markdown("## Key Risk Metrics")
    
    if 'dashboard_metrics' in st.session_state:
        metrics = st.session_state['dashboard_metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            display_metric_with_benchmark('sharpe_ratio', metrics.get('sharpe_ratio', 0))
        
        with col2:
            display_metric_with_benchmark('annual_volatility', metrics.get('annual_volatility', 0))
        
        with col3:
            display_metric_with_benchmark('max_drawdown', metrics.get('max_drawdown', 0))
        
        with col4:
            display_metric_with_benchmark('var_95', metrics.get('var_95', 0))
        
        # NEW: Add interpretation for most critical metric
        sharpe = metrics.get('sharpe_ratio', 0)
        interpretation = show_metric_interpretation('sharpe_ratio', sharpe)
        if interpretation:
            st.info(f"💡 {interpretation}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Additional metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            display_metric_with_benchmark('sortino_ratio', metrics.get('sortino_ratio', 0), show_explanation=False)
        
        with col2:
            cvar = metrics.get('cvar_95', 0)
            abs_cvar = abs(cvar * 100)
            st.metric("CVaR (95%)", f"{abs_cvar:.2f}%")
        
        with col3:
            display_metric_with_benchmark('correlation', metrics.get('avg_correlation', 0.5), show_explanation=False)
        
        with col4:
            display_metric_with_benchmark('beta', metrics.get('beta', 1.0), show_explanation=False)
        
        st.markdown("---")
        
        # Health Status Section with ACTION BRIDGE
        if 'health_score' in st.session_state and 'priority_actions' in st.session_state:
            st.markdown("## Portfolio Health Status")
            
            health_score = st.session_state['health_score']
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                fig = create_mini_health_gauge(health_score)
                st.plotly_chart(fig, use_container_width=True)
                
                if health_score >= 80:
                    st.success("**Excellent** - Portfolio well-optimized")
                elif health_score >= 65:
                    st.info("**Good** - Minor improvements possible")
                elif health_score >= 50:
                    st.warning("**Fair** - Several improvements needed")
                else:
                    st.error("**Poor** - Significant changes recommended")
            
            with col2:
                st.markdown("#### Top Priority Actions")
                
                actions = st.session_state['priority_actions'][:3]
                if actions:
                    for i, action in enumerate(actions, 1):
                        urgency_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(action.get('urgency', 'Low'), "⚪")
                        with st.expander(f"{urgency_icon} {action.get('title', 'Action')}", expanded=(i==1)):
                            st.markdown(f"**Problem:** {action.get('issue', 'N/A')}")
                            st.markdown(f"**Recommendation:** {action.get('recommendation', 'N/A')}")
                            
                            # NEW: Action bridge based on recommendation
                            if 'optimize' in action.get('recommendation', '').lower():
                                if st.button("🎯 Go Optimize Now", key=f"action_opt_{i}"):
                                    st.switch_page("pages/1_Portfolio_Analysis.py")
                            elif 'stress' in action.get('recommendation', '').lower() or 'risk' in action.get('recommendation', '').lower():
                                if st.button("💥 Run Stress Test", key=f"action_stress_{i}"):
                                    st.switch_page("pages/2_Risk_Analytics.py")
                else:
                    st.success("No critical actions needed!")
                
                if st.button("📊 View Full Insights Report", use_container_width=True):
                    st.switch_page("pages/6_Portfolio_Insights.py")
            
            st.markdown("---")
    else:
        st.info("📊 Run analysis below to populate your dashboard with real-time metrics")
    
    # Quick Actions with better grouping
    st.markdown("## Quick Actions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Analyze & Understand")
        
        subcol1, subcol2 = st.columns(2)
        
        with subcol1:
            if st.button("🔍 Run Full Analysis", type="primary", use_container_width=True):
                with st.spinner("Running comprehensive analysis... 15-20 seconds"):
                    symbols_tuple = tuple(symbols)
                    weights_tuple = tuple(weights)
                    
                    result = safe_api_call(
                        lambda: get_cached_risk_analysis(symbols_tuple, weights_tuple, "1year"),
                        error_context="dashboard analysis"
                    )
                    
                    if result:
                        metrics_data = result.get('metrics', {})
                        st.session_state['dashboard_metrics'] = {
                            'sharpe_ratio': metrics_data.get('sharpe_ratio', 0),
                            'sortino_ratio': metrics_data.get('sortino_ratio', 0),
                            'annual_volatility': metrics_data.get('annualized_volatility', 0),
                            'max_drawdown': metrics_data.get('max_drawdown_pct', 0),
                            'var_95': metrics_data.get('portfolio_var_95', 0),
                            'cvar_95': metrics_data.get('portfolio_cvar_95', 0),
                            'avg_correlation': 0.5,
                            'beta': 1.0
                        }
                        st.session_state['last_analysis_time'] = datetime.now()
                        st.success("✓ Analysis complete!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Analysis failed. Check API connectivity.")
        
        with subcol2:
            if st.button("💡 Generate Insights", use_container_width=True):
                st.switch_page("pages/6_Portfolio_Insights.py")
    
    with col2:
        st.markdown("#### Take Action")
        
        if st.button("🎯 Optimize Weights", use_container_width=True):
            st.switch_page("pages/1_Portfolio_Analysis.py")
        
        if st.button("🔥 Stress Test", use_container_width=True):
            st.switch_page("pages/2_Risk_Analytics.py")
    
    # Footer with helpful info
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.caption("💡 Metrics refresh when you run analysis. Data is cached for 10 minutes for performance.")
    
    with col2:
        # NEW: Feedback mechanism
        feedback_col1, feedback_col2 = st.columns(2)
        with feedback_col1:
            if st.button("👍 Helpful", key="feedback_up"):
                st.session_state['feedback_given'] = True
                st.success("Thanks for feedback!")
        with feedback_col2:
            if st.button("👎 Not helpful", key="feedback_down"):
                st.session_state['feedback_given'] = True
                st.info("We'll improve this!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("An unexpected error occurred on Dashboard")
        request_logger.logger.exception("Unhandled exception in Dashboard")
        with st.expander("🔍 Error Details"):
            st.code(str(e))