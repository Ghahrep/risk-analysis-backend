"""
Risk Analytics Page - Enhanced UX for User Testing
Key Improvements: Auto-load optimized weights, scenario interpretations, action bridges
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

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
    show_weight_summary,
    show_empty_state,
    add_footer_tip
)
from utils.metric_benchmarks import display_metric_with_benchmark

st.set_page_config(page_title="Risk Analytics", page_icon="⚠️", layout="wide")

initialize_portfolio()
api_client = get_risk_api_client()

def load_example_portfolio(portfolio_type):
    """Load example portfolios for quick testing"""
    examples = {
        "tech_growth": {
            "symbols": ["AAPL", "MSFT", "GOOGL", "NVDA"],
            "weights": [0.30, 0.30, 0.25, 0.15],
            "name": "Tech Growth"
        },
        "defensive": {
            "symbols": ["JNJ", "PG", "KO", "MCD"],
            "weights": [0.30, 0.30, 0.20, 0.20],
            "name": "Defensive"
        },
        "diversified": {
            "symbols": ["VTI", "BND", "VEA", "VWO", "GLD"],
            "weights": [0.35, 0.25, 0.20, 0.10, 0.10],
            "name": "Diversified"
        }
    }
    
    if portfolio_type in examples:
        portfolio = examples[portfolio_type]
        set_portfolio(portfolio["symbols"], portfolio["weights"])
        st.success(f"✓ {portfolio['name']} portfolio loaded!")
        time.sleep(0.5)
        st.rerun()

def interpret_stress_scenario(scenario_name, loss_pct):
    """NEW: Provide plain-language interpretation of stress scenarios"""
    interpretations = {
        "covid_2020": {
            "description": "March 2020 COVID-19 market crash",
            "context": "A rapid, pandemic-driven selloff causing 30-35% market decline in weeks",
            "comparison": "One of the fastest market drops in history"
        },
        "financial_crisis_2008": {
            "description": "2008 Global Financial Crisis", 
            "context": "Banking system collapse causing 50%+ equity market decline over 18 months",
            "comparison": "The worst financial crisis since the Great Depression"
        },
        "dotcom_2000": {
            "description": "2000-2002 Dot-com bubble burst",
            "context": "Tech-heavy NASDAQ fell 78% over 2.5 years",
            "comparison": "Devastating for growth stocks, moderate for diversified portfolios"
        },
        "rate_shock_2022": {
            "description": "2022 Rate hike cycle",
            "context": "Fed raising rates from 0% to 5% causing bond/stock decline",
            "comparison": "Particularly painful for growth stocks and bonds"
        },
        "black_monday_1987": {
            "description": "October 1987 Black Monday",
            "context": "Single-day 22% market crash, fastest in history",
            "comparison": "Extreme one-day event, rapid recovery"
        }
    }
    
    scenario_key = scenario_name.lower().replace(' ', '_')
    info = interpretations.get(scenario_key, {
        "description": scenario_name,
        "context": "Historical stress event",
        "comparison": "Significant market disruption"
    })
    
    # Assess severity
    if abs(loss_pct) > 30:
        severity = "SEVERE"
        color = "🔴"
        advice = "Your portfolio would face catastrophic losses. **Urgent action needed**: Add defensive assets (bonds, gold), reduce equity concentration, or implement hedging strategies."
    elif abs(loss_pct) > 20:
        severity = "HIGH"
        color = "🟠"
        advice = "Your portfolio would suffer major losses. **Consider**: Increasing diversification across asset classes, adding low-correlation assets, or reducing position sizes."
    elif abs(loss_pct) > 10:
        severity = "MODERATE"
        color = "🟡"
        advice = "Your portfolio shows moderate vulnerability. This is typical for equity-heavy allocations. **Optional**: Add some defensive positions to improve resilience."
    else:
        severity = "LOW"
        color = "🟢"
        advice = "Your portfolio demonstrates good resilience to this scenario. Current diversification appears adequate."
    
    return {
        "description": info["description"],
        "context": info["context"],
        "comparison": info["comparison"],
        "severity": severity,
        "color": color,
        "advice": advice
    }

def main():
    inject_custom_css()
    
    add_page_header(
        "Risk Analytics",
        "Stress test your portfolio and quantify potential losses",
        "⚠️"
    )
    
    # NEW: Check if coming from Portfolio Analysis with optimized weights
    if 'pending_stress_test' in st.session_state:
        pending = st.session_state['pending_stress_test']
        st.info("""
        🎯 **Testing Optimized Portfolio**: You're analyzing the optimized allocation from Portfolio Analysis. 
        This helps verify if optimization improved resilience to market stress.
        """)
        set_portfolio(pending['symbols'], pending['weights'])
        del st.session_state['pending_stress_test']
        # Auto-trigger analysis
        st.session_state['auto_run_stress'] = True
    
    # Sidebar configuration
    with st.sidebar:
        add_sidebar_branding()
        
        st.markdown("### 🎯 What do you want to know?")
        
        goal_buttons = {
            "How bad could losses get?": "stress",
            "What's my daily risk exposure?": "var",
            "Will volatility increase?": "volatility",
            "Overall risk assessment": "comprehensive"
        }
        
        for question, analysis in goal_buttons.items():
            if st.button(question, use_container_width=True, key=f"goal_{analysis}"):
                st.session_state['selected_analysis'] = analysis
        
        st.markdown("---")
        
        # Preset selector
        st.markdown("### Quick Load")
        preset_options = ["Custom Portfolio"] + list(list_presets().values())
        preset_name = st.selectbox("Portfolio", preset_options, label_visibility="collapsed")
        
        if preset_name != "Custom Portfolio":
            preset_key = [k for k, v in list_presets().items() if v == preset_name][0]
            preset = get_preset(preset_key)
            
            if st.button("📥 Load", key="load_preset_risk", use_container_width=True):
                set_portfolio(preset["symbols"], preset["weights"])
                st.success("✓ Loaded!")
                time.sleep(0.5)
                st.rerun()
        
        st.markdown("---")
        
        # Analysis period
        st.markdown("### Analysis Period")
        period = st.selectbox(
            "Period",
            ["6months", "1year", "2years"],
            index=1,
            format_func=lambda x: x.replace("months", " Months").replace("year", " Year").replace("s", "s").title(),
            label_visibility="collapsed"
        )
    
    symbols, weights = get_portfolio()
    
    # Determine which analysis to show
    analysis_type = st.session_state.get('selected_analysis', 'stress')
    
    # Map to display names
    analysis_map = {
        'stress': 'Stress Testing',
        'var': 'VaR Analysis',
        'volatility': 'Volatility Forecasting',
        'comprehensive': 'Comprehensive Risk'
    }
    
    # IMPROVED EMPTY STATE
    if not symbols:
        show_empty_state(
            icon="⚠️",
            title="Risk Analysis Ready",
            message="Load a portfolio to begin stress testing and risk analysis"
        )
        
        st.markdown("### 🚀 Quick Start")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 12px; border: 2px solid rgba(220, 53, 69, 0.2);'>
                <h4 style='color: #dc3545; margin-top: 0;'>📈 Tech Growth</h4>
                <p style='color: #808495; font-size: 0.9rem;'>High risk, high reward<br>Expected stress loss: 25-35%</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Test Tech Portfolio", key="try_tech", use_container_width=True):
                load_example_portfolio("tech_growth")
        
        with col2:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 12px; border: 2px solid rgba(40, 167, 69, 0.2);'>
                <h4 style='color: #28a745; margin-top: 0;'>🛡️ Defensive</h4>
                <p style='color: #808495; font-size: 0.9rem;'>Lower risk, stable<br>Expected stress loss: 10-20%</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Test Defensive Portfolio", key="try_defensive", use_container_width=True):
                load_example_portfolio("defensive")
        
        with col3:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 12px; border: 2px solid rgba(23, 162, 184, 0.2);'>
                <h4 style='color: #17a2b8; margin-top: 0;'>🌍 Diversified</h4>
                <p style='color: #808495; font-size: 0.9rem;'>Balanced approach<br>Expected stress loss: 15-25%</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Test Diversified Portfolio", key="try_diversified", use_container_width=True):
                load_example_portfolio("diversified")
        
        return
    
    # Validate portfolio
    is_valid, error_msg = validate_portfolio(symbols, weights)
    if not is_valid:
        st.error(f"⚠️ {error_msg}")
        return
    
    # Portfolio overview
    st.markdown("## Current Portfolio")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Holdings", len(symbols))
    with col2:
        max_weight = max(weights)
        st.metric("Largest Position", f"{max_weight:.1%}")
    with col3:
        concentration = sum(w**2 for w in weights)
        st.metric("Concentration", f"{concentration:.3f}",
                 help="Lower = more diversified")
    with col4:
        # NEW: Portfolio type indicator
        if max_weight > 0.5:
            port_type = "Concentrated"
        elif concentration < 0.2:
            port_type = "Diversified"
        else:
            port_type = "Balanced"
        st.metric("Type", port_type)
    
    st.markdown("---")
    
    # Analysis type selector
    st.markdown("## Select Analysis Type")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔥 Stress Testing", type="primary" if analysis_type == 'stress' else "secondary", use_container_width=True):
            st.session_state['selected_analysis'] = 'stress'
            st.rerun()
    
    with col2:
        if st.button("📊 VaR Analysis", type="primary" if analysis_type == 'var' else "secondary", use_container_width=True):
            st.session_state['selected_analysis'] = 'var'
            st.rerun()
    
    with col3:
        if st.button("📈 Volatility Forecast", type="primary" if analysis_type == 'volatility' else "secondary", use_container_width=True):
            st.session_state['selected_analysis'] = 'volatility'
            st.rerun()
    
    with col4:
        if st.button("⚠️ Full Risk Report", type="primary" if analysis_type == 'comprehensive' else "secondary", use_container_width=True):
            st.session_state['selected_analysis'] = 'comprehensive'
            st.rerun()
    
    st.markdown("---")
    
    # Run analysis button
    run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=False)
    
    # NEW: Auto-run if coming from optimization
    if st.session_state.get('auto_run_stress') and analysis_type == 'stress':
        run_analysis = True
        del st.session_state['auto_run_stress']
    
    # STRESS TESTING
    if analysis_type == 'stress':
        st.markdown("## 🔥 Stress Testing Analysis")
        st.caption("See how your portfolio performs during historical crisis scenarios")
        
        if run_analysis:
            with st.spinner("🔥 Running stress tests across 7 crisis scenarios... 15-20 seconds"):
                result = safe_api_call(
                    lambda: api_client.stress_test(symbols, weights),
                    error_context="stress testing"
                )
                
                if result:
                    st.session_state['stress_result'] = result
                    st.success("✓ Stress testing complete!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Stress testing failed. Verify symbols have sufficient historical data.")
        
        if 'stress_result' in st.session_state:
            result = st.session_state['stress_result']
            
            stress_scenarios = result.get('stress_scenarios', {})
            worst_case = result.get('worst_case_scenario', 'Unknown')
            resilience_score = result.get('resilience_score', 0)
            
            if not stress_scenarios:
                st.warning("⚠️ No stress test results available.")
                return
            
            # Key metrics
            st.markdown("### Overall Assessment")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Color-code resilience score
                if resilience_score > 75:
                    color = "normal"
                    status = "Strong 💪"
                elif resilience_score > 60:
                    color = "off"
                    status = "Moderate 👍"
                else:
                    color = "inverse"
                    status = "Weak ⚠️"
                
                st.metric(
                    "Resilience Score",
                    f"{resilience_score:.0f}/100",
                    delta=status,
                    delta_color=color
                )
            
            with col2:
                worst_loss = stress_scenarios.get(worst_case, {}).get('total_loss_pct', 0)
                st.metric(
                    "Worst Case Loss",
                    f"{abs(worst_loss):.1f}%",
                    delta=worst_case.replace('_', ' ').title()
                )
            
            with col3:
                avg_loss = sum(s.get('total_loss_pct', 0) for s in stress_scenarios.values()) / len(stress_scenarios)
                st.metric(
                    "Average Crisis Loss",
                    f"{abs(avg_loss):.1f}%",
                    help="Average loss across all 7 scenarios"
                )
            
            # NEW: Plain-language interpretation
            worst_interpretation = interpret_stress_scenario(worst_case, worst_loss)
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(220, 53, 69, 0.1) 0%, rgba(255, 193, 7, 0.1) 100%); 
                        padding: 1.5rem; border-radius: 12px; border-left: 4px solid #dc3545; margin: 1rem 0;'>
                <h4 style='margin-top: 0;'>{worst_interpretation['color']} Worst Case: {worst_interpretation['description']}</h4>
                <p><strong>What happened:</strong> {worst_interpretation['context']}</p>
                <p><strong>Your portfolio impact:</strong> {abs(worst_loss):.1f}% loss - {worst_interpretation['severity']} risk</p>
                <p style='margin-bottom: 0;'><strong>Action item:</strong> {worst_interpretation['advice']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Scenario visualization
            st.markdown("### Scenario Breakdown")
            
            scenario_data = []
            for scenario_name, metrics in stress_scenarios.items():
                loss = abs(metrics.get('total_loss_pct', 0))
                scenario_data.append({
                    'Scenario': scenario_name.replace('_', ' ').title(),
                    'Loss %': loss,
                    'Severity': 'Severe' if loss > 25 else 'High' if loss > 15 else 'Moderate'
                })
            
            df = pd.DataFrame(scenario_data).sort_values('Loss %', ascending=False)
            
            fig = px.bar(
                df,
                x='Scenario',
                y='Loss %',
                title="Portfolio Loss Across Crisis Scenarios",
                color='Loss %',
                color_continuous_scale='Reds',
                text='Loss %'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed scenarios with interpretations
            with st.expander("📋 Detailed Scenario Analysis"):
                for scenario_name, metrics in sorted(stress_scenarios.items(), 
                                                    key=lambda x: x[1].get('total_loss_pct', 0)):
                    loss_pct = metrics.get('total_loss_pct', 0)
                    interp = interpret_stress_scenario(scenario_name, loss_pct)
                    
                    st.markdown(f"**{interp['color']} {interp['description']}**")
                    st.caption(interp['context'])
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Loss", f"{abs(loss_pct):.1f}%")
                    with col2:
                        st.metric("Max Daily Loss", f"{abs(metrics.get('max_daily_loss_pct', 0)):.1f}%")
                    with col3:
                        st.metric("Recovery Est.", f"{metrics.get('recovery_estimate_days', 0):.0f} days")
                    st.markdown("---")
            
            # ACTION BRIDGE
            st.markdown("### 🎯 Next Steps")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Reduce Risk")
                if abs(worst_loss) > 20:
                    st.caption("High vulnerability detected")
                    if st.button("🎯 Optimize for Lower Risk", use_container_width=True):
                        st.switch_page("pages/1_Portfolio_Analysis.py")
                else:
                    st.caption("Stress resilience looks acceptable")
                    st.button("✓ No action needed", disabled=True, use_container_width=True)
            
            with col2:
                st.markdown("#### Understand Diversification")
                st.caption("Check if holdings are too correlated")
                if st.button("🔗 Analyze Correlations", use_container_width=True):
                    st.switch_page("pages/3_Correlation_Analysis.py")
            
            with col3:
                st.markdown("#### Full Report")
                st.caption("Get comprehensive health assessment")
                if st.button("💡 Generate Insights", use_container_width=True):
                    st.switch_page("pages/6_Portfolio_Insights.py")
    
    # VAR ANALYSIS
    elif analysis_type == 'var':
        st.markdown("## 📊 Value at Risk (VaR) Analysis")
        st.caption("Quantify daily portfolio risk with statistical confidence")
        
        confidence_level = st.select_slider(
            "Confidence Level",
            options=[0.90, 0.95, 0.99],
            value=0.95,
            format_func=lambda x: f"{x:.0%} Confidence"
        )
        
        st.info(f"""
        💡 **What this means:** With {confidence_level:.0%} confidence, your daily loss will not exceed the VaR threshold. 
        In other words, {confidence_level:.0%} of trading days should have losses less than VaR.
        """)
        
        if run_analysis:
            with st.spinner("📊 Calculating VaR metrics... 5-10 seconds"):
                result = safe_api_call(
                    lambda: api_client.calculate_var(symbols, weights, confidence_level),
                    error_context="VaR calculation"
                )
                
                if result:
                    st.session_state['var_result'] = result
                    st.success("✓ VaR calculation complete!")
                    time.sleep(0.5)
                    st.rerun()
        
        if 'var_result' in st.session_state:
            result = st.session_state['var_result']
            var_estimates = result.get('var_cvar_estimates', {})
            confidence_key = f"{int(confidence_level*100)}%"
            var_data = var_estimates.get(confidence_key, {})
            
            if var_data:
                col1, col2, col3 = st.columns(3)
                
                var_value = var_data.get('var', 0)
                cvar_value = var_data.get('cvar', 0)
                
                with col1:
                    st.metric(
                        f"Daily VaR ({confidence_level:.0%})",
                        f"{abs(var_value):.2%}",
                        help="Maximum expected daily loss at this confidence"
                    )
                
                with col2:
                    st.metric(
                        f"CVaR ({confidence_level:.0%})",
                        f"{abs(cvar_value):.2%}",
                        help="Average loss when VaR threshold is breached"
                    )
                
                with col3:
                    portfolio_value = st.number_input(
                        "Portfolio Value ($)",
                        min_value=1000,
                        value=100000,
                        step=10000,
                        format="%d"
                    )
                
                # Dollar risk
                st.markdown("### 💵 Dollar Risk Exposure")
                
                dollar_var = abs(var_value * portfolio_value)
                dollar_cvar = abs(cvar_value * portfolio_value)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        f"Daily VaR on ${portfolio_value:,.0f}",
                        f"${dollar_var:,.0f}",
                        help="Maximum expected dollar loss per day"
                    )
                
                with col2:
                    st.metric(
                        f"CVaR on ${portfolio_value:,.0f}",
                        f"${dollar_cvar:,.0f}",
                        help="Average dollar loss in worst scenarios"
                    )
                
                # NEW: Interpretation
                with st.expander("🤖 What does this mean?", expanded=True):
                    days_per_year = 252
                    expected_breaches = days_per_year * (1 - confidence_level)
                    
                    st.markdown(f"""
                    **Daily Risk Profile:**
                    - On a typical day, with {confidence_level:.0%} confidence, you won't lose more than **${dollar_var:,.0f}** ({abs(var_value):.2%})
                    - However, about **{expected_breaches:.0f} days per year**, losses could exceed this threshold
                    - When that happens, average loss is **${dollar_cvar:,.0f}** ({abs(cvar_value):.2%})
                    
                    **Risk Assessment:**
                    """)
                    
                    if abs(var_value) > 0.05:
                        st.warning("🟠 **High daily risk.** Your portfolio can swing 5%+ in a single day. Consider reducing volatility or position sizes.")
                    elif abs(var_value) > 0.03:
                        st.info("🟡 **Moderate daily risk.** Typical for equity portfolios. Monitor during volatile periods.")
                    else:
                        st.success("🟢 **Low daily risk.** Your portfolio has good daily stability.")
    
    # VOLATILITY FORECASTING
    elif analysis_type == 'volatility':
        st.markdown("## 📈 Volatility Forecasting (GARCH)")
        st.caption("Predict future volatility trends using time-series models")
        
        forecast_horizon = st.slider(
            "Forecast Horizon",
            min_value=10,
            max_value=60,
            value=30,
            help="Days to forecast ahead"
        )
        
        if run_analysis:
            with st.spinner("📈 Running GARCH volatility model... 10-15 seconds"):
                result = safe_api_call(
                    lambda: api_client.forecast_volatility_garch(symbols, forecast_horizon, period),
                    error_context="volatility forecasting"
                )
                
                if result:
                    st.session_state['volatility_forecast'] = result
                    st.success("✓ Volatility forecast complete!")
                    time.sleep(0.5)
                    st.rerun()
        
        if 'volatility_forecast' in st.session_state:
            result = st.session_state['volatility_forecast']
            vol_data = result.get('volatility_forecast', {})
            
            if vol_data:
                col1, col2, col3 = st.columns(3)
                
                current_vol = vol_data.get('current_volatility', 0)
                forecast_vol = vol_data.get('forecast_mean', 0)
                trend = vol_data.get('trend', 'stable')
                
                with col1:
                    st.metric("Current Volatility", f"{current_vol:.2%}")
                
                with col2:
                    change = forecast_vol - current_vol
                    st.metric(
                        "Forecast Volatility",
                        f"{forecast_vol:.2%}",
                        delta=f"{change:+.2%}"
                    )
                
                with col3:
                    trend_emoji = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}.get(trend, "➡️")
                    st.metric("Trend", f"{trend_emoji} {trend.capitalize()}")
                
                # NEW: Trading implications
                st.markdown("### 📊 Market Implications")
                
                if trend == 'increasing':
                    st.warning(f"""
                    **⚠️ Rising Volatility Ahead**
                    
                    Volatility expected to increase from {current_vol:.2%} to {forecast_vol:.2%}.
                    
                    **What this means:**
                    - Larger daily price swings likely
                    - Higher risk of drawdowns
                    - Options premiums will increase
                    
                    **Consider:**
                    - Reducing position sizes
                    - Adding defensive assets (bonds, gold)
                    - Setting tighter stop losses
                    - Waiting for volatility to stabilize before adding risk
                    """)
                elif trend == 'decreasing':
                    st.success(f"""
                    **✅ Declining Volatility Ahead**
                    
                    Volatility expected to decrease from {current_vol:.2%} to {forecast_vol:.2%}.
                    
                    **What this means:**
                    - Calmer market conditions ahead
                    - Lower drawdown risk
                    - Favorable environment for risk assets
                    
                    **Consider:**
                    - Favorable conditions for adding equity exposure
                    - Good timing for portfolio rebalancing
                    - Options selling strategies may be attractive
                    """)
                else:
                    st.info(f"""
                    **➡️ Stable Volatility Expected**
                    
                    Volatility forecast to remain near {current_vol:.2%}.
                    
                    **What this means:**
                    - Current conditions likely to persist
                    - No major regime change expected
                    
                    **Consider:**
                    - Continue current strategy
                    - Monitor for changes in forecast
                    """)
    
    # COMPREHENSIVE RISK
    else:  # comprehensive
        st.markdown("## ⚠️ Comprehensive Risk Analysis")
        st.caption("Full risk assessment across all metrics")
        
        if run_analysis:
            with st.spinner("⚠️ Running comprehensive analysis... 10-15 seconds"):
                result = safe_api_call(
                    lambda: api_client.analyze_risk(symbols, weights, period),
                    error_context="comprehensive risk"
                )
                
                if result:
                    st.session_state['comprehensive_risk'] = result
                    st.success("✓ Analysis complete!")
                    time.sleep(0.5)
                    st.rerun()
        
        if 'comprehensive_risk' in st.session_state:
            result = st.session_state['comprehensive_risk']
            metrics = result.get('metrics', {})
            
            if metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    display_metric_with_benchmark('sharpe_ratio', metrics.get('sharpe_ratio', 0))
                
                with col2:
                    display_metric_with_benchmark('annual_volatility', metrics.get('annualized_volatility', 0))
                
                with col3:
                    display_metric_with_benchmark('max_drawdown', metrics.get('max_drawdown_pct', 0))
                
                with col4:
                    display_metric_with_benchmark('var_95', metrics.get('portfolio_var_95', 0))
                
                # Overall assessment
                sharpe = metrics.get('sharpe_ratio', 0)
                vol = metrics.get('annualized_volatility', 0)
                
                st.markdown("### Overall Risk Assessment")
                
                if sharpe > 1.0 and vol < 0.20:
                    st.success("✅ **Excellent risk profile**: Strong risk-adjusted returns with manageable volatility")
                elif sharpe > 0.5:
                    st.info("ℹ️ **Good risk profile**: Reasonable risk-adjusted returns")
                else:
                    st.warning("⚠️ **Needs improvement**: Low risk-adjusted returns suggest rebalancing needed")
    
    # Footer
    st.markdown("---")
    add_footer_tip("💡 Run stress tests regularly, especially before major portfolio changes or market volatility spikes")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("An unexpected error occurred in Risk Analytics")
        request_logger.logger.exception("Unhandled exception in Risk Analytics page")
        with st.expander("🔍 Error Details"):
            st.code(str(e))