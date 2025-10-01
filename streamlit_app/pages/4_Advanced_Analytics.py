"""
Advanced Analytics Page - Enhanced UX for User Testing
Key Improvements: Plain-language interpretations, actionable insights, simplified navigation
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    show_empty_state,
    add_footer_tip
)

st.set_page_config(page_title="Advanced Analytics", page_icon="📊", layout="wide")

initialize_portfolio()
api_client = get_risk_api_client()

def load_example_portfolio(portfolio_type):
    """Load example portfolios"""
    examples = {
        "value_tilt": {
            "symbols": ["BRK.B", "JPM", "XOM", "PG", "JNJ"],
            "weights": [0.25, 0.20, 0.20, 0.20, 0.15],
            "name": "Value Tilt"
        },
        "growth_focused": {
            "symbols": ["NVDA", "TSLA", "SHOP", "SQ", "ROKU"],
            "weights": [0.25, 0.25, 0.20, 0.15, 0.15],
            "name": "Growth Focused"
        },
        "quality": {
            "symbols": ["AAPL", "MSFT", "V", "MA", "GOOGL"],
            "weights": [0.25, 0.25, 0.20, 0.15, 0.15],
            "name": "Quality"
        }
    }
    
    if portfolio_type in examples:
        portfolio = examples[portfolio_type]
        set_portfolio(portfolio["symbols"], portfolio["weights"])
        st.success(f"✓ {portfolio['name']} portfolio loaded!")
        time.sleep(0.5)
        st.rerun()

def interpret_risk_decomposition(systematic_pct, idiosyncratic_pct):
    """NEW: Plain-language risk decomposition interpretation"""
    systematic_ratio = systematic_pct / (systematic_pct + idiosyncratic_pct) if (systematic_pct + idiosyncratic_pct) > 0 else 0
    
    if systematic_ratio > 0.80:
        return {
            "level": "High Market Risk",
            "color": "🔴",
            "explanation": f"Your portfolio's risk is {systematic_ratio:.0%} tied to overall market movements. This is very high - your portfolio essentially acts as a leveraged market bet.",
            "implications": "When the market drops 10%, expect your portfolio to drop 10%+ too. Individual stock selection provides minimal risk reduction.",
            "action": "Add bonds, gold, or defensive sectors (utilities, consumer staples) to reduce market dependence. Consider inverse correlation assets."
        }
    elif systematic_ratio > 0.65:
        return {
            "level": "Moderate Market Risk",
            "color": "🟡",
            "explanation": f"Your portfolio has {systematic_ratio:.0%} systematic risk, which is typical for equity portfolios. About {(1-systematic_ratio):.0%} is stock-specific.",
            "implications": "Most of your risk comes from market direction, but individual holdings do matter. Diversification provides some benefit.",
            "action": "Current balance is acceptable for equity portfolio. Consider adding some bonds or alternatives to further reduce market sensitivity."
        }
    else:
        return {
            "level": "Balanced Risk Mix",
            "color": "🟢",
            "explanation": f"Only {systematic_ratio:.0%} of your risk is market-driven. The rest ({(1-systematic_ratio):.0%}) comes from individual stock characteristics.",
            "implications": "Good diversification! Your holdings have distinct risk profiles. Market moves won't dominate your returns as much.",
            "action": "Maintain this balance. Continue diversifying across uncorrelated stocks and sectors."
        }

def interpret_alpha(alpha, tracking_error, info_ratio):
    """NEW: Plain-language alpha interpretation"""
    if abs(alpha) < 0.02:  # Less than 2%
        return {
            "assessment": "Neutral Performance",
            "color": "🟡",
            "explanation": f"Your portfolio returned {alpha:+.1%} vs benchmark - essentially market-matching performance.",
            "meaning": "You're neither outperforming nor underperforming. Your returns track the market closely.",
            "verdict": "Acceptable for passive strategy. If actively managing, consider if extra effort is worthwhile."
        }
    elif alpha > 0.05:  # Positive 5%+
        if info_ratio > 0.5:
            return {
                "assessment": "Strong Outperformance",
                "color": "🟢",
                "explanation": f"Your portfolio beat the benchmark by {alpha:.1%} with an information ratio of {info_ratio:.2f}.",
                "meaning": "You're generating real alpha (skill-based returns) efficiently. The outperformance isn't just luck or excess volatility.",
                "verdict": "Excellent! You're adding value through security selection. Keep tracking what's working."
            }
        else:
            return {
                "assessment": "Risky Outperformance",
                "color": "🟡",
                "explanation": f"You beat the benchmark by {alpha:.1%}, but with high volatility ({tracking_error:.1%} tracking error).",
                "meaning": "Outperformance came with significant deviations from benchmark. Information ratio of {info_ratio:.2f} suggests inefficient alpha generation.",
                "verdict": "Positive alpha but risky. Consider if the volatility is worth the extra return."
            }
    else:  # Negative 5%+
        return {
            "assessment": "Underperformance",
            "color": "🔴",
            "explanation": f"Your portfolio lagged the benchmark by {abs(alpha):.1%}. This is significant underperformance.",
            "meaning": "The benchmark would have been a better choice. Your security selection or timing detracted value.",
            "verdict": "Action needed: Analyze what went wrong. Consider passive indexing or revising selection strategy."
        }

def main():
    inject_custom_css()
    
    add_page_header(
        "Advanced Analytics",
        "Institutional-grade portfolio analysis and attribution",
        "📊"
    )
    
    # Sidebar with goal-based navigation
    with st.sidebar:
        add_sidebar_branding()
        
        st.markdown("### 🎯 What do you want to understand?")
        st.caption("Choose your question")
        
        goal_questions = {
            "Where does my risk come from?": "risk",
            "Am I beating the market?": "performance",
            "How well diversified am I?": "metrics"
        }
        
        for question, analysis in goal_questions.items():
            if st.button(question, use_container_width=True, key=f"goal_{analysis}"):
                st.session_state['selected_advanced_analysis'] = analysis
        
        st.markdown("---")
        
        # Quick preset loader
        st.markdown("### Quick Load")
        preset_options = ["Custom Portfolio"] + list(list_presets().values())
        preset_name = st.selectbox("Portfolio", preset_options, label_visibility="collapsed")
        
        if preset_name != "Custom Portfolio":
            preset_key = [k for k, v in list_presets().items() if v == preset_name][0]
            preset = get_preset(preset_key)
            
            if st.button("📥 Load", key="load_preset_advanced", use_container_width=True):
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
    
    # Determine analysis type
    analysis_type = st.session_state.get('selected_advanced_analysis', 'risk')
    
    # IMPROVED EMPTY STATE
    if not symbols:
        show_empty_state(
            icon="📊",
            title="Advanced Analytics Ready",
            message="Load a portfolio to access institutional-grade analytics"
        )
        
        st.markdown("### 🚀 Example Portfolios")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 12px; border: 2px solid rgba(102, 126, 234, 0.2);'>
                <h4 style='color: #667eea; margin-top: 0;'>📉 Value Tilt</h4>
                <p style='color: #808495; font-size: 0.9rem;'>Traditional value stocks<br>Low P/E, high dividends</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Try Value", key="try_value", use_container_width=True):
                load_example_portfolio("value_tilt")
        
        with col2:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 12px; border: 2px solid rgba(102, 126, 234, 0.2);'>
                <h4 style='color: #667eea; margin-top: 0;'>📈 Growth</h4>
                <p style='color: #808495; font-size: 0.9rem;'>High momentum stocks<br>Revenue growth focus</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Try Growth", key="try_growth", use_container_width=True):
                load_example_portfolio("growth_focused")
        
        with col3:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 12px; border: 2px solid rgba(102, 126, 234, 0.2);'>
                <h4 style='color: #667eea; margin-top: 0;'>⭐ Quality</h4>
                <p style='color: #808495; font-size: 0.9rem;'>Strong fundamentals<br>High ROE, low debt</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Try Quality", key="try_quality", use_container_width=True):
                load_example_portfolio("quality")
        
        return
    
    # Validate portfolio
    is_valid, error_msg = validate_portfolio(symbols, weights)
    if not is_valid:
        st.error(f"⚠️ {error_msg}")
        return
    
    # Analysis type selector
    st.markdown("## Select Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Risk Sources", type="primary" if analysis_type == 'risk' else "secondary", use_container_width=True):
            st.session_state['selected_advanced_analysis'] = 'risk'
            st.rerun()
    
    with col2:
        if st.button("📈 vs Benchmark", type="primary" if analysis_type == 'performance' else "secondary", use_container_width=True):
            st.session_state['selected_advanced_analysis'] = 'performance'
            st.rerun()
    
    with col3:
        if st.button("📊 Diversification", type="primary" if analysis_type == 'metrics' else "secondary", use_container_width=True):
            st.session_state['selected_advanced_analysis'] = 'metrics'
            st.rerun()
    
    st.markdown("---")
    
    # Run analysis button
    run_analysis = st.button("🚀 Run Analysis", type="primary")
    
    # RISK ATTRIBUTION
    if analysis_type == 'risk':
        st.markdown("## 🔍 Risk Decomposition")
        st.caption("Where does your portfolio risk actually come from?")
        
        with st.expander("ℹ️ What is risk attribution?"):
            st.markdown("""
            **Risk attribution** breaks down your total portfolio risk into two components:
            
            - **Systematic Risk (Market Risk):** Risk from overall market movements. Affects all stocks together.
              Example: During 2008 crisis, market systematic risk dominated - nearly everything fell.
            
            - **Idiosyncratic Risk (Stock-Specific):** Risk unique to individual companies.
              Example: A drug trial failure affects only that pharma company, not the whole market.
            
            **Why it matters:** You can't diversify away systematic risk, but you CAN reduce idiosyncratic risk 
            by holding more uncorrelated stocks. This analysis shows which type dominates your portfolio.
            """)
        
        if run_analysis:
            with st.spinner("📊 Analyzing risk sources... 15-20 seconds"):
                result = safe_api_call(
                    lambda: api_client.risk_attribution(symbols, weights, period),
                    error_context="risk attribution"
                )
                
                if result:
                    st.session_state['risk_attribution'] = result
                    st.success("✓ Analysis complete!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Analysis failed. Verify US stocks with 1+ year history.")
        
        if 'risk_attribution' in st.session_state:
            result = st.session_state['risk_attribution']
            risk_attr = result.get('risk_attribution', {})
            
            if risk_attr:
                total_risk = risk_attr.get('total_risk_pct', 0)
                systematic_risk = risk_attr.get('systematic_risk_pct', 0)
                idiosyncratic_risk = risk_attr.get('idiosyncratic_risk_pct', 0)
                
                # Key metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Risk", f"{total_risk:.1%}")
                
                with col2:
                    st.metric("Market Risk", f"{systematic_risk:.1%}",
                             help="Can't be diversified away")
                
                with col3:
                    st.metric("Stock-Specific", f"{idiosyncratic_risk:.1%}",
                             help="Can be reduced by diversifying")
                
                # Visualization
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = go.Figure(data=[go.Pie(
                        labels=['Market Risk', 'Stock-Specific Risk'],
                        values=[systematic_risk, idiosyncratic_risk],
                        hole=0.4,
                        marker_colors=['#dc3545', '#667eea']
                    )])
                    fig.update_layout(
                        title="Risk Breakdown",
                        height=350,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # NEW: Interpretation
                    interpretation = interpret_risk_decomposition(systematic_risk, idiosyncratic_risk)
                    
                    st.markdown(f"""
                    <div style='background: rgba(102, 126, 234, 0.05); padding: 1rem; border-radius: 8px;'>
                        <h4>{interpretation['color']} {interpretation['level']}</h4>
                        <p style='font-size: 0.9rem; margin: 0.5rem 0;'>{interpretation['explanation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed interpretation
                with st.expander("🤖 What does this mean for me?", expanded=True):
                    st.markdown(f"""
                    **What's happening:**
                    {interpretation['implications']}
                    
                    **What you should do:**
                    {interpretation['action']}
                    """)
                
                # Factor contributions if available
                factor_contribs = risk_attr.get('factor_contributions', {})
                if factor_contribs:
                    st.markdown("### Factor Breakdown")
                    st.caption("Which market factors drive your risk?")
                    
                    contrib_df = pd.DataFrame([
                        {'Factor': k.replace('_', ' ').title(), 'Risk %': v}
                        for k, v in factor_contribs.items()
                    ])
                    
                    fig = px.bar(
                        contrib_df,
                        x='Factor',
                        y='Risk %',
                        title="Risk by Factor",
                        color='Risk %',
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
    
    # PERFORMANCE ATTRIBUTION
    elif analysis_type == 'performance':
        st.markdown("## 📈 Performance vs Benchmark")
        st.caption("Are you beating the market? Is it worth the extra risk?")
        
        benchmark = st.selectbox(
            "Choose Benchmark",
            ["SPY", "QQQ", "DIA"],
            format_func=lambda x: {"SPY": "S&P 500", "QQQ": "NASDAQ 100", "DIA": "Dow Jones"}[x]
        )
        
        with st.expander("ℹ️ Understanding performance metrics"):
            st.markdown("""
            **Alpha:** How much you beat (or lagged) the benchmark. 5% alpha = 5% better than market.
            
            **Tracking Error:** How much your returns deviate from benchmark. High tracking error = big bets.
            
            **Information Ratio:** Alpha divided by tracking error. Shows if outperformance is worth the extra volatility.
            - **>0.5:** Good alpha generation
            - **0.0-0.5:** Marginal value added
            - **<0.0:** Destroying value vs benchmark
            """)
        
        if run_analysis:
            with st.spinner("📈 Comparing to benchmark... 15-20 seconds"):
                result = safe_api_call(
                    lambda: api_client.performance_attribution(symbols, weights, benchmark, period),
                    error_context="performance attribution"
                )
                
                if result:
                    st.session_state['perf_attribution'] = result
                    st.success("✓ Analysis complete!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Analysis failed. Ensure 1+ year period and valid symbols.")
        
        if 'perf_attribution' in st.session_state:
            result = st.session_state['perf_attribution']
            perf_attr = result.get('performance_attribution', {})
            
            if perf_attr:
                total_return = perf_attr.get('total_return_pct', 0)
                alpha = perf_attr.get('alpha_pct', 0)
                risk_metrics = perf_attr.get('risk_adjusted_metrics', {})
                tracking_error = risk_metrics.get('tracking_error', 0)
                info_ratio = risk_metrics.get('information_ratio', 0)
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Your Return",
                        f"{total_return:.1%}",
                        delta="Positive" if total_return > 0 else "Negative"
                    )
                
                with col2:
                    st.metric(
                        "Alpha",
                        f"{alpha:+.1%}",
                        delta="Outperforming" if alpha > 0 else "Underperforming",
                        help="vs benchmark"
                    )
                
                with col3:
                    st.metric(
                        "Tracking Error",
                        f"{tracking_error:.1%}",
                        help="Deviation from benchmark"
                    )
                
                with col4:
                    st.metric(
                        "Information Ratio",
                        f"{info_ratio:.2f}",
                        delta="Strong" if info_ratio > 0.5 else "Weak"
                    )
                
                # NEW: Plain-language assessment
                assessment = interpret_alpha(alpha, tracking_error, info_ratio)
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                            padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0;'>
                    <h3 style='margin-top: 0;'>{assessment['color']} {assessment['assessment']}</h3>
                    <p><strong>Bottom line:</strong> {assessment['explanation']}</p>
                    <p><strong>What this means:</strong> {assessment['meaning']}</p>
                    <p style='margin-bottom: 0;'><strong>Verdict:</strong> {assessment['verdict']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Return comparison chart
                benchmark_return = total_return - alpha
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Benchmark',
                    x=['Return'],
                    y=[benchmark_return],
                    marker_color='#808495'
                ))
                fig.add_trace(go.Bar(
                    name='Your Portfolio',
                    x=['Return'],
                    y=[total_return],
                    marker_color='#667eea'
                ))
                fig.update_layout(
                    title=f"Return Comparison vs {benchmark}",
                    yaxis_title="Return %",
                    yaxis_tickformat='.1%',
                    height=350,
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # ADVANCED METRICS
    else:  # metrics
        st.markdown("## 📊 Diversification & Risk Metrics")
        st.caption("How well-constructed is your portfolio?")
        
        if run_analysis:
            with st.spinner("📊 Calculating comprehensive metrics... 10-15 seconds"):
                result = safe_api_call(
                    lambda: api_client.advanced_analytics(symbols, weights, period),
                    error_context="advanced metrics"
                )
                
                if result:
                    st.session_state['advanced_metrics'] = result
                    st.success("✓ Analysis complete!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Analysis failed. Use 1+ year period with 3-7 liquid stocks.")
        
        if 'advanced_metrics' in st.session_state:
            result = st.session_state['advanced_metrics']
            analytics = result.get('advanced_analytics', {})
            
            if analytics:
                div_metrics = analytics.get('diversification_metrics', {})
                
                st.markdown("### Diversification Quality")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    div_ratio = div_metrics.get('diversification_ratio', 0)
                    st.metric(
                        "Diversification Ratio",
                        f"{div_ratio:.2f}",
                        help="Higher = better diversification"
                    )
                
                with col2:
                    eff_assets = div_metrics.get('effective_num_assets', 0)
                    st.metric(
                        "Effective # of Assets",
                        f"{eff_assets:.1f}",
                        help="Independent positions equivalent"
                    )
                
                with col3:
                    avg_corr = div_metrics.get('avg_correlation', 0)
                    st.metric(
                        "Avg Correlation",
                        f"{avg_corr:.2f}",
                        delta="Low" if avg_corr < 0.5 else "High",
                        delta_color="normal" if avg_corr < 0.5 else "inverse"
                    )
                
                # Interpretation
                with st.expander("🤖 What do these numbers mean?", expanded=True):
                    if div_ratio > 1.3:
                        st.success(f"""
                        **Excellent diversification** (ratio: {div_ratio:.2f})
                        
                        Your portfolio is well-diversified. The diversification ratio above 1.3 means you're 
                        getting meaningful risk reduction from holding multiple assets. This is working as intended.
                        """)
                    elif div_ratio > 1.1:
                        st.info(f"""
                        **Moderate diversification** (ratio: {div_ratio:.2f})
                        
                        You have some diversification benefit, but it's modest. Your holdings still move together 
                        fairly often. Consider adding assets from different sectors or asset classes.
                        """)
                    else:
                        st.warning(f"""
                        **Limited diversification** (ratio: {div_ratio:.2f})
                        
                        Your holdings are too similar - diversification isn't providing much risk reduction. 
                        Your {len(symbols)} holdings act more like {eff_assets:.1f} independent positions. 
                        Add uncorrelated assets for real diversification benefit.
                        """)
                
                # Risk-adjusted performance
                st.markdown("### Risk-Adjusted Performance")
                
                risk_adj = analytics.get('risk_adjusted_performance', {})
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sortino = risk_adj.get('sortino_ratio', 0)
                    st.metric("Sortino Ratio", f"{sortino:.2f}",
                             delta="Strong" if sortino > 1.5 else "Weak")
                
                with col2:
                    calmar = risk_adj.get('calmar_ratio', 0)
                    st.metric("Calmar Ratio", f"{calmar:.2f}",
                             delta="Strong" if calmar > 1.0 else "Weak")
                
                with col3:
                    omega = risk_adj.get('omega_ratio', 0)
                    st.metric("Omega Ratio", f"{omega:.2f}",
                             delta="Strong" if omega > 1.2 else "Weak")
                
                st.caption("Sortino = Return/Downside risk | Calmar = Return/Max drawdown | Omega = Gains/Losses probability-weighted")
    
    # Footer
    st.markdown("---")
    add_footer_tip("💡 Advanced analytics reveal the 'why' behind your returns and risk - use these insights to refine your strategy")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("An unexpected error occurred in Advanced Analytics")
        request_logger.logger.exception("Unhandled exception in Advanced Analytics")
        with st.expander("🔍 Error Details"):
            st.code(str(e))