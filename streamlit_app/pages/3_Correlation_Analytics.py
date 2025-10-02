"""
Correlation Analysis Page - Enhanced UX for User Testing
Key Improvements: Plain-language interpretations, diversification scoring, action bridges
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
    show_empty_state,
    add_footer_tip
)

st.set_page_config(page_title="Correlation Analysis", page_icon="🔗", layout="wide")

initialize_portfolio()
api_client = get_risk_api_client()

def load_example_portfolio(portfolio_type):
    """Load example portfolios for correlation testing"""
    examples = {
        "sector_diverse": {
            "symbols": ["AAPL", "JPM", "XOM", "JNJ", "DIS"],
            "name": "Sector Diverse"
        },
        "tech_cluster": {
            "symbols": ["AAPL", "MSFT", "GOOGL", "META", "NVDA"],
            "name": "Tech Cluster"
        },
        "global_mix": {
            "symbols": ["SPY", "EFA", "EEM", "AGG", "GLD"],
            "name": "Global Mix"
        }
    }
    
    if portfolio_type in examples:
        portfolio = examples[portfolio_type]
        symbols = portfolio["symbols"]
        weights = [1.0/len(symbols)] * len(symbols)
        set_portfolio(symbols, weights)
        st.success(f"✓ {portfolio['name']} portfolio loaded!")
        time.sleep(0.5)
        st.rerun()

def interpret_diversification(avg_correlation, num_holdings):
    """NEW: Provide plain-language diversification assessment"""
    
    # Adjust thresholds based on number of holdings
    if num_holdings >= 8:
        high_threshold = 0.65
        low_threshold = 0.35
    elif num_holdings >= 5:
        high_threshold = 0.70
        low_threshold = 0.40
    else:
        high_threshold = 0.75
        low_threshold = 0.45
    
    if avg_correlation > high_threshold:
        level = "Poor"
        color = "🔴"
        score = 30
        explanation = f"Your holdings move together {avg_correlation:.0%} of the time, meaning they tend to rise and fall as a group."
        problem = "When one asset drops, others likely drop too - reducing the safety net diversification should provide."
        solution = "Add assets from different sectors, countries, or asset classes (bonds, commodities, international stocks)."
    elif avg_correlation > low_threshold:
        level = "Moderate"
        color = "🟡"
        score = 65
        explanation = f"Your holdings have {avg_correlation:.0%} correlation - they move together somewhat, but maintain some independence."
        problem = "Decent diversification, but there's room to reduce overlap between holdings."
        solution = "Consider adding low-correlation assets like bonds, gold, or international exposure to improve diversification."
    else:
        level = "Good"
        color = "🟢"
        score = 90
        explanation = f"Your holdings have only {avg_correlation:.0%} correlation - they move relatively independently."
        problem = "Good diversification! Holdings provide meaningful risk reduction."
        solution = "Maintain this diversification structure. Monitor correlations as they can increase during market stress."
    
    return {
        "level": level,
        "color": color,
        "score": score,
        "explanation": explanation,
        "problem": problem,
        "solution": solution
    }

def plot_correlation_heatmap(corr_data, title="Correlation Matrix"):
    """Create correlation heatmap with improved styling"""
    try:
        if isinstance(corr_data, dict):
            df = pd.DataFrame(corr_data)
        else:
            df = corr_data
        
        fig = px.imshow(
            df,
            title=title,
            color_continuous_scale="RdBu_r",
            aspect="auto",
            zmin=-1, zmax=1,
            labels=dict(color="Correlation"),
            text_auto='.2f'
        )
        fig.update_layout(
            height=500,
            font=dict(size=11)
        )
        fig.update_traces(textfont_size=10)
        return fig
    except Exception as e:
        st.error(f"Error creating heatmap: {str(e)}")
        return None

def main():
    inject_custom_css()
    
    add_page_header(
        "Diversification Analysis",
        "Understand how your holdings relate to each other",
        "🔗"
    )
    
    # Sidebar with goal-based navigation
    with st.sidebar:
        add_sidebar_branding()
        
        st.markdown("### 🎯 What do you want to know?")
        st.caption("Choose your question")
        
        goal_questions = {
            "Are my holdings too similar?": "basic",
            "Do correlations change over time?": "rolling",
            "How do crisis periods affect correlations?": "regime",
            "Which assets group together?": "clustering"
        }
        
        for question, analysis in goal_questions.items():
            if st.button(question, use_container_width=True, key=f"goal_{analysis}"):
                st.session_state['selected_correlation_analysis'] = analysis
        
        st.markdown("---")
        
        # Quick preset loader
        st.markdown("### Quick Load")
        preset_options = ["Custom Portfolio"] + list(list_presets().values())
        preset_name = st.selectbox("Portfolio", preset_options, label_visibility="collapsed")
        
        if preset_name != "Custom Portfolio":
            preset_key = [k for k, v in list_presets().items() if v == preset_name][0]
            preset = get_preset(preset_key)
            
            if st.button("📥 Load", key="load_preset_corr", use_container_width=True):
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

        st.markdown("---")
        st.markdown("### 📚 Help & Feedback")
        st.markdown("[Quick Start Guide](https://docs.google.com/document/d/1BX93dy0fOcFdeiiXxT3T7XlDgdC4CRv5Ehp72linzIc/view)")
        st.markdown("[Give Feedback](https://forms.gle/87hpD7gvPVQnsPfc7)")
    symbols, weights = get_portfolio()
    
    # Determine analysis type
    analysis_type = st.session_state.get('selected_correlation_analysis', 'basic')
    
    # Map to display names
    analysis_names = {
        'basic': 'Basic Correlation',
        'rolling': 'Time-Varying Analysis',
        'regime': 'Crisis Behavior',
        'clustering': 'Asset Grouping'
    }
    
    # IMPROVED EMPTY STATE
    if not symbols:
        show_empty_state(
            icon="🔗",
            title="Diversification Analysis Ready",
            message="Load a portfolio to see how your holdings relate to each other"
        )
        
        st.markdown("### 🚀 Test Different Correlation Patterns")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 12px; border: 2px solid rgba(40, 167, 69, 0.3);'>
                <h4 style='color: #28a745; margin-top: 0;'>🌐 Well Diversified</h4>
                <p style='color: #808495; font-size: 0.9rem;'>Tech, Finance, Energy, Health<br>Expected: ~40% correlation</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Try Diversified", key="try_sector", use_container_width=True):
                load_example_portfolio("sector_diverse")
        
        with col2:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 12px; border: 2px solid rgba(220, 53, 69, 0.3);'>
                <h4 style='color: #dc3545; margin-top: 0;'>💻 Poorly Diversified</h4>
                <p style='color: #808495; font-size: 0.9rem;'>All tech stocks<br>Expected: ~75% correlation</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Try Tech Cluster", key="try_tech_cluster", use_container_width=True):
                load_example_portfolio("tech_cluster")
        
        with col3:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 12px; border: 2px solid rgba(23, 162, 184, 0.3);'>
                <h4 style='color: #17a2b8; margin-top: 0;'>🌍 Global Mix</h4>
                <p style='color: #808495; font-size: 0.9rem;'>Stocks, Bonds, Gold<br>Expected: ~30% correlation</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Try Global Mix", key="try_global", use_container_width=True):
                load_example_portfolio("global_mix")
        
        return
    
    if len(symbols) < 2:
        st.error("⚠️ Need at least 2 holdings to analyze correlations")
        return
    
    # Analysis type selector
    st.markdown("## Select Analysis Type")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Basic Analysis", type="primary" if analysis_type == 'basic' else "secondary", use_container_width=True):
            st.session_state['selected_correlation_analysis'] = 'basic'
            st.rerun()
    
    with col2:
        if st.button("📈 Over Time", type="primary" if analysis_type == 'rolling' else "secondary", use_container_width=True):
            st.session_state['selected_correlation_analysis'] = 'rolling'
            st.rerun()
    
    with col3:
        if st.button("💥 During Crises", type="primary" if analysis_type == 'regime' else "secondary", use_container_width=True):
            st.session_state['selected_correlation_analysis'] = 'regime'
            st.rerun()
    
    with col4:
        if st.button("🔍 Find Groups", type="primary" if analysis_type == 'clustering' else "secondary", use_container_width=True):
            st.session_state['selected_correlation_analysis'] = 'clustering'
            st.rerun()
    
    st.markdown("---")
    
    # Run analysis button
    run_analysis = st.button("🚀 Run Analysis", type="primary")
    
    # BASIC CORRELATION
    if analysis_type == 'basic':
        st.markdown("## 📊 Diversification Assessment")
        st.caption("How similar are your holdings?")
        
        if run_analysis:
            with st.spinner("🔗 Analyzing correlations... 5-10 seconds"):
                result = safe_api_call(
                    lambda: api_client.correlation_analysis(symbols, period),
                    error_context="correlation analysis"
                )
                
                if result:
                    st.session_state['corr_basic'] = result
                    st.success("✓ Analysis complete!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Analysis failed. Verify symbols have sufficient overlapping history.")
        
        if 'corr_basic' in st.session_state:
            result = st.session_state['corr_basic']
            correlation_data = result.get('correlation_analysis', {})
            
            if correlation_data:
                avg_corr = correlation_data.get('average_correlation', 0)
                div_score = correlation_data.get('diversification_score', 0)
                
                # NEW: Interpret diversification
                assessment = interpret_diversification(avg_corr, len(symbols))
                
                st.markdown("### Your Diversification Score")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Create gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=assessment['score'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Diversification", 'font': {'size': 16}},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 40], 'color': "#ffc1c1"},
                                {'range': [40, 70], 'color': "#fff3cd"},
                                {'range': [70, 100], 'color': "#c3f7c3"}
                            ],
                            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
                        }
                    ))
                    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown(f"#### {assessment['color']} {assessment['level']} Diversification")
                    st.markdown(f"**Average Correlation:** {avg_corr:.1%}")
                    
                    st.markdown(f"""
                    <div style='background: rgba(102, 126, 234, 0.05); padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
                        <p style='margin: 0 0 0.5rem 0;'><strong>What this means:</strong></p>
                        <p style='margin: 0;'>{assessment['explanation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if assessment['level'] != "Good":
                        st.markdown(f"""
                        <div style='background: rgba(255, 193, 7, 0.05); padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
                            <p style='margin: 0 0 0.5rem 0;'><strong>The problem:</strong></p>
                            <p style='margin: 0 0 0.5rem 0;'>{assessment['problem']}</p>
                            <p style='margin: 0 0 0.5rem 0;'><strong>How to improve:</strong></p>
                            <p style='margin: 0;'>{assessment['solution']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Correlation matrix
                st.markdown("### Correlation Matrix")
                
                corr_matrix = correlation_data.get('correlation_matrix')
                if corr_matrix:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        fig = plot_correlation_heatmap(corr_matrix)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### Reading the Matrix")
                        st.markdown("""
                        **Color Guide:**
                        - 🔴 Red (1.0): Move together
                        - ⚪ White (0.0): Independent
                        - 🔵 Blue (-1.0): Move opposite
                        
                        **Ideal Range:**
                        0.2 - 0.5 for good diversification
                        """)
                
                # Action bridges
                st.markdown("---")
                st.markdown("### 🎯 Next Steps")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if avg_corr > 0.6:
                        st.markdown("#### Improve Diversification")
                        st.caption("Correlations too high")
                        if st.button("🎯 Optimize Portfolio", use_container_width=True):
                            st.switch_page("pages/1_Portfolio_Analysis.py")
                    else:
                        st.markdown("#### Diversification OK")
                        st.caption("Current mix looks good")
                        st.button("✓ No action needed", disabled=True, use_container_width=True)
                
                with col2:
                    st.markdown("#### Test Resilience")
                    st.caption("Check stress performance")
                    if st.button("💥 Run Stress Test", use_container_width=True):
                        st.switch_page("pages/2_Risk_Analytics.py")
                
                with col3:
                    st.markdown("#### Full Report")
                    st.caption("Complete health assessment")
                    if st.button("💡 Generate Insights", use_container_width=True):
                        st.switch_page("pages/6_Portfolio_Insights.py")
    
    # ROLLING CORRELATIONS
    elif analysis_type == 'rolling':
        st.markdown("## 📈 How Correlations Change Over Time")
        st.caption("Relationships between assets aren't constant - see how they evolve")
        
        window_size = st.slider("Analysis Window (days)", 20, 90, 30, step=10,
                                help="Larger = smoother trends, Smaller = more responsive")
        
        if run_analysis:
            with st.spinner("📊 Calculating time-varying correlations... 10-15 seconds"):
                result = safe_api_call(
                    lambda: api_client.rolling_correlations(symbols, window_size, period),
                    error_context="rolling correlation"
                )
                
                if result:
                    st.session_state['corr_rolling'] = result
                    st.success("✓ Analysis complete!")
                    time.sleep(0.5)
                    st.rerun()
        
        if 'corr_rolling' in st.session_state:
            result = st.session_state['corr_rolling']
            rolling_data = result.get('rolling_correlations', {})
            
            if rolling_data:
                stability_metrics = rolling_data.get('stability_metrics', {})
                stability_score = stability_metrics.get('stability_score', 0)
                corr_volatility = stability_metrics.get('correlation_volatility', 0)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Stability Score", f"{stability_score:.0%}")
                
                with col2:
                    num_windows = rolling_data.get('total_windows', 0)
                    st.metric("Analysis Windows", num_windows)
                
                with col3:
                    st.metric("Correlation Volatility", f"{corr_volatility:.2f}")
                
                # NEW: Interpretation
                with st.expander("🤖 What does this mean?", expanded=True):
                    if stability_score > 0.8:
                        st.success(f"""
                        **Stable relationships ({stability_score:.0%} stability)**
                        
                        Your holdings maintain consistent correlation patterns over time. This is good - 
                        it means diversification benefits are reliable and predictable.
                        
                        **What this means for you:** Continue monitoring, but your current diversification 
                        structure appears sound.
                        """)
                    elif stability_score > 0.6:
                        st.info(f"""
                        **Moderate stability ({stability_score:.0%} stability)**
                        
                        Correlations vary somewhat over time, which is normal. Some periods show stronger 
                        relationships between assets than others.
                        
                        **What this means for you:** Diversification benefits exist but may weaken during 
                        volatile periods. Monitor correlation trends regularly.
                        """)
                    else:
                        st.warning(f"""
                        **Unstable relationships ({stability_score:.0%} stability)**
                        
                        Correlations change significantly over time. Assets that were independent in some 
                        periods move together in others, making diversification benefits unpredictable.
                        
                        **What this means for you:** Consider adding more structurally independent assets 
                        (different asset classes, geographies) for reliable diversification.
                        """)
    
    # REGIME CORRELATIONS
    elif analysis_type == 'regime':
        st.markdown("## 💥 Crisis Correlation Behavior")
        st.caption("Do your diversification benefits disappear when you need them most?")
        
        if run_analysis:
            with st.spinner("🔄 Analyzing crisis vs. normal correlations... 15-20 seconds"):
                result = safe_api_call(
                    lambda: api_client.regime_correlations(symbols, "volatility", period),
                    error_context="regime correlation"
                )
                
                if result:
                    st.session_state['corr_regime'] = result
                    st.success("✓ Analysis complete!")
                    time.sleep(0.5)
                    st.rerun()
        
        if 'corr_regime' in st.session_state:
            result = st.session_state['corr_regime']
            regime_data = result.get('regime_correlations', {})
            
            if regime_data:
                regime_sensitivity = regime_data.get('regime_sensitivity', {})
                market_regimes = regime_data.get('market_regime_correlations', {})
                
                crisis_multiplier = regime_sensitivity.get('crisis_correlation_multiplier', 0)
                bull_corr = market_regimes.get('bull', {}).get('avg_correlation', 0)
                bear_corr = market_regimes.get('bear', {}).get('avg_correlation', 0)
                
                st.markdown("### Crisis Impact Assessment")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Crisis Multiplier",
                        f"{crisis_multiplier:.2f}x",
                        help="How much correlations increase during crises"
                    )
                
                with col2:
                    st.metric("Normal Markets", f"{bull_corr:.1%}")
                
                with col3:
                    st.metric("Crisis Periods", f"{bear_corr:.1%}")
                
                # NEW: Plain-language crisis assessment
                increase_pct = ((bear_corr - bull_corr) / bull_corr * 100) if bull_corr > 0 else 0
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(220, 53, 69, 0.1) 0%, rgba(255, 193, 7, 0.1) 100%); 
                            padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0;'>
                    <h4 style='margin-top: 0;'>Crisis Diversification Analysis</h4>
                """, unsafe_allow_html=True)
                
                if crisis_multiplier > 2.0:
                    st.markdown(f"""
                    <p><strong>🔴 SEVERE CRISIS CORRELATION ({crisis_multiplier:.1f}x increase)</strong></p>
                    <p>Your diversification largely <strong>breaks down during market stress</strong>. 
                    When markets crash, your holdings correlate {increase_pct:.0f}% more than normal, 
                    meaning they tend to drop together.</p>
                    <p><strong>Real-world impact:</strong> During a 2008-style crisis, you'd likely see 
                    most/all holdings decline simultaneously, defeating the purpose of diversification.</p>
                    <p><strong>Action needed:</strong> Add truly defensive assets - bonds, gold, or strategies 
                    that aren't tied to equity market direction.</p>
                    """, unsafe_allow_html=True)
                elif crisis_multiplier > 1.5:
                    st.markdown(f"""
                    <p><strong>🟡 MODERATE CRISIS CORRELATION ({crisis_multiplier:.1f}x increase)</strong></p>
                    <p>Your holdings show <strong>significantly higher correlation during stress</strong> 
                    ({increase_pct:.0f}% increase). Some diversification benefit remains, but it weakens when needed.</p>
                    <p><strong>Real-world impact:</strong> Most holdings would decline together during major 
                    corrections, though perhaps at different rates.</p>
                    <p><strong>Consider:</strong> Adding low-correlation assets (bonds, international, alternatives) 
                    to maintain diversification during downturns.</p>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <p><strong>🟢 STABLE CRISIS CORRELATION ({crisis_multiplier:.1f}x increase)</strong></p>
                    <p>Excellent! Your diversification benefits <strong>remain intact during crises</strong>. 
                    Correlations increase only {increase_pct:.0f}%, meaning your holdings maintain independence 
                    when you need it most.</p>
                    <p><strong>Real-world impact:</strong> During market downturns, some holdings should decline 
                    less (or even rise), providing the cushion diversification promises.</p>
                    <p><strong>Maintain this structure</strong> - your portfolio shows institutional-grade 
                    crisis resilience.</p>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Comparison chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Normal Markets',
                    x=['Average Correlation'],
                    y=[bull_corr],
                    marker_color='#28a745'
                ))
                
                fig.add_trace(go.Bar(
                    name='Crisis Periods',
                    x=['Average Correlation'],
                    y=[bear_corr],
                    marker_color='#dc3545'
                ))
                
                fig.update_layout(
                    title="Normal vs Crisis Correlations",
                    yaxis_title="Correlation",
                    yaxis_tickformat='.0%',
                    height=350,
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # CLUSTERING
    elif analysis_type == 'clustering':
        st.markdown("## 🔍 Asset Grouping Analysis")
        st.caption("Which holdings behave similarly?")
        
        if run_analysis:
            with st.spinner("🔍 Identifying asset clusters... 10-15 seconds"):
                result = safe_api_call(
                    lambda: api_client.correlation_clustering(symbols, period),
                    error_context="correlation clustering"
                )
                
                if result:
                    st.session_state['corr_clustering'] = result
                    st.success("✓ Clustering complete!")
                    time.sleep(0.5)
                    st.rerun()
        
        if 'corr_clustering' in st.session_state:
            result = st.session_state['corr_clustering']
            clustering_data = result.get('correlation_clustering', {})
            
            if clustering_data:
                optimal_clusters = clustering_data.get('optimal_clusters', 0)
                cluster_analysis = clustering_data.get('cluster_analysis', {})
                
                st.markdown(f"### Found {optimal_clusters} Distinct Groups")
                
                if cluster_analysis:
                    for i, (cluster_name, cluster_data) in enumerate(cluster_analysis.items(), 1):
                        cluster_symbols = cluster_data.get('symbols', [])
                        avg_corr = cluster_data.get('avg_internal_correlation', 0)
                        
                        if cluster_symbols:
                            st.markdown(f"""
                            <div style='background: rgba(102, 126, 234, 0.05); padding: 1rem; 
                                        border-radius: 8px; border-left: 4px solid #667eea; margin: 1rem 0;'>
                                <h4 style='margin-top: 0;'>Cluster {i}: {', '.join(cluster_symbols)}</h4>
                                <p><strong>Internal correlation:</strong> {avg_corr:.1%}</p>
                                <p style='margin-bottom: 0;'><strong>Meaning:</strong> These assets move together 
                                {avg_corr:.0%} of the time - they represent similar market exposures.</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # NEW: Actionable interpretation
                    if optimal_clusters == 1:
                        st.warning("""
                        ⚠️ **All holdings in one cluster** - This means your entire portfolio moves as a single unit. 
                        Diversification is minimal. Consider adding assets from different sectors or asset classes.
                        """)
                    elif optimal_clusters == len(symbols):
                        st.success("""
                        ✅ **Each holding is independent** - Excellent diversification! Your assets represent 
                        distinct market exposures, providing maximum diversification benefit.
                        """)
                    else:
                        st.info(f"""
                        ℹ️ **{optimal_clusters} groups identified** - Your portfolio has some clustering. 
                        Assets within each group provide limited diversification to each other. For better 
                        diversification, ensure you have holdings across all clusters.
                        """)
    
    # Footer
    st.markdown("---")
    add_footer_tip("💡 Correlations can change quickly during market stress. Monitor them regularly, especially before major portfolio changes.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("An unexpected error occurred in Correlation Analysis")
        request_logger.logger.exception("Unhandled exception in Correlation Analysis")
        with st.expander("🔍 Error Details"):
            st.code(str(e))