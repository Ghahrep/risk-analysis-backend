"""
Advanced Analytics Page
Risk attribution, performance attribution, factor analysis, and advanced metrics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
from utils.portfolio_manager import get_portfolio, set_portfolio, normalize_weights, initialize_portfolio

# Initialize portfolio
initialize_portfolio()

sys.path.append(str(Path(__file__).parent.parent))
from utils.api_client import get_risk_api_client

st.set_page_config(page_title="Advanced Analytics", page_icon="📊", layout="wide")

api_client = get_risk_api_client()

def main():
    st.title("📊 Advanced Analytics")
    st.markdown("Institutional-grade risk attribution, performance analysis, and factor decomposition")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Portfolio Configuration")
        
        # Get current portfolio
        current_symbols, current_weights = get_portfolio()
        
        # Symbol input
        symbols_input = st.text_area(
            "Stock Symbols (one per line)",
            value='\n'.join(current_symbols),
            height=120
        )
        symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]
        
        # Weights
        st.subheader("Portfolio Weights")
        weights = []
        
        for i, symbol in enumerate(symbols):
            default_weight = current_weights[i] if i < len(current_weights) else 1.0/len(symbols)
            weight = st.slider(
                f"{symbol}",
                0.0, 1.0, default_weight, 0.01,
                key=f"weight_{symbol}_{st.session_state.get('page_key', 'default')}"
            )
            weights.append(weight)
        
        # Normalize weights
        weights = normalize_weights(weights)
        
        st.write(f"**Total Weight:** {sum(weights):.2%}")
        
        # Save button
        if st.button("💾 Save Portfolio", width='stretch'):
            set_portfolio(symbols, weights)
            st.success("Portfolio saved!")
        
        # Period selection
        period = st.selectbox("Analysis Period", ["1year", "2years", "3months"], index=0)
        
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # Main content
    if not symbols:
        st.info("Enter stock symbols in the sidebar to begin advanced analytics")
        return
    
    if analysis_type == "Risk Attribution":
        st.header("Portfolio Risk Attribution")
        st.markdown("Decompose total portfolio risk into systematic and idiosyncratic components")
        
        factor_model = st.selectbox(
            "Factor Model",
            ["fama_french_3", "fama_french_5"],
            format_func=lambda x: {
                "fama_french_3": "Fama-French 3-Factor",
                "fama_french_5": "Fama-French 5-Factor"
            }[x]
        )
        
        if run_analysis:
            with st.spinner("Calculating risk attribution..."):
                result = api_client.risk_attribution(symbols, weights, period)
                
                if result:
                    st.session_state['risk_attribution'] = result
        
        if 'risk_attribution' in st.session_state:
            result = st.session_state['risk_attribution']
            
            # Key metrics
            st.subheader("Risk Decomposition")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_risk = result.get('risk_attribution', {}).get('total_risk_pct', 0)
                st.metric("Total Portfolio Risk", f"{total_risk:.2%}")
            
            with col2:
                systematic_risk = result.get('risk_attribution', {}).get('systematic_risk_pct', 0)
                st.metric("Systematic Risk", f"{systematic_risk:.2%}")
            
            with col3:
                idiosyncratic_risk = result.get('risk_attribution', {}).get('idiosyncratic_risk_pct', 0)
                st.metric("Idiosyncratic Risk", f"{idiosyncratic_risk:.2%}")
            
            # Risk breakdown chart
            if systematic_risk > 0 or idiosyncratic_risk > 0:
                fig = go.Figure(data=[go.Pie(
                    labels=['Systematic Risk', 'Idiosyncratic Risk'],
                    values=[systematic_risk, idiosyncratic_risk],
                    hole=0.4,
                    marker_colors=['#ff6b6b', '#4ecdc4']
                )])
                fig.update_layout(title="Risk Components")
                st.plotly_chart(fig, use_container_width=True)
            
            # Factor contributions
            st.subheader("Factor Contributions to Risk")
            
            factor_contribs = result.get('risk_attribution', {}).get('factor_contributions', {})
            
            if factor_contribs:
                contrib_df = pd.DataFrame([
                    {'Factor': k.upper(), 'Contribution %': v}
                    for k, v in factor_contribs.items()
                ])
                
                fig = px.bar(
                    contrib_df,
                    x='Factor',
                    y='Contribution %',
                    title="Risk Contribution by Factor",
                    color='Contribution %',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Concentration metrics
            st.subheader("Concentration Metrics")
            
            concentration = result.get('risk_attribution', {}).get('concentration_metrics', {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Herfindahl Index",
                    f"{concentration.get('herfindahl_index', 0):.3f}",
                    help="Portfolio concentration (lower is more diversified)"
                )
            
            with col2:
                st.metric(
                    "Top 5 Concentration",
                    f"{concentration.get('top_5_concentration_pct', 0):.1f}%",
                    help="Weight of top 5 holdings"
                )
            
            with col3:
                st.metric(
                    "Largest Position",
                    f"{concentration.get('largest_position_pct', 0):.1f}%",
                    help="Largest individual holding"
                )
    
    elif analysis_type == "Performance Attribution":
        st.header("Performance Attribution Analysis")
        st.markdown("Analyze sources of portfolio returns vs benchmark")
        
        benchmark = st.selectbox(
            "Benchmark",
            ["SPY", "QQQ", "DIA", "IWM"],
            format_func=lambda x: {
                "SPY": "S&P 500 (SPY)",
                "QQQ": "NASDAQ 100 (QQQ)",
                "DIA": "Dow Jones (DIA)",
                "IWM": "Russell 2000 (IWM)"
            }[x]
        )
        
        if run_analysis:
            with st.spinner("Calculating performance attribution..."):
                result = api_client.performance_attribution(symbols, weights, benchmark, period)
                
                if result:
                    st.session_state['perf_attribution'] = result
        
        if 'perf_attribution' in st.session_state:
            result = st.session_state['perf_attribution']
            
            perf_attr = result.get('performance_attribution', {})
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Return",
                    f"{perf_attr.get('total_return_pct', 0):.2%}"
                )
            
            with col2:
                alpha = perf_attr.get('alpha_pct', 0)
                st.metric(
                    "Alpha",
                    f"{alpha:.2%}",
                    delta=f"{'Significant' if abs(alpha) > 1.0 else 'Not significant'}"
                )
            
            with col3:
                st.metric(
                    "Tracking Error",
                    f"{perf_attr.get('risk_adjusted_metrics', {}).get('tracking_error', 0):.2%}"
                )
            
            with col4:
                st.metric(
                    "Information Ratio",
                    f"{perf_attr.get('risk_adjusted_metrics', {}).get('information_ratio', 0):.3f}"
                )
            
            # Factor contributions
            st.subheader("Factor Contributions to Returns")
            
            factor_contribs = perf_attr.get('factor_contributions', {})
            
            if factor_contribs:
                contrib_df = pd.DataFrame([
                    {'Factor': k.upper(), 'Return Contribution %': v}
                    for k, v in factor_contribs.items()
                ])
                
                fig = px.bar(
                    contrib_df,
                    x='Factor',
                    y='Return Contribution %',
                    title="Return Attribution by Factor",
                    color='Return Contribution %',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Attribution effects
            st.subheader("Attribution Effects")
            
            effects = perf_attr.get('attribution_effects', {})
            
            if effects:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Selection Effect",
                        f"{effects.get('selection_effect', 0):.2%}",
                        help="Return from security selection"
                    )
                
                with col2:
                    st.metric(
                        "Allocation Effect",
                        f"{effects.get('allocation_effect', 0):.2%}",
                        help="Return from sector/asset allocation"
                    )
                
                with col3:
                    st.metric(
                        "Interaction Effect",
                        f"{effects.get('interaction_effect', 0):.2%}",
                        help="Combined selection and allocation effect"
                    )
    
    elif analysis_type == "Advanced Metrics":
        st.header("Advanced Portfolio Metrics")
        st.markdown("Comprehensive analytics including diversification, risk-adjusted performance, and tail risk")
        
        if run_analysis:
            with st.spinner("Calculating advanced metrics..."):
                result = api_client.advanced_analytics(symbols, weights, period)
                
                if result:
                    st.session_state['advanced_metrics'] = result
        
        if 'advanced_metrics' in st.session_state:
            result = st.session_state['advanced_metrics']
            
            analytics = result.get('advanced_analytics', {})
            
            # Diversification metrics
            st.subheader("Diversification Metrics")
            
            div_metrics = analytics.get('diversification_metrics', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Diversification Ratio",
                    f"{div_metrics.get('diversification_ratio', 0):.3f}",
                    help="Ratio of weighted avg volatility to portfolio volatility"
                )
            
            with col2:
                st.metric(
                    "Effective # Assets",
                    f"{div_metrics.get('effective_num_assets', 0):.2f}",
                    help="Number of uncorrelated assets equivalent"
                )
            
            with col3:
                st.metric(
                    "Avg Correlation",
                    f"{div_metrics.get('avg_correlation', 0):.3f}",
                    help="Average pairwise correlation"
                )
            
            with col4:
                st.metric(
                    "Correlation Clusters",
                    div_metrics.get('correlation_clusters', 0),
                    help="Number of distinct correlation groups"
                )
            
            # Risk-adjusted performance
            st.subheader("Risk-Adjusted Performance")
            
            risk_adj = analytics.get('risk_adjusted_performance', {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Calmar Ratio",
                    f"{risk_adj.get('calmar_ratio', 0):.3f}",
                    help="Return / Max Drawdown"
                )
            
            with col2:
                st.metric(
                    "Sortino Ratio",
                    f"{risk_adj.get('sortino_ratio', 0):.3f}",
                    help="Return / Downside Deviation"
                )
            
            with col3:
                st.metric(
                    "Omega Ratio",
                    f"{risk_adj.get('omega_ratio', 0):.3f}",
                    help="Probability-weighted gains/losses"
                )
            
            # Tail risk measures
            st.subheader("Tail Risk Measures")
            
            tail_risk = analytics.get('tail_risk_measures', {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "VaR (95%)",
                    f"{tail_risk.get('var_95_pct', 0):.2%}"
                )
            
            with col2:
                st.metric(
                    "CVaR (95%)",
                    f"{tail_risk.get('cvar_95_pct', 0):.2%}"
                )
            
            with col3:
                st.metric(
                    "Max Drawdown",
                    f"{tail_risk.get('max_drawdown_pct', 0):.2%}"
                )
    
    elif analysis_type == "Factor Analysis":
        st.header("Factor Analysis")
        st.markdown("Fama-French factor exposure analysis")
        
        st.info("Factor analysis requires individual stock analysis. Select a single symbol for detailed factor exposure.")
        
        if len(symbols) > 0:
            selected_symbol = st.selectbox("Select Symbol for Factor Analysis", symbols)
            
            if run_analysis and selected_symbol:
                with st.spinner(f"Analyzing factor exposure for {selected_symbol}..."):
                    # Note: This would need a dedicated factor analysis endpoint
                    st.warning("Factor analysis endpoint integration coming soon")
                    
                    # Placeholder visualization
                    st.subheader(f"Factor Exposure: {selected_symbol}")
                    
                    # Mock data for demonstration
                    factors = ['Market', 'Size', 'Value', 'Profitability', 'Investment']
                    exposures = [0.85, 0.15, -0.10, 0.25, -0.05]
                    
                    fig = go.Figure(go.Bar(
                        x=factors,
                        y=exposures,
                        marker_color=['red' if e < 0 else 'green' for e in exposures]
                    ))
                    fig.update_layout(
                        title=f"Fama-French Factor Exposures: {selected_symbol}",
                        yaxis_title="Factor Loading"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please enter at least one symbol in the sidebar")
    
    # Footer
    st.markdown("---")
    st.caption("💡 Advanced analytics provide institutional-grade insights into portfolio construction and performance")

if __name__ == "__main__":
    main()