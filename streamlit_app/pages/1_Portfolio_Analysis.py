"""
Portfolio Analysis Page
Optimization, risk metrics, and portfolio configuration
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.portfolio_manager import get_portfolio, set_portfolio, normalize_weights, initialize_portfolio
from utils.portfolio_presets import list_presets, get_preset
from utils.api_client import get_risk_api_client

st.set_page_config(page_title="Portfolio Analysis", page_icon="📈", layout="wide")

# Initialize portfolio
initialize_portfolio()

# Initialize API client
api_client = get_risk_api_client()

def main():
    st.title("📈 Portfolio Analysis & Optimization")
    
    # Sidebar for portfolio configuration
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
            # Find preset key
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
                key=f"weight_{symbol}_portfolio"
            )
            weights.append(weight)
        
        # Normalize weights
        weights = normalize_weights(weights)
        
        st.write(f"**Total Weight:** {sum(weights):.2%}")
        if abs(sum(weights) - 1.0) > 0.01:
            st.caption("Weights normalized to 100%")
        
        # Save button
        if st.button("💾 Save Portfolio", width='stretch'):
            set_portfolio(symbols, weights)
            st.success("Portfolio saved!")
        
        # Period selection
        period = st.selectbox(
            "Analysis Period",
            ["1month", "3months", "6months", "1year", "2years"],
            index=3
        )
        
        # Run analysis button
        run_analysis = st.button("🚀 Run Analysis", type="primary", width='stretch')
    
    # Main content area
    if not symbols:
        st.info("👈 Select a preset or enter stock symbols in the sidebar to begin analysis")
        return
    
    # Portfolio overview
    st.header("Portfolio Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Holdings", len(symbols))
    with col2:
        largest_holding = max(zip(symbols, weights), key=lambda x: x[1]) if weights else ("N/A", 0)
        st.metric("Largest Holding", f"{largest_holding[0]}")
    with col3:
        st.metric("Concentration", f"{max(weights) if weights else 0:.1%}")
    with col4:
        # Simple diversification score based on weights
        herfindahl = sum(w**2 for w in weights) if weights else 0
        diversification = (1 - herfindahl) * 100
        st.metric("Diversification", f"{diversification:.0f}/100")
    
    # Portfolio allocation visualization
    if symbols and weights:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.pie(
                values=weights,
                names=symbols,
                title="Current Portfolio Allocation",
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("Holdings Detail")
            holdings_df = pd.DataFrame({
                'Symbol': symbols,
                'Weight': [f"{w:.2%}" for w in weights],
                'Weight Value': weights
            }).sort_values('Weight Value', ascending=False)
            st.dataframe(
                holdings_df[['Symbol', 'Weight']],
                hide_index=True,
                width='stretch'
            )
    
    st.markdown("---")
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["Portfolio Optimization", "Risk Metrics", "Comparison"])
    
    with tab1:
        st.header("Portfolio Optimization")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            optimization_method = st.selectbox(
                "Optimization Method",
                ["max_sharpe", "min_variance", "equal_weight"],
                format_func=lambda x: {
                    "max_sharpe": "Maximum Sharpe Ratio",
                    "min_variance": "Minimum Variance",
                    "equal_weight": "Equal Weight"
                }[x]
            )
            
            if st.button("Optimize Portfolio", type="primary"):
                with st.spinner("Running optimization..."):
                    result = api_client.optimize_portfolio(symbols, optimization_method, period)
                    
                    if result:
                        st.session_state['optimization_result'] = result
        
        with col2:
            if 'optimization_result' in st.session_state:
                result = st.session_state['optimization_result']
                
                # Navigate nested structure
                opt_results = result.get('optimization_results', {})
                
                st.success("✅ Optimization Complete")
                
                # Performance metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    expected_return = opt_results.get('expected_return', 0)
                    st.metric("Expected Return", f"{expected_return:.2%}")
                with col_b:
                    expected_vol = opt_results.get('expected_volatility', 0)
                    st.metric("Expected Volatility", f"{expected_vol:.2%}")
                with col_c:
                    sharpe = opt_results.get('sharpe_ratio', 0)
                    st.metric("Sharpe Ratio", f"{sharpe:.3f}")
        
        # Weight comparison
        if 'optimization_result' in st.session_state:
            result = st.session_state['optimization_result']
            opt_results = result.get('optimization_results', {})
            
            if 'optimal_weights' in opt_results:
                optimal_weights_dict = opt_results['optimal_weights']
                optimized_weights = [optimal_weights_dict.get(s, 0) for s in symbols]
                
                comparison_df = pd.DataFrame({
                    'Symbol': symbols,
                    'Current': [w * 100 for w in weights],
                    'Optimized': [w * 100 for w in optimized_weights]
                })
                
                fig = px.bar(
                    comparison_df,
                    x='Symbol',
                    y=['Current', 'Optimized'],
                    title="Current vs Optimized Weights (%)",
                    barmode='group'
                )
                st.plotly_chart(fig, width='stretch')
    
    with tab2:
        st.header("Portfolio Risk Metrics")
        
        if st.button("Calculate Risk Metrics", type="primary"):
            with st.spinner("Analyzing risk..."):
                result = api_client.analyze_risk(symbols, weights, period)
                
                if result:
                    st.session_state['risk_result'] = result
        
        if 'risk_result' in st.session_state:
            result = st.session_state['risk_result']
            
            # Navigate nested structure
            metrics = result.get('metrics', {})
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                sharpe_ratio = metrics.get('sharpe_ratio', 0)
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
            with col2:
                annual_vol = metrics.get('annualized_volatility', 0)
                st.metric("Annual Volatility", f"{annual_vol:.2%}")
            with col3:
                max_dd = metrics.get('max_drawdown_pct', 0)
                st.metric("Max Drawdown", f"{max_dd:.2f}%")
            with col4:
                var_95 = metrics.get('portfolio_var_95', 0)
                st.metric("VaR (95%)", f"{var_95:.2%}")
            
            # Additional metrics
            with st.expander("View Additional Metrics"):
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.write("**Return Metrics**")
                    sortino = metrics.get('sortino_ratio', 0)
                    st.write(f"Sortino Ratio: {sortino:.3f}")
                    skewness = metrics.get('skewness', 0)
                    st.write(f"Skewness: {skewness:.3f}")
                
                with metrics_col2:
                    st.write("**Risk Metrics**")
                    cvar_95 = metrics.get('portfolio_cvar_95', 0)
                    st.write(f"CVaR (95%): {cvar_95:.2%}")
                    kurtosis = metrics.get('kurtosis', 0)
                    st.write(f"Kurtosis: {kurtosis:.3f}")
    
    with tab3:
        st.header("Benchmark Comparison")
        
        benchmark = st.selectbox(
            "Select Benchmark",
            ["SPY", "QQQ", "DIA", "IWM"],
            format_func=lambda x: {
                "SPY": "S&P 500 (SPY)",
                "QQQ": "NASDAQ 100 (QQQ)",
                "DIA": "Dow Jones (DIA)",
                "IWM": "Russell 2000 (IWM)"
            }[x]
        )
        
        st.info("Benchmark comparison coming in next update - will show your portfolio vs benchmark performance")
    
    # Footer
    st.markdown("---")
    st.caption("💡 Tip: Adjust portfolio weights using the sliders in the sidebar to see how changes affect metrics")

if __name__ == "__main__":
    main()