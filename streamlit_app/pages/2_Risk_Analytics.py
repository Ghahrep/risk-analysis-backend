"""
Risk Analytics Page
VaR, stress testing, volatility forecasting, and comprehensive risk metrics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.portfolio_manager import get_portfolio, set_portfolio, normalize_weights, initialize_portfolio
from utils.portfolio_presets import list_presets, get_preset
from utils.api_client import get_risk_api_client

st.set_page_config(page_title="Risk Analytics", page_icon="⚠️", layout="wide")

# Initialize portfolio
initialize_portfolio()

api_client = get_risk_api_client()

def main():
    st.title("⚠️ Risk Analytics")
    st.markdown("Comprehensive risk analysis including VaR, stress testing, and volatility forecasting")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Risk Analysis Configuration")
        
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
                key=f"weight_{symbol}_risk"
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
        
        st.markdown("---")
        
        # Analysis type
        st.subheader("Analysis Type")
        analysis_type = st.radio(
            "Select Analysis",
            ["Comprehensive Risk", "VaR Analysis", "Stress Testing", "Volatility Forecasting"]
        )
        
        # Period selection
        period = st.selectbox(
            "Analysis Period",
            ["1month", "3months", "6months", "1year", "2years"],
            index=3
        )
        
        run_analysis = st.button("🚀 Run Risk Analysis", type="primary", width='stretch')
    
    # Main content
    if not symbols:
        st.info("👈 Select a preset or enter stock symbols in the sidebar to begin risk analysis")
        return
    
    # Portfolio overview
    st.header("Portfolio Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Holdings", len(symbols))
    with col2:
        st.metric("Total Weight", f"{sum(weights):.2%}")
    with col3:
        st.metric("Max Position", f"{max(weights):.2%}")
    with col4:
        concentration = sum(w**2 for w in weights)
        st.metric("Concentration", f"{concentration:.3f}")
    
    st.markdown("---")
    
    # Analysis sections
    if analysis_type == "Comprehensive Risk":
        st.header("Comprehensive Risk Analysis")
        
        if run_analysis:
            with st.spinner("Calculating comprehensive risk metrics..."):
                result = api_client.analyze_risk(symbols, weights, period)
                
                if result:
                    st.session_state['comprehensive_risk'] = result
        
        if 'comprehensive_risk' in st.session_state:
            result = st.session_state['comprehensive_risk']
            
            # Navigate nested structure
            metrics = result.get('metrics', {})
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                sharpe_ratio = metrics.get('sharpe_ratio', 0)
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}", help="Risk-adjusted return metric")
            with col2:
                annual_vol = metrics.get('annualized_volatility', 0)
                st.metric("Annual Volatility", f"{annual_vol:.2%}", help="Annualized standard deviation of returns")
            with col3:
                max_dd = metrics.get('max_drawdown_pct', 0)
                st.metric("Max Drawdown", f"{max_dd:.2f}%", help="Largest peak-to-trough decline")
            with col4:
                var_95 = metrics.get('portfolio_var_95', 0)
                st.metric("VaR (95%)", f"{var_95:.2%}", help="Value at Risk at 95% confidence")
            
            # Additional metrics in expandable section
            with st.expander("Additional Risk Metrics"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Return Metrics")
                    sortino = metrics.get('sortino_ratio', 0)
                    skewness = metrics.get('skewness', 0)
                    st.write(f"**Sortino Ratio:** {sortino:.3f}")
                    st.write(f"**Skewness:** {skewness:.3f}")
                
                with col2:
                    st.subheader("Risk Metrics")
                    cvar_95 = metrics.get('portfolio_cvar_95', 0)
                    kurtosis = metrics.get('kurtosis', 0)
                    st.write(f"**CVaR (95%):** {cvar_95:.2%}")
                    st.write(f"**Kurtosis:** {kurtosis:.3f}")
            
            # Risk interpretation
            st.subheader("Risk Assessment")
            
            if sharpe_ratio > 1.0:
                st.success("Strong risk-adjusted returns (Sharpe > 1.0)")
            elif sharpe_ratio > 0.5:
                st.info("Moderate risk-adjusted returns (Sharpe 0.5-1.0)")
            else:
                st.warning("Low risk-adjusted returns (Sharpe < 0.5)")
            
            if annual_vol > 0.25:
                st.warning("High volatility portfolio (>25% annual)")
            elif annual_vol > 0.15:
                st.info("Moderate volatility (15-25% annual)")
            else:
                st.success("Low volatility portfolio (<15% annual)")
    
    elif analysis_type == "VaR Analysis":
        st.header("Value at Risk (VaR) Analysis")
        
        confidence_level = st.selectbox(
            "Confidence Level",
            [0.90, 0.95, 0.99],
            index=1,
            format_func=lambda x: f"{x:.0%}"
        )
        
        if run_analysis:
            with st.spinner("Calculating VaR metrics..."):
                result = api_client.calculate_var(symbols, weights, confidence_level)
                
                if result:
                    st.session_state['var_result'] = result
        
        if 'var_result' in st.session_state:
            result = st.session_state['var_result']
            
            st.subheader("VaR Metrics")
            
            # Navigate nested structure
            var_estimates = result.get('var_cvar_estimates', {})
            confidence_key = f"{int(confidence_level*100)}%"
            var_data = var_estimates.get(confidence_key, {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                var_value = var_data.get('var', 0)
                st.metric(
                    f"VaR ({confidence_level:.0%})",
                    f"{var_value:.2%}" if var_value else "0.00%",
                    help=f"Expected loss not exceeded with {confidence_level:.0%} confidence"
                )
            with col2:
                cvar_value = var_data.get('cvar', 0)
                st.metric(
                    f"CVaR ({confidence_level:.0%})",
                    f"{cvar_value:.2%}" if cvar_value else "0.00%",
                    help="Expected loss given VaR threshold is exceeded"
                )
            with col3:
                portfolio_value = st.number_input(
                    "Portfolio Value ($)",
                    min_value=1000,
                    value=100000,
                    step=1000
                )
            
            # Dollar VaR
            st.subheader("Dollar Risk Exposure")
            col1, col2 = st.columns(2)
            
            with col1:
                dollar_var = abs(var_value * portfolio_value) if var_value else 0
                st.metric(
                    f"VaR (${portfolio_value:,.0f})",
                    f"${dollar_var:,.2f}",
                    help="Maximum expected loss in dollars"
                )
            with col2:
                dollar_cvar = abs(cvar_value * portfolio_value) if cvar_value else 0
                st.metric(
                    f"CVaR (${portfolio_value:,.0f})",
                    f"${dollar_cvar:,.2f}",
                    help="Expected loss in worst scenarios"
                )
            
            # VaR interpretation
            st.info(f"""
            **Interpretation**: With {confidence_level:.0%} confidence, your portfolio will not lose more than 
            {abs(var_value):.2%} (${dollar_var:,.2f}) in a single period. However, when losses 
            exceed this threshold, the average loss is {abs(cvar_value):.2%} (${dollar_cvar:,.2f}).
            """)
    
    elif analysis_type == "Stress Testing":
        st.header("Stress Testing Analysis")
        st.markdown("Analyze portfolio performance under extreme market scenarios")
        
        if run_analysis:
            with st.spinner("Running stress tests..."):
                result = api_client.stress_test(symbols, weights)
                
                if result:
                    st.session_state['stress_result'] = result
        
        if 'stress_result' in st.session_state:
            result = st.session_state['stress_result']
            
            # Navigate nested structure
            stress_scenarios = result.get('stress_scenarios', {})
            worst_case = result.get('worst_case_scenario', 'Unknown')
            resilience_score = result.get('resilience_score', 0)
            monte_carlo = result.get('monte_carlo_results', {})
            
            # Display resilience score
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Resilience Score", f"{resilience_score:.1f}/100")
            with col2:
                st.metric("Worst Case Scenario", worst_case.replace('_', ' ').title())
            with col3:
                worst_loss = stress_scenarios.get(worst_case, {}).get('total_loss_pct', 0)
                st.metric("Worst Case Loss", f"{worst_loss:.2f}%")
            
            st.subheader("Stress Test Results")
            
            # Create stress test visualization
            if stress_scenarios:
                scenario_data = []
                for scenario_name, metrics in stress_scenarios.items():
                    scenario_data.append({
                        'Scenario': scenario_name.replace('_', ' ').title(),
                        'Loss %': metrics.get('total_loss_pct', 0),
                        'Max Daily Loss %': metrics.get('max_daily_loss_pct', 0),
                        'Max Drawdown %': metrics.get('max_drawdown_pct', 0)
                    })
                
                df = pd.DataFrame(scenario_data)
                
                fig = px.bar(
                    df,
                    x='Scenario',
                    y='Loss %',
                    title="Portfolio Loss Under Stress Scenarios",
                    color='Loss %',
                    color_continuous_scale='Reds_r',
                    hover_data=['Max Daily Loss %', 'Max Drawdown %']
                )
                st.plotly_chart(fig, width='stretch')
                
                # Detailed scenario table
                with st.expander("View Detailed Scenario Metrics"):
                    detailed_data = []
                    for scenario_name, metrics in stress_scenarios.items():
                        detailed_data.append({
                            'Scenario': scenario_name.replace('_', ' ').title(),
                            'Total Loss': f"{metrics.get('total_loss_pct', 0):.2f}%",
                            'Max Daily Loss': f"{metrics.get('max_daily_loss_pct', 0):.2f}%",
                            'Max Drawdown': f"{metrics.get('max_drawdown_pct', 0):.2f}%",
                            'Volatility': f"{metrics.get('annualized_volatility_pct', 0):.2f}%",
                            'Recovery Days': f"{metrics.get('recovery_estimate_days', 0):.0f}"
                        })
                    
                    detailed_df = pd.DataFrame(detailed_data)
                    st.dataframe(detailed_df, width='stretch', hide_index=True)
                
                # Worst scenario warning
                if worst_loss < -15:
                    st.error(f"""
                    **Worst Case Scenario**: {worst_case.replace('_', ' ').title()}  
                    **Expected Loss**: {abs(worst_loss):.2f}%
                    
                    This scenario would result in significant portfolio losses. Consider:
                    - Increasing diversification
                    - Adding defensive assets
                    - Implementing stop-loss strategies
                    """)
                elif worst_loss < -10:
                    st.warning(f"""
                    **Worst Case Scenario**: {worst_case.replace('_', ' ').title()}  
                    **Expected Loss**: {abs(worst_loss):.2f}%
                    
                    Portfolio shows moderate stress vulnerability.
                    """)
                else:
                    st.success(f"""
                    **Worst Case Scenario**: {worst_case.replace('_', ' ').title()}  
                    **Expected Loss**: {abs(worst_loss):.2f}%
                    
                    Portfolio demonstrates good stress resilience.
                    """)
            
            # Monte Carlo results
            if monte_carlo:
                st.subheader("Monte Carlo Stress Simulation")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    var_30 = monte_carlo.get('var_95_30day', 0)
                    st.metric("VaR 95% (30-day)", f"{var_30:.2%}")
                with col2:
                    expected_shortfall = monte_carlo.get('expected_shortfall_95', 0)
                    st.metric("Expected Shortfall", f"{expected_shortfall:.2%}")
                with col3:
                    prob_loss_10 = monte_carlo.get('probability_loss_10pct', 0)
                    st.metric("Prob Loss >10%", f"{prob_loss_10:.1f}%")
                with col4:
                    simulations = monte_carlo.get('simulations', 0)
                    st.metric("Simulations", f"{simulations:,}")
                
                st.caption(f"Distribution: {monte_carlo.get('distribution_used', 'Unknown')}")
    
    elif analysis_type == "Volatility Forecasting":
        st.header("Volatility Forecasting (GARCH)")
        st.markdown("Forecast future volatility using GARCH models")
        
        forecast_horizon = st.slider(
            "Forecast Horizon (days)",
            min_value=5,
            max_value=90,
            value=30
        )
        
        if run_analysis:
            with st.spinner("Forecasting volatility..."):
                result = api_client.forecast_volatility_garch(symbols, forecast_horizon, period)
                
                if result:
                    st.session_state['volatility_forecast'] = result
        
        if 'volatility_forecast' in st.session_state:
            result = st.session_state['volatility_forecast']
            
            st.subheader("Volatility Forecast")
            
            volatility_data = result.get('volatility_forecast', {})
            
            if volatility_data:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Current Volatility",
                        f"{volatility_data.get('current_volatility', 0):.2%}"
                    )
                with col2:
                    st.metric(
                        "Forecast Volatility",
                        f"{volatility_data.get('forecast_mean', 0):.2%}"
                    )
                with col3:
                    trend = volatility_data.get('trend', 'stable')
                    st.metric("Trend", trend.capitalize())
                
                # Interpretation
                if trend == 'increasing':
                    st.warning("Volatility expected to increase - consider defensive positioning")
                elif trend == 'decreasing':
                    st.success("Volatility expected to decrease - favorable for risk assets")
                else:
                    st.info("Volatility expected to remain stable")
    
    # Footer
    st.markdown("---")
    st.caption("Risk metrics based on historical data and may not predict future performance")

if __name__ == "__main__":
    main()