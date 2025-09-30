"""
Enhanced Correlation Analysis Page
Time-varying, regime-conditional, clustering, and network analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.portfolio_manager import get_portfolio, set_portfolio, normalize_weights, initialize_portfolio
from utils.portfolio_presets import list_presets, get_preset
from utils.api_client import get_risk_api_client

st.set_page_config(page_title="Correlation Analysis", page_icon="🔗", layout="wide")

# Initialize portfolio
initialize_portfolio()

api_client = get_risk_api_client()

def plot_correlation_heatmap(corr_data, title="Correlation Matrix"):
    """Create correlation heatmap"""
    if isinstance(corr_data, dict):
        df = pd.DataFrame(corr_data)
    else:
        df = corr_data
    
    fig = px.imshow(
        df,
        title=title,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        zmin=-1, zmax=1
    )
    fig.update_layout(height=500)
    return fig

def main():
    st.title("🔗 Enhanced Correlation Analysis")
    st.markdown("Analyze correlation structures, dynamics, and network topology")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Analysis Configuration")
        
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
            height=150
        )
        symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]
        
        st.metric("Symbols Selected", len(symbols))
        
        # Analysis period
        period = st.selectbox(
            "Analysis Period",
            ["1month", "3months", "6months", "1year", "2years"],
            index=3
        )
        
        # Analysis type
        st.subheader("Analysis Type")
        analysis_type = st.radio(
            "Select Analysis",
            ["Basic Correlation", "Rolling Correlations", "Regime Correlations", 
             "Correlation Clustering", "Network Analysis", "Comprehensive"]
        )
        
        run_analysis = st.button("🚀 Run Analysis", type="primary", width='stretch')
    
    # Main content
    if not symbols:
        st.info("👈 Select a preset or enter at least 2 stock symbols to begin correlation analysis")
        return
    
    if len(symbols) < 2:
        st.warning("Please enter at least 2 symbols for correlation analysis")
        return
    
    # Analysis tabs
    if analysis_type == "Basic Correlation":
        st.header("Basic Correlation Analysis")
        
        if run_analysis:
            with st.spinner("Calculating correlations..."):
                result = api_client.correlation_analysis(symbols, period)
                
                if result:
                    st.session_state['corr_basic'] = result
        
        if 'corr_basic' in st.session_state:
            result = st.session_state['corr_basic']
            
            # Navigate nested structure
            correlation_data = result.get('correlation_analysis', {})
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                corr_matrix = correlation_data.get('correlation_matrix')
                if corr_matrix:
                    fig = plot_correlation_heatmap(corr_matrix, "Asset Correlation Matrix")
                    st.plotly_chart(fig, width='stretch')
            
            with col2:
                st.subheader("Summary Statistics")
                avg_corr = correlation_data.get('average_correlation', 0)
                div_score = correlation_data.get('diversification_score', 0)
                
                st.metric("Average Correlation", f"{avg_corr:.3f}")
                st.metric("Diversification Score", f"{div_score:.3f}")
                
                # Interpretation
                if avg_corr > 0.7:
                    st.error("High average correlation - limited diversification")
                elif avg_corr > 0.4:
                    st.warning("Moderate correlation - some diversification")
                else:
                    st.success("Low correlation - good diversification")
    
    elif analysis_type == "Rolling Correlations":
        st.header("Time-Varying Correlation Analysis")
        st.markdown("Analyze how correlations change over time")
        
        window_size = st.slider("Rolling Window (days)", 10, 90, 30)
        
        if run_analysis:
            with st.spinner("Calculating rolling correlations..."):
                result = api_client.rolling_correlations(symbols, window_size, period)
                
                if result:
                    st.session_state['corr_rolling'] = result
        
        if 'corr_rolling' in st.session_state:
            result = st.session_state['corr_rolling']
            
            # Navigate nested structure
            rolling_data = result.get('rolling_correlations', {})
            stability_metrics = rolling_data.get('stability_metrics', {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                stability_score = stability_metrics.get('stability_score', 0)
                st.metric("Stability Score", f"{stability_score:.3f}")
            with col2:
                num_windows = rolling_data.get('total_windows', 0)
                st.metric("Analysis Windows", num_windows)
            with col3:
                corr_volatility = stability_metrics.get('correlation_volatility', 0)
                st.metric("Correlation Volatility", f"{corr_volatility:.3f}")
            
            # Stability interpretation
            if stability_score > 0.8:
                st.success("High stability - consistent correlation relationships")
            elif stability_score > 0.6:
                st.info("Moderate stability - some variation in relationships")
            else:
                st.warning("Low stability - highly variable correlation dynamics")
            
            # Show pair correlations if available
            pair_correlations = rolling_data.get('pair_correlations', {})
            if pair_correlations:
                st.subheader("Correlation Pairs Analysis")
                
                pair_df = pd.DataFrame([
                    {
                        'Pair': pair_name,
                        'Mean Correlation': data['mean_correlation'],
                        'Min': data['correlation_range'][0],
                        'Max': data['correlation_range'][1],
                        'Volatility': data['correlation_volatility']
                    }
                    for pair_name, data in pair_correlations.items()
                ])
                
                st.dataframe(pair_df, width='stretch')
    
    elif analysis_type == "Regime Correlations":
        st.header("Regime-Conditional Correlation Analysis")
        st.markdown("How do correlations change during different market regimes?")
        
        regime_method = st.selectbox(
            "Regime Detection Method",
            ["volatility", "hmm", "returns"],
            format_func=lambda x: {
                "volatility": "Volatility-Based",
                "hmm": "Hidden Markov Model",
                "returns": "Returns-Based"
            }[x]
        )
        
        if run_analysis:
            with st.spinner("Analyzing regime-conditional correlations..."):
                result = api_client.regime_correlations(symbols, regime_method, period)
                
                if result:
                    st.session_state['corr_regime'] = result
        
        if 'corr_regime' in st.session_state:
            result = st.session_state['corr_regime']
            
            # Navigate nested structure properly
            regime_data = result.get('regime_correlations', {})
            regime_sensitivity = regime_data.get('regime_sensitivity', {})
            market_regimes = regime_data.get('market_regime_correlations', {})
            
            # Extract the actual values
            crisis_multiplier = regime_sensitivity.get('crisis_correlation_multiplier', 0)
            bull_corr = market_regimes.get('bull', {}).get('avg_correlation', 0)
            bear_corr = market_regimes.get('bear', {}).get('avg_correlation', 0)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Crisis Multiplier", f"{crisis_multiplier:.2f}x")
            with col2:
                st.metric("Normal Correlation", f"{bull_corr:.3f}")
            with col3:
                st.metric("Crisis Correlation", f"{bear_corr:.3f}")
            
            # Crisis multiplier interpretation
            if crisis_multiplier > 2.0:
                st.error("Severe correlation increase during crisis - diversification breaks down")
            elif crisis_multiplier > 1.5:
                st.warning("Significant correlation increase - reduced diversification in stress")
            else:
                st.success("Correlations remain stable across regimes")
            
            # Show regime details
            st.subheader("Regime Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Bull Market Regime**")
                bull_data = market_regimes.get('bull', {})
                st.write(f"Periods: {bull_data.get('period_count', 0)}")
                st.write(f"Percentage: {bull_data.get('period_percentage', 0):.1%}")
                st.write(f"Avg Correlation: {bull_corr:.3f}")
            
            with col2:
                st.markdown("**Bear Market Regime**")
                bear_data = market_regimes.get('bear', {})
                st.write(f"Periods: {bear_data.get('period_count', 0)}")
                st.write(f"Percentage: {bear_data.get('period_percentage', 0):.1%}")
                st.write(f"Avg Correlation: {bear_corr:.3f}")
            
            st.markdown("---")
            st.markdown("""
            **Crisis Multiplier Interpretation:**
            - **< 1.3x**: Correlations stable across regimes
            - **1.3-1.8x**: Moderate increase in crisis periods
            - **> 1.8x**: Significant diversification breakdown during stress
            """)
    
    elif analysis_type == "Correlation Clustering":
        st.header("Hierarchical Correlation Clustering")
        st.markdown("Identify groups of assets with similar correlation behavior")
        
        if run_analysis:
            with st.spinner("Performing clustering analysis..."):
                result = api_client.correlation_clustering(symbols, period)
                
                if result:
                    st.session_state['corr_clustering'] = result
        
        if 'corr_clustering' in st.session_state:
            result = st.session_state['corr_clustering']
            
            # Navigate nested structure
            clustering_data = result.get('correlation_clustering', {})
            quality_metrics = clustering_data.get('quality_metrics', {})
            cluster_analysis = clustering_data.get('cluster_analysis', {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                optimal_clusters = clustering_data.get('optimal_clusters', 0)
                st.metric("Optimal Clusters", optimal_clusters)
            with col2:
                clustering_efficiency = quality_metrics.get('clustering_efficiency', 0)
                st.metric("Clustering Efficiency", f"{clustering_efficiency:.3f}")
            with col3:
                st.metric("Total Assets", len(symbols))
            
            # Show cluster assignments
            if cluster_analysis:
                st.subheader("Cluster Assignments")
                
                # Create DataFrame for cluster visualization
                cluster_rows = []
                for cluster_name, cluster_data in cluster_analysis.items():
                    cluster_symbols = cluster_data.get('symbols', [])
                    cluster_size = cluster_data.get('size', 0)
                    avg_corr = cluster_data.get('avg_internal_correlation', 0)
                    
                    if cluster_size > 0:
                        cluster_rows.append({
                            'Cluster': cluster_name.replace('cluster_', 'Cluster '),
                            'Symbols': ', '.join(cluster_symbols),
                            'Size': cluster_size,
                            'Avg Correlation': f"{avg_corr:.3f}" if avg_corr > 0 else "N/A"
                        })
                
                if cluster_rows:
                    cluster_df = pd.DataFrame(cluster_rows)
                    st.dataframe(cluster_df, width='stretch')
    
    elif analysis_type == "Network Analysis":
        st.header("Correlation Network Topology")
        st.markdown("Analyze the network structure of correlation relationships")
        
        if run_analysis:
            with st.spinner("Analyzing network topology..."):
                result = api_client.correlation_network(symbols, period)
                
                if result:
                    st.session_state['corr_network'] = result
        
        if 'corr_network' in st.session_state:
            result = st.session_state['corr_network']
            
            # Navigate nested structure
            network_data = result.get('correlation_network', {})
            network_health = network_data.get('network_health', {})
            degree_centrality = network_data.get('degree_centrality', {})
            systemic_importance = network_data.get('systemic_importance', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                network_density = network_health.get('network_density', 0)
                st.metric("Network Density", f"{network_density:.3f}")
            with col2:
                total_connections = network_health.get('total_connections', 0)
                avg_connections = total_connections / len(symbols) if symbols else 0
                st.metric("Avg Connections", f"{avg_connections:.1f}")
            with col3:
                # Find most central asset
                if degree_centrality:
                    most_central = max(degree_centrality.items(), key=lambda x: x[1]['degree'])[0]
                    st.metric("Central Node", most_central)
                else:
                    st.metric("Central Node", "N/A")
            with col4:
                network_efficiency = network_health.get('network_efficiency', 0)
                st.metric("Network Efficiency", f"{network_efficiency:.3f}")
            
            # Network density interpretation
            if network_density > 0.7:
                st.warning("High network density - assets are highly interconnected")
            elif network_density > 0.4:
                st.info("Moderate network density - some independence between assets")
            else:
                st.success("Low network density - assets show independence")
            
            # Show systemic importance ranking
            if systemic_importance:
                st.subheader("Systemic Importance Ranking")
                
                importance_df = pd.DataFrame([
                    {
                        'Symbol': symbol,
                        'Importance Score': data['importance_score'],
                        'Rank': data['systemic_rank'],
                        'Risk Contribution': data['risk_contribution']
                    }
                    for symbol, data in systemic_importance.items()
                ]).sort_values('Rank')
                
                st.dataframe(importance_df, width='stretch')
    
    elif analysis_type == "Comprehensive":
        st.header("Comprehensive Correlation Analysis")
        st.markdown("Integrated analysis combining all correlation methods")
        
        if run_analysis:
            with st.spinner("Running comprehensive analysis..."):
                result = api_client.comprehensive_correlation(symbols, period)
                
                if result:
                    st.session_state['corr_comprehensive'] = result
        
        if 'corr_comprehensive' in st.session_state:
            result = st.session_state['corr_comprehensive']
            
            # Navigate nested structure
            comprehensive_data = result.get('comprehensive_correlation', {})
            rolling_analysis = comprehensive_data.get('rolling_analysis', {})
            regime_analysis = comprehensive_data.get('regime_analysis', {})
            clustering_analysis = comprehensive_data.get('clustering_analysis', {})
            network_analysis = comprehensive_data.get('network_analysis', {})
            insights = comprehensive_data.get('synthesized_insights', [])
            
            # Extract key metrics
            stability_score = rolling_analysis.get('stability_metrics', {}).get('stability_score', 0)
            crisis_multiplier = regime_analysis.get('regime_sensitivity', {}).get('crisis_correlation_multiplier', 0)
            optimal_clusters = clustering_analysis.get('optimal_clusters', 0)
            network_density = network_analysis.get('network_health', {}).get('network_density', 0)
            
            # Summary metrics
            st.subheader("Key Findings")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Stability Score", f"{stability_score:.3f}")
                st.metric("Network Density", f"{network_density:.3f}")
            
            with col2:
                st.metric("Crisis Multiplier", f"{crisis_multiplier:.2f}x")
                st.metric("Optimal Clusters", optimal_clusters)
            
            with col3:
                bull_corr = regime_analysis.get('market_regime_correlations', {}).get('bull', {}).get('avg_correlation', 0)
                bear_corr = regime_analysis.get('market_regime_correlations', {}).get('bear', {}).get('avg_correlation', 0)
                st.metric("Bull Correlation", f"{bull_corr:.3f}")
                st.metric("Bear Correlation", f"{bear_corr:.3f}")
            
            # Synthesized insights
            if insights:
                st.subheader("Synthesized Insights")
                for i, insight in enumerate(insights, 1):
                    st.info(f"**Insight {i}:** {insight}")
            
            # Latest correlation matrix from rolling analysis
            latest_corr = rolling_analysis.get('latest_correlation_matrix')
            if latest_corr:
                st.subheader("Current Correlation Matrix")
                fig = plot_correlation_heatmap(latest_corr, "Comprehensive Correlation Analysis")
                st.plotly_chart(fig, width='stretch')
    
    # Footer tips
    st.markdown("---")
    st.caption("""
    **Tips:**
    - Use Rolling Correlations to identify changing relationships over time
    - Regime Correlations show how diversification breaks down in crises
    - Network Analysis reveals which assets are most interconnected
    - Comprehensive Analysis provides an integrated view across all methods
    """)

if __name__ == "__main__":
    main()