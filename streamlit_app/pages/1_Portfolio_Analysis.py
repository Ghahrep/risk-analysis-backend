"""
Portfolio Analysis Page - Enhanced UX for User Testing
Key Improvements: Table-based weight entry, trade breakdown, action bridges
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
from utils.cache_utils import (
    cached_risk_analysis,
    cached_optimization,
    save_analysis_timestamp,
    show_staleness_warning,
    show_last_updated_badge,
    to_hashable
)
from utils.styling import (
    inject_custom_css,
    add_page_header,
    add_sidebar_branding,
    show_weight_summary,
    show_empty_state,
    add_footer_tip
)
from utils.metric_benchmarks import display_metric_with_benchmark

st.set_page_config(page_title="Portfolio Analysis", page_icon="📈", layout="wide")

initialize_portfolio()
api_client = get_risk_api_client()

def load_example_portfolio(portfolio_type):
    """Load example portfolios for quick testing"""
    examples = {
        "balanced": {
            "symbols": ["VTI", "BND", "VEA", "GLD"],
            "weights": [0.40, 0.30, 0.20, 0.10],
            "name": "Balanced 60/40"
        },
        "growth": {
            "symbols": ["QQQ", "VUG", "VXUS", "VTI"],
            "weights": [0.35, 0.30, 0.20, 0.15],
            "name": "Growth-Focused"
        },
        "income": {
            "symbols": ["BND", "VCIT", "VYM", "VNQ"],
            "weights": [0.35, 0.30, 0.20, 0.15],
            "name": "Income-Focused"
        }
    }
    
    if portfolio_type in examples:
        portfolio = examples[portfolio_type]
        set_portfolio(portfolio["symbols"], portfolio["weights"])
        st.success(f"✓ {portfolio['name']} portfolio loaded!")
        time.sleep(0.5)
        st.rerun()

def create_weights_comparison_chart(current_weights, optimized_weights, symbols):
    """Create side-by-side comparison of current vs optimized weights"""
    df = pd.DataFrame({
        'Symbol': symbols * 2,
        'Weight': list(current_weights) + list(optimized_weights.values()),
        'Type': ['Current'] * len(symbols) + ['Optimized'] * len(symbols)
    })
    
    fig = px.bar(
        df,
        x='Symbol',
        y='Weight',
        color='Type',
        barmode='group',
        title='Current vs Optimized Allocation',
        color_discrete_map={'Current': '#667eea', 'Optimized': '#28a745'}
    )
    
    fig.update_layout(
        height=400,
        yaxis_title='Portfolio Weight',
        yaxis_tickformat='.0%',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def show_trade_breakdown(current_weights, optimized_weights, symbols, portfolio_value=100000):
    """NEW: Show what trades are needed to move from current to optimized"""
    st.markdown("### 📋 Trade Breakdown")
    st.caption(f"Based on ${portfolio_value:,.0f} portfolio value")
    
    trades = []
    for i, symbol in enumerate(symbols):
        current = current_weights[i]
        optimized = list(optimized_weights.values())[i]
        change = optimized - current
        
        if abs(change) > 0.001:  # Only show meaningful changes
            current_value = current * portfolio_value
            optimized_value = optimized * portfolio_value
            trade_value = abs(change * portfolio_value)
            
            trades.append({
                'Symbol': symbol,
                'Action': 'BUY' if change > 0 else 'SELL',
                'Current': f"{current:.1%}",
                'Target': f"{optimized:.1%}",
                'Change': f"{change:+.1%}",
                'Trade Value': f"${trade_value:,.0f}"
            })
    
    if trades:
        trades_df = pd.DataFrame(trades)
        
        # Color code by action
        def highlight_action(row):
            if row['Action'] == 'BUY':
                return ['background-color: rgba(40, 167, 69, 0.1)'] * len(row)
            else:
                return ['background-color: rgba(220, 53, 69, 0.1)'] * len(row)
        
        st.dataframe(
            trades_df.style.apply(highlight_action, axis=1),
            hide_index=True,
            use_container_width=True
        )
        
        total_turnover = sum(abs(t['Change'].strip('%+').replace(',','')) for t in trades) / 2
        st.info(f"Total portfolio turnover: ~{total_turnover:.1f}% (half of total changes)")
    else:
        st.success("No trades needed - portfolio is already optimal!")

def main():
    inject_custom_css()
    
    add_page_header(
        "Portfolio Analysis & Optimization",
        "Optimize your portfolio allocation for maximum risk-adjusted returns",
        "📈"
    )
    
    # Staleness warning
    show_staleness_warning('portfolio_analysis')
    
    # IMPROVED SIDEBAR: Simplified weight entry
    with st.sidebar:
        add_sidebar_branding()
        
        st.markdown("### Quick Actions")
        
        # Load from presets
        preset_options = ["Custom Portfolio"] + list(list_presets().values())
        preset_name = st.selectbox("Load Preset", preset_options)
        
        if preset_name != "Custom Portfolio":
            preset_key = [k for k, v in list_presets().items() if v == preset_name][0]
            preset = get_preset(preset_key)
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                        padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0;'>
                <small style='color: #667eea; font-weight: 600;'>{preset['description']}</small>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("📥 Load This Preset", key="load_preset_portfolio", use_container_width=True):
                request_logger.log_user_action("load_preset", {"preset": preset_name, "page": "portfolio_analysis"})
                
                with st.spinner("Loading..."):
                    time.sleep(0.3)
                    set_portfolio(preset['symbols'], preset['weights'])
                
                st.success(f"✓ Loaded!")
                time.sleep(1)
                st.rerun()
        
        st.markdown("---")
        st.markdown("### Goal-Based Navigation")
        
        if st.button("🔍 Understand Risk", use_container_width=True):
            st.switch_page("pages/2_Risk_Analytics.py")
        
        if st.button("🔗 Check Diversification", use_container_width=True):
            st.switch_page("pages/3_Correlation_Analysis.py")
        
        if st.button("💡 Get Health Report", use_container_width=True):
            st.switch_page("pages/6_Portfolio_Insights.py")
        
        st.markdown("---")
        
        # Feedback section
        st.markdown("---")
        st.markdown("### 📚 Help & Feedback")
        st.markdown("[Quick Start Guide](https://docs.google.com/document/d/1BX93dy0fOcFdeiiXxT3T7XlDgdC4CRv5Ehp72linzIc/view)")
        st.markdown("[Give Feedback](https://forms.gle/87hpD7gvPVQnsPfc7)")

    symbols, weights = get_portfolio()
      
    # IMPROVED EMPTY STATE
    if not symbols:
        show_empty_state(
            icon="📈",
            title="Portfolio Optimization Ready",
            message="Load a preset or enter your holdings to begin optimization"
        )
        
        st.markdown("### 🚀 Quick Start - Example Portfolios")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 12px; border: 2px solid rgba(102, 126, 234, 0.2); height: 100%;'>
                <h4 style='color: #667eea; margin-top: 0;'>⚖️ Balanced 60/40</h4>
                <p style='color: #808495; font-size: 0.9rem;'>Traditional stocks/bonds split<br>VTI, BND, VEA, GLD</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Try Balanced", key="try_balanced", use_container_width=True):
                load_example_portfolio("balanced")
        
        with col2:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 12px; border: 2px solid rgba(102, 126, 234, 0.2); height: 100%;'>
                <h4 style='color: #667eea; margin-top: 0;'>📈 Growth-Focused</h4>
                <p style='color: #808495; font-size: 0.9rem;'>Equity-heavy allocation<br>QQQ, VUG, VXUS, VTI</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Try Growth", key="try_growth", use_container_width=True):
                load_example_portfolio("growth")
        
        with col3:
            st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 12px; border: 2px solid rgba(102, 126, 234, 0.2); height: 100%;'>
                <h4 style='color: #667eea; margin-top: 0;'>💰 Income-Focused</h4>
                <p style='color: #808495; font-size: 0.9rem;'>Dividend & bond heavy<br>BND, VCIT, VYM, VNQ</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Try Income", key="try_income", use_container_width=True):
                load_example_portfolio("income")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 **Tip:** Optimization finds the allocation that maximizes your Sharpe ratio (return per unit of risk)")
        return
    
    # Validate portfolio
    is_valid, error_msg = validate_portfolio(symbols, weights)
    if not is_valid:
        st.error(f"⚠️ {error_msg}")
        return
    
    # Display current portfolio with IMPROVED WEIGHT EDITING
    st.markdown("## Current Portfolio")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.pie(
            values=weights,
            names=symbols,
            title="Current Allocation",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Quick Edit Weights")
        st.caption("Adjust allocation (auto-normalizes to 100%)")
        
        # NEW: Table-based weight entry instead of sliders
        new_weights = []
        for i, symbol in enumerate(symbols):
            weight_pct = st.number_input(
                symbol,
                min_value=0.0,
                max_value=100.0,
                value=weights[i] * 100,
                step=5.0,
                key=f"weight_edit_{symbol}",
                help=f"Percentage allocation for {symbol}"
            )
            new_weights.append(weight_pct / 100)
        
        # Show total
        total = sum(new_weights)
        if abs(total - 1.0) > 0.01:
            st.warning(f"Total: {total*100:.0f}% (will normalize)")
        else:
            st.success(f"✓ Total: 100%")
        
        if st.button("💾 Update Weights", use_container_width=True):
            normalized = normalize_weights(new_weights)
            set_portfolio(symbols, normalized)
            st.success("✓ Updated!")
            time.sleep(0.5)
            st.rerun()
    
    st.markdown("---")
    
    # Optimization section
    st.markdown("## Portfolio Optimization")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        optimization_method = st.selectbox(
            "Optimization Objective",
            ["max_sharpe", "min_variance", "equal_weight"],
            format_func=lambda x: {
                "max_sharpe": "🎯 Max Sharpe (Recommended)",
                "min_variance": "🛡️ Min Volatility",
                "equal_weight": "⚖️ Equal Weight Baseline"
            }[x]
        )
    
    with col2:
        period = st.selectbox(
            "Historical Period",
            ["1month", "3months", "6months", "1year", "2years"],
            index=3,
            format_func=lambda x: x.replace("month", " Month").replace("year", " Year").replace("s", "s").title()
        )
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        optimize_button = st.button("🔄 Run Optimization", type="primary", use_container_width=True)
    
    # Quick explainer
    with st.expander("ℹ️ What does this do?"):
        method_info = {
            "max_sharpe": "Finds the allocation with the **best risk-adjusted returns** (highest return per unit of risk). Recommended for most investors seeking optimal performance.",
            "min_variance": "Minimizes portfolio volatility by finding the **most stable allocation**. Best for conservative investors prioritizing capital preservation over maximum returns.",
            "equal_weight": "Simple 1/N allocation giving **equal weight to each holding**. Useful as a benchmark to compare against optimized strategies."
        }
        st.info(method_info[optimization_method])
    
    # Run optimization
    if optimize_button:
        request_logger.log_user_action("optimize_portfolio", {
            "symbols": symbols,
            "method": optimization_method,
            "period": period
        })
        
        with st.spinner(f"🔄 Optimizing using {optimization_method.replace('_', ' ')}... 10-15 seconds"):
            symbols_tuple, weights_tuple = to_hashable(symbols, weights)
            
            result = safe_api_call(
                lambda: cached_optimization(symbols_tuple, optimization_method, period, api_client),
                error_context="portfolio optimization"
            )
            
            if result:
                st.session_state['optimization_result'] = result
                st.session_state['current_symbols'] = symbols
                st.session_state['current_weights'] = weights
                save_analysis_timestamp('portfolio_analysis')
                st.success("✓ Optimization complete!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("❌ Optimization failed. Try a different period or verify symbols are valid.")
    
    # Display optimization results with ACTION BRIDGES
    if 'optimization_result' in st.session_state:
        result = st.session_state['optimization_result']
        
        show_last_updated_badge('portfolio_analysis')
        
        st.markdown("---")
        st.markdown("## Optimization Results")
        
        opt_weights = result.get('optimized_weights', {})
        
        if not opt_weights:
            st.warning("⚠️ No optimization results available. Try running optimization again.")
            return
        
        # Performance metrics
        st.markdown("### 📊 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            expected_return = result.get('expected_return', 0)
            st.metric(
                "Expected Return",
                f"{expected_return:.2%}",
                delta="Annual",
                help="Projected annual return based on historical data"
            )
        
        with col2:
            volatility = result.get('volatility', 0)
            st.metric(
                "Volatility",
                f"{volatility:.2%}",
                delta="Risk measure",
                help="Expected annual price fluctuation"
            )
        
        with col3:
            sharpe = result.get('sharpe_ratio', 0)
            benchmark = 1.0
            delta_val = sharpe - benchmark
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.2f}",
                delta=f"{delta_val:+.2f} vs benchmark",
                delta_color="normal" if delta_val > 0 else "inverse",
                help="Return per unit of risk (higher is better)"
            )
        
        with col4:
            max_dd = result.get('max_drawdown', 0)
            st.metric(
                "Max Drawdown",
                f"{abs(max_dd):.1%}" if max_dd else "N/A",
                delta="Historical worst case",
                help="Largest peak-to-trough decline"
            )
        
        # NEW: Interpretation
        with st.expander("🤖 What does this mean?", expanded=True):
            if sharpe > 1.5:
                st.success(f"""
                **Strong performance!** Your optimized portfolio has a Sharpe ratio of {sharpe:.2f}, 
                indicating excellent risk-adjusted returns. This allocation delivers good returns 
                relative to the volatility you're taking on.
                """)
            elif sharpe > 0.8:
                st.info(f"""
                **Solid performance.** Your Sharpe ratio of {sharpe:.2f} shows decent risk-adjusted returns. 
                This is a reasonable allocation, though there may be room for improvement with different holdings.
                """)
            else:
                st.warning(f"""
                **Room for improvement.** A Sharpe ratio of {sharpe:.2f} suggests your returns may not 
                justify the risk. Consider diversifying across less correlated assets or adjusting your holdings.
                """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Weights comparison with TRADE BREAKDOWN
        st.markdown("### 📈 Allocation Changes")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            current_weights_for_chart = st.session_state.get('current_weights', weights)
            fig = create_weights_comparison_chart(current_weights_for_chart, opt_weights, symbols)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # NEW: Trade Breakdown
            show_trade_breakdown(
                st.session_state.get('current_weights', weights),
                opt_weights,
                symbols
            )
        
        # NEW: IMPROVED ACTION BRIDGE - Clear next steps
        st.markdown("---")
        st.markdown("### 🎯 Next Steps")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 1️⃣ Apply Changes")
            st.caption("Update your portfolio with optimized weights")
            
            if st.button("✅ Apply Optimized Weights", type="primary", use_container_width=True):
                new_symbols = list(opt_weights.keys())
                new_weights = list(opt_weights.values())
                set_portfolio(new_symbols, new_weights)
                
                # Clear optimization result after applying
                del st.session_state['optimization_result']
                
                st.success("✓ Portfolio updated!")
                st.balloons()
                time.sleep(1.5)
                st.rerun()
        
        with col2:
            st.markdown("#### 2️⃣ Test Resilience")
            st.caption("See how this allocation handles stress")
            
            if st.button("💥 Run Stress Test", use_container_width=True):
                # Store optimized weights for stress testing
                st.session_state['pending_stress_test'] = {
                    'symbols': list(opt_weights.keys()),
                    'weights': list(opt_weights.values())
                }
                st.switch_page("pages/2_Risk_Analytics.py")
        
        with col3:
            st.markdown("#### 3️⃣ Get Insights")
            st.caption("Full health report on optimized portfolio")
            
            if st.button("📊 View Health Report", use_container_width=True):
                # Apply weights temporarily for insights
                new_symbols = list(opt_weights.keys())
                new_weights = list(opt_weights.values())
                set_portfolio(new_symbols, new_weights)
                st.switch_page("pages/6_Portfolio_Insights.py")
        
        # Comparison table
        with st.expander("📋 Detailed Comparison"):
            weights_df = pd.DataFrame({
                'Symbol': list(opt_weights.keys()),
                'Current': [f"{st.session_state.get('current_weights', weights)[i]:.1%}" for i in range(len(symbols))],
                'Optimized': [f"{w:.1%}" for w in opt_weights.values()],
                'Change': [
                    f"{(list(opt_weights.values())[i] - st.session_state.get('current_weights', weights)[i]):.1%}" 
                    for i in range(len(symbols))
                ]
            })
            st.dataframe(weights_df, hide_index=True, use_container_width=True)
    
    # Footer
    st.markdown("---")
    add_footer_tip("💡 Optimization uses historical data. Rebalance regularly as market conditions change.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("An unexpected error occurred in Portfolio Analysis")
        import logging
        logging.exception("Unhandled exception in Portfolio Analysis")
        with st.expander("🔍 Error Details"):
            st.code(str(e))