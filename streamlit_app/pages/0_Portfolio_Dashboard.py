"""
Portfolio Dashboard - User Testing Version
Simple portfolio input with clear next steps
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import time
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.portfolio_manager import get_portfolio, set_portfolio, normalize_weights
from utils.portfolio_presets import list_presets, get_preset
from utils.request_logger import request_logger
from utils.styling import inject_custom_css, add_sidebar_branding

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {'symbols': [], 'weights': []}

def main():
    inject_custom_css()

    # Header
    st.markdown("""
    <div style='margin-bottom: 2rem;'>
        <h1 style='margin-bottom: 0.5rem;'>📊 Portfolio Dashboard</h1>
        <p style='color: #808495; font-size: 1.1rem; margin: 0;'>
            Define your portfolio to get started with analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        add_sidebar_branding()
        
        st.markdown("### 📥 Quick Load")
        st.caption("Try a sample portfolio")
        
        preset_options = ["Select a preset..."] + list(list_presets().values())
        preset_name = st.selectbox("Preset", preset_options, label_visibility="collapsed")

        if preset_name != "Select a preset...":
            preset_key = [k for k, v in list_presets().items() if v == preset_name][0]
            preset = get_preset(preset_key)
            
            if st.button("Load Preset", use_container_width=True):
                set_portfolio(preset["symbols"], preset["weights"])
                st.success("Loaded!")
                time.sleep(0.5)
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 🎯 What's Next?")
        st.caption("After defining your portfolio:")
        
        if st.button("View Health Report", use_container_width=True):
            st.switch_page("pages/6_Portfolio_Insights.py")
        
        if st.button("Optimize Weights", use_container_width=True):
            st.switch_page("pages/1_Portfolio_Analysis.py")
        
        if st.button("Run Stress Test", use_container_width=True):
            st.switch_page("pages/2_Risk_Analytics.py")

    symbols, weights = get_portfolio()

    # Empty state
    if not symbols:
        st.markdown("""
        <div style='text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%); border-radius: 16px; margin: 2rem 0;'>
            <div style='font-size: 4rem; margin-bottom: 1rem;'>👋</div>
            <h2 style='color: #667eea; margin-bottom: 1rem;'>Welcome!</h2>
            <p style='color: #808495; font-size: 1.1rem; max-width: 600px; margin: 0 auto;'>
                Load a sample portfolio from the sidebar or enter your own below to get started.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Portfolio Input
    st.markdown("## 📝 Your Portfolio")
    
    tab1, tab2 = st.tabs(["Manual Entry", "Paste from Spreadsheet"])
    
    with tab1:
        st.caption("Enter your holdings one by one")
        
        with st.form("manual_entry_form"):
            num_holdings = st.number_input(
                "How many holdings?", 
                min_value=2, 
                max_value=20, 
                value=len(symbols) if symbols else 5
            )
            
            st.markdown("---")
            
            symbols_input = []
            weights_input = []
            
            for i in range(num_holdings):
                col1, col2 = st.columns([2, 1])
                with col1:
                    default_symbol = symbols[i] if i < len(symbols) else ""
                    symbol = st.text_input(
                        f"Symbol {i+1}", 
                        value=default_symbol,
                        placeholder="AAPL",
                        key=f"sym_{i}"
                    )
                    symbols_input.append(symbol.upper().strip())
                
                with col2:
                    default_weight = weights[i] * 100 if i < len(weights) else 100.0 / num_holdings
                    weight = st.number_input(
                        f"Weight %", 
                        min_value=0.0, 
                        max_value=100.0, 
                        value=default_weight,
                        step=1.0,
                        key=f"wt_{i}"
                    )
                    weights_input.append(weight / 100)
            
            # Show total
            total_weight = sum(w * 100 for w in weights_input)
            if abs(total_weight - 100) < 0.01:
                st.success(f"Total: {total_weight:.1f}%")
            else:
                st.error(f"Total: {total_weight:.1f}% (must equal 100%)")
            
            submitted = st.form_submit_button("Update Portfolio", type="primary", use_container_width=True)
            
            if submitted:
                symbols_clean = [s for s in symbols_input if s]
                weights_clean = weights_input[:len(symbols_clean)]
                total = sum(w * 100 for w in weights_clean)
                
                if len(symbols_clean) < 2:
                    st.error("Enter at least 2 symbols")
                elif abs(total - 100) > 0.01:
                    st.error(f"Weights total {total:.1f}%, must equal 100%")
                else:
                    set_portfolio(symbols_clean, weights_clean)
                    st.success(f"Portfolio updated: {len(symbols_clean)} holdings")
                    time.sleep(0.5)
                    st.rerun()
    
    with tab2:
        st.caption("Copy from Excel or Google Sheets")
        
        paste_data = st.text_area(
            "Paste data here",
            height=200,
            placeholder="AAPL\t30\nMSFT\t25\nGOOGL\t20\nNVDA\t15\nTSLA\t10",
            help="Format: Symbol [Tab or Comma] Weight (one per line)"
        )
        
        if st.button("Load from Paste", type="primary", use_container_width=True):
            try:
                lines = [line.strip() for line in paste_data.split('\n') if line.strip()]
                symbols_paste = []
                weights_paste = []
                
                for line in lines:
                    parts = line.replace(',', '\t').split('\t')
                    if len(parts) >= 2:
                        symbols_paste.append(parts[0].strip().upper())
                        weights_paste.append(float(parts[1].strip()) / 100)
                
                total = sum(w * 100 for w in weights_paste)
                
                if len(symbols_paste) < 2:
                    st.error("Need at least 2 holdings")
                elif abs(total - 100) > 0.01:
                    st.error(f"Weights total {total:.1f}%, must equal 100%")
                else:
                    set_portfolio(symbols_paste, weights_paste)
                    st.success(f"Loaded {len(symbols_paste)} holdings")
                    time.sleep(0.5)
                    st.rerun()
            
            except Exception as e:
                st.error(f"Error parsing data: {str(e)}")
                st.caption("Expected format: Symbol [Tab or Comma] Weight")

    # Current Portfolio Display
    if symbols:
        st.markdown("---")
        st.markdown("## Current Portfolio")
        
        col1, col2 = st.columns([2, 1], gap="large")
        
        with col1:
            fig = px.pie(
                values=weights, 
                names=symbols, 
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                height=350, 
                showlegend=False, 
                margin=dict(t=0, b=0, l=0, r=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Summary")
            st.metric("Total Holdings", len(symbols))
            
            largest = max(zip(symbols, weights), key=lambda x: x[1])
            st.metric("Largest Position", f"{largest[0]}")
            st.caption(f"{largest[1]:.1%} of portfolio")
            
            st.markdown("---")
            st.markdown("#### Quick Actions")
            
            if st.button("📊 Get Health Report", use_container_width=True, type="primary"):
                st.switch_page("pages/6_Portfolio_Insights.py")
            
            if st.button("🎯 Optimize Portfolio", use_container_width=True):
                st.switch_page("pages/1_Portfolio_Analysis.py")
            
            if st.button("🔥 Run Stress Test", use_container_width=True):
                st.switch_page("pages/2_Risk_Analytics.py")
        
        st.markdown("---")
        st.markdown("### 📚 Help & Feedback")
        st.markdown("[Quick Start Guide](bit.ly/4gWHEMu)")
        st.markdown("[Give Feedback](https://forms.gle/87hpD7gvPVQnsPfc7)")
        # Holdings table
        st.markdown("---")
        st.markdown("#### Holdings Detail")
        
        df = pd.DataFrame({
            'Symbol': symbols,
            'Weight': [f"{w:.1%}" for w in weights],
            'Type': ['Equity'] * len(symbols)  # Placeholder
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("An error occurred on the Dashboard.")
        request_logger.logger.exception("Unhandled exception in Dashboard")
        with st.expander("Error Details"):
            st.code(str(e))