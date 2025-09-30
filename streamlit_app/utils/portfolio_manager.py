"""
Shared Portfolio State Management
Allows portfolio configuration to persist across pages
"""

import streamlit as st

def initialize_portfolio():
    """Initialize portfolio in session state if not exists"""
    if 'portfolio_symbols' not in st.session_state:
        st.session_state['portfolio_symbols'] = ['AAPL', 'MSFT', 'GOOGL']
    if 'portfolio_weights' not in st.session_state:
        st.session_state['portfolio_weights'] = [0.33, 0.33, 0.34]

def get_portfolio():
    """Get current portfolio configuration"""
    initialize_portfolio()
    return (
        st.session_state['portfolio_symbols'],
        st.session_state['portfolio_weights']
    )

def set_portfolio(symbols, weights):
    """Update portfolio configuration"""
    st.session_state['portfolio_symbols'] = symbols
    st.session_state['portfolio_weights'] = weights

def normalize_weights(weights):
    """Normalize weights to sum to 1.0"""
    total = sum(weights)
    return [w/total for w in weights] if total > 0 else weights