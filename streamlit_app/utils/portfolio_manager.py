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

def normalize_weights(weights: list) -> list:
    """
    Normalize weights to sum to 1.0
    
    Args:
        weights: List of numerical weights
        
    Returns:
        List of normalized weights summing to 1.0
        
    Raises:
        ValueError: If weights list is empty or sums to zero
    """
    # Check for empty list
    if not weights or len(weights) == 0:
        raise ValueError("Cannot normalize empty weights list")
    
    # Calculate sum
    total = sum(weights)
    
    # Check for zero sum
    if total == 0 or abs(total) < 1e-10:
        raise ValueError("Cannot normalize weights that sum to zero")
    
    # Normalize and return
    return [w / total for w in weights]