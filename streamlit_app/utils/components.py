"""
Reusable UI components
"""

import streamlit as st
import plotly.graph_objects as go

def metric_card(title, value, change=None, icon="📊"):
    """Enhanced metric card with icon"""
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown(f"<div style='font-size: 3rem; text-align: center;'>{icon}</div>", 
                   unsafe_allow_html=True)
    with col2:
        st.metric(title, value, delta=change)

def status_badge(status, message):
    """Create status badge"""
    colors = {
        "success": "#28a745",
        "warning": "#ffc107",
        "error": "#dc3545",
        "info": "#17a2b8"
    }
    
    color = colors.get(status, colors["info"])
    
    st.markdown(f"""
    <div style='
        display: inline-block;
        background-color: {color}22;
        color: {color};
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: 2px solid {color};
        font-weight: 600;
        font-size: 0.9rem;
    '>
        {message}
    </div>
    """, unsafe_allow_html=True)

def progress_bar(label, value, max_value=100):
    """Custom progress bar"""
    percentage = (value / max_value) * 100
    color = "#28a745" if percentage > 70 else "#ffc107" if percentage > 40 else "#dc3545"
    
    st.markdown(f"""
    <div style='margin: 1rem 0;'>
        <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
            <span style='font-weight: 600;'>{label}</span>
            <span style='color: #808495;'>{value}/{max_value}</span>
        </div>
        <div style='
            width: 100%;
            height: 8px;
            background-color: #262730;
            border-radius: 4px;
            overflow: hidden;
        '>
            <div style='
                width: {percentage}%;
                height: 100%;
                background-color: {color};
                transition: width 0.3s ease;
            '></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def info_tooltip(text, tooltip):
    """Text with tooltip"""
    st.markdown(f"""
    <div style='display: inline-block; position: relative; cursor: help;'>
        {text}
        <span style='
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background-color: #262730;
            color: #fafafa;
            padding: 0.5rem;
            border-radius: 4px;
            font-size: 0.85rem;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s;
        ' class='tooltip-text'>
            {tooltip}
        </span>
    </div>
    """, unsafe_allow_html=True)

def action_button(label, icon, on_click=None, style="primary"):
    """Styled action button"""
    styles = {
        "primary": "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);",
        "success": "background: linear-gradient(135deg, #5ee7df 0%, #b490ca 100%);",
        "danger": "background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);"
    }
    
    button_style = styles.get(style, styles["primary"])
    
    if st.button(label, key=f"btn_{label}"):
        if on_click:
            on_click()