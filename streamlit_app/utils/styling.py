"""
Custom styling and UI components for Streamlit app
"""

import streamlit as st

def inject_custom_css():
    """Inject custom CSS for enhanced UI/UX across all pages"""
    st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Typography - Headers */
    h1 {
        padding-top: 0rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 {
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-top: 1rem;
        border-top: 2px solid rgba(49, 51, 63, 0.2);
        font-weight: 600;
    }
    
    h3 {
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    /* Enhanced metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 500;
        color: #808495;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: rgba(49, 51, 63, 0.1);
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1rem;
        background-color: rgba(49, 51, 63, 0.05);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        border: 1px solid rgba(49, 51, 63, 0.1);
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(49, 51, 63, 0.1);
    }
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid rgba(49, 51, 63, 0.1);
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .stSuccess {
        background-color: rgba(40, 167, 69, 0.1);
        border-left-color: #28a745;
    }
    
    .stInfo {
        background-color: rgba(23, 162, 184, 0.1);
        border-left-color: #17a2b8;
    }
    
    .stWarning {
        background-color: rgba(255, 193, 7, 0.1);
        border-left-color: #ffc107;
    }
    
    .stError {
        background-color: rgba(220, 53, 69, 0.1);
        border-left-color: #dc3545;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] h2 {
        border-top: none;
        padding-top: 0;
    }
    
    /* Input styling */
    .stTextInput > div > div > input,
    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid rgba(49, 51, 63, 0.1);
        padding: 0.75rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 8px;
    }
    
    /* Slider styling */
    .stSlider {
        padding: 1rem 0;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: #667eea transparent transparent transparent;
    }
    
    /* Chart containers */
    .js-plotly-plot {
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    </style>
    """, unsafe_allow_html=True)

def add_page_header(title, subtitle, icon="📊"):
    """Add styled page header with title and subtitle"""
    st.markdown(f"""
    <div style='margin-bottom: 2rem;'>
        <h1 style='margin-bottom: 0.5rem;'>{icon} {title}</h1>
        <p style='color: #808495; font-size: 1.1rem; margin: 0;'>
            {subtitle}
        </p>
    </div>
    """, unsafe_allow_html=True)

def add_sidebar_branding():
    """Add branding to sidebar"""
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem 0 2rem 0; border-bottom: 2px solid rgba(102, 126, 234, 0.2); margin-bottom: 2rem;'>
        <div style='font-size: 3rem; margin-bottom: 0.5rem;'>📊</div>
        <h2 style='margin: 0; color: #667eea; font-size: 1.5rem;'>Risk Analytics</h2>
        <p style='margin: 0.25rem 0 0 0; color: #808495; font-size: 0.9rem;'>
            Portfolio Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_weight_summary(weights):
    """Display weight summary with visual indicator"""
    total_weight = sum(weights)
    weight_color = "#28a745" if abs(total_weight - 1.0) < 0.01 else "#ffc107"
    st.markdown(f"""
    <div style='background-color: rgba(102, 126, 234, 0.05); padding: 0.75rem; 
                border-radius: 8px; border-left: 4px solid {weight_color};'>
        <strong>Total Weight:</strong> <span style='color: {weight_color}; font-size: 1.2rem; font-weight: 700;'>{total_weight:.2%}</span>
    </div>
    """, unsafe_allow_html=True)
    
    if abs(total_weight - 1.0) > 0.01:
        st.caption("⚡ Weights auto-normalized to 100%")

def show_empty_state(icon="📊", title="Get Started", message="Select a preset or enter symbols to begin"):
    """Display empty state message"""
    st.markdown(f"""
    <div style='text-align: center; padding: 4rem 2rem; 
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%); 
                border-radius: 16px; margin: 2rem 0;'>
        <div style='font-size: 4rem; margin-bottom: 1rem;'>{icon}</div>
        <h2 style='color: #667eea; margin-bottom: 1rem;'>{title}</h2>
        <p style='color: #808495; font-size: 1.1rem; max-width: 600px; margin: 0 auto;'>
            {message}
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_validation_error(error_msg):
    """Display validation error in styled format"""
    st.markdown(f"""
    <div style='background-color: rgba(220, 53, 69, 0.1); padding: 1.5rem; 
                border-radius: 12px; border-left: 4px solid #dc3545; margin: 2rem 0;'>
        <h3 style='color: #dc3545; margin-top: 0;'>⚠️ Validation Error</h3>
        <p style='margin-bottom: 0; font-size: 1.05rem;'>{error_msg}</p>
        <p style='margin-top: 1rem; margin-bottom: 0; color: #808495;'>
            <small>Please correct the issues in the sidebar before running analysis.</small>
        </p>
    </div>
    """, unsafe_allow_html=True)

def add_footer_tip(message):
    """Add footer tip in styled format"""
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
                padding: 1rem; border-radius: 8px; text-align: center; margin-top: 2rem;'>
        <p style='margin: 0; color: #808495;'>
            💡 <strong>Pro Tip:</strong> {message}
        </p>
    </div>
    """, unsafe_allow_html=True)