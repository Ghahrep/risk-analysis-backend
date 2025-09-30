"""
Risk Analysis Platform - Streamlit Dashboard
Main entry point and home page
"""

import streamlit as st
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.api_client import get_risk_api_client, get_behavioral_api_client

# Page configuration
st.set_page_config(
    page_title="Risk Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .feature-box h3 {
        color: white;
        margin-top: 0;
    }
    .feature-box ul {
        margin-bottom: 0;
    }
    .cta-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_api_status():
    """Check if both APIs are running"""
    risk_client = get_risk_api_client()
    behavioral_client = get_behavioral_api_client()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Financial Analysis API")
        health = risk_client.health_check()
        if health:
            st.success("✅ Status: Healthy")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Port", "8001")
            with col_b:
                if 'api_version' in health:
                    st.metric("Version", health['api_version'])
            
            # Show service status
            if 'services' in health:
                services = health['services']
                with st.expander("Service Details"):
                    for service, status in services.items():
                        if status:
                            st.write(f"✓ {service.replace('_', ' ').title()}")
        else:
            st.error("❌ Status: Unavailable")
            st.warning("Make sure minimal_api.py is running on port 8001")
    
    with col2:
        st.subheader("Behavioral Analysis API")
        health = behavioral_client.health_check()
        if health:
            st.success("✅ Status: Healthy")
            st.metric("Port", "8003")
        else:
            st.error("❌ Status: Unavailable")
            st.warning("Make sure behavioral_complete_api.py is running on port 8003")

def main():
    # Header
    st.markdown('<div class="main-header">📊 Risk Analysis Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Institutional-Grade Portfolio Analytics & Insights</div>', unsafe_allow_html=True)
    
    # Quick Actions Banner
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💡 Get Portfolio Insights", use_container_width=True, type="primary"):
            st.switch_page("pages/6_Portfolio_Insights.py")
    with col2:
        if st.button("📈 Optimize Portfolio", use_container_width=True):
            st.switch_page("pages/1_Portfolio_Analysis.py")
    with col3:
        if st.button("⚠️ Analyze Risk", use_container_width=True):
            st.switch_page("pages/2_Risk_Analytics.py")
    
    st.markdown("---")
    
    # API Status Check
    with st.expander("🔌 System Status", expanded=False):
        check_api_status()
    
    st.markdown("---")
    
    # Value Proposition
    st.header("Transform Portfolio Data into Actionable Insights")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Our platform analyzes your portfolio across multiple dimensions to deliver:
        
        - **Clear Health Scores** - Know your portfolio's overall strength (0-100)
        - **Priority Actions** - Ranked recommendations for improvement
        - **Risk Intelligence** - Understand vulnerabilities before they hurt you
        - **Optimization Guidance** - Data-driven rebalancing suggestions
        - **Crisis Protection** - See how correlations break down when markets crash
        
        All powered by real market data and institutional-grade analytics.
        """)
    
    with col2:
        st.info("""
        **New Feature**
        
        📊 Portfolio Insights Dashboard
        
        Get a comprehensive health assessment with actionable recommendations in minutes.
        """)
    
    st.markdown("---")
    
    # Key Capabilities
    st.header("Platform Capabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
        <h3>💡 Portfolio Insights</h3>
        <ul>
            <li>Portfolio health scoring</li>
            <li>Prioritized action items</li>
            <li>Strength identification</li>
            <li>Clear recommendations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
        <h3>📈 Risk Analytics</h3>
        <ul>
            <li>VaR & stress testing</li>
            <li>Volatility forecasting (GARCH)</li>
            <li>Maximum drawdown analysis</li>
            <li>Monte Carlo simulations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
        <h3>🔗 Advanced Correlation</h3>
        <ul>
            <li>Regime-conditional analysis</li>
            <li>Crisis correlation multipliers</li>
            <li>Network topology</li>
            <li>Time-varying relationships</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Platform Performance
    st.header("Technical Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("API Endpoints", "30+", help="Complete API coverage")
    
    with col2:
        st.metric("Response Time", "2.1s", help="Average for complex analytics")
    
    with col3:
        st.metric("Data Coverage", "253 days", help="Historical market data")
    
    with col4:
        st.metric("Success Rate", "100%", help="All endpoints functional")
    
st.markdown("---")
st.header("Quick Start with Sample Portfolios")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Tech Portfolio")
    st.markdown("""
    - AAPL: 30%
    - MSFT: 25%
    - GOOGL: 20%
    - NVDA: 15%
    - TSLA: 10%
    """)
    if st.button("Load Tech Portfolio", width='stretch'):
        from utils.portfolio_manager import set_portfolio
        set_portfolio(
            ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA'],
            [0.30, 0.25, 0.20, 0.15, 0.10]
        )
        st.success("Tech portfolio loaded!")
        st.rerun()

with col2:
    st.subheader("Diversified Portfolio")
    st.markdown("""
    - SPY: 40%
    - AGG: 30%
    - VNQ: 15%
    - GLD: 10%
    - TLT: 5%
    """)
    if st.button("Load Diversified Portfolio", width='stretch'):
        from utils.portfolio_manager import set_portfolio
        set_portfolio(
            ['SPY', 'AGG', 'VNQ', 'GLD', 'TLT'],
            [0.40, 0.30, 0.15, 0.10, 0.05]
        )
        st.success("Diversified portfolio loaded!")
        st.rerun()
    
    st.markdown("---")
    
    # Navigation Guide
    st.header("Navigation Guide")
    
    nav_col1, nav_col2 = st.columns(2)
    
    with nav_col1:
        st.markdown("""
        **Core Analysis Pages:**
        
        1. **Portfolio Insights** 💡 - Start here for executive summary
        2. **Portfolio Analysis** 📈 - Optimization & composition
        3. **Risk Analytics** ⚠️ - VaR, stress testing, volatility
        4. **Correlation Analytics** 🔗 - Diversification analysis
        """)
    
    with nav_col2:
        st.markdown("""
        **Advanced Features:**
        
        5. **Advanced Analytics** 🎯 - Factor analysis & attribution
        6. **Behavioral Analysis** 🧠 - Bias detection & sentiment
        
        Use the sidebar to navigate between pages.
        """)
    
    st.markdown("---")
    
    # Data Source Info
    with st.expander("📊 Data Sources & Methodology"):
        st.markdown("""
        **Market Data:**
        - Real-time pricing via Financial Modeling Prep (FMP) API
        - 250+ days of historical returns data
        - Daily frequency for volatility and correlation calculations
        
        **Analytics Methods:**
        - GARCH(1,1) for volatility forecasting
        - Monte Carlo simulation (1,000+ scenarios)
        - Fama-French factor models
        - Hidden Markov Models for regime detection
        - Hierarchical clustering for correlation structure
        
        **Risk Models:**
        - Historical VaR/CVaR at 95% and 99% confidence
        - Parametric and non-parametric stress testing
        - Maximum drawdown analysis
        - Sharpe ratio optimization
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>Built with FastAPI + Streamlit • Powered by FMP API • Version 4.1.0</strong></p>
        <p><small>Real market data • Institutional analytics • Production-ready backend</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()