import streamlit as st
from utils.styling import inject_custom_css

st.set_page_config(
    page_title="Portfolio Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

def main():
    inject_custom_css()
    
    # Hero Section
    st.markdown("""
    <div style='text-align: center; padding: 3rem 2rem;'>
        <h1 style='font-size: 3rem; margin-bottom: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Portfolio Intelligence Platform
        </h1>
        <p style='font-size: 1.5rem; color: #808495; max-width: 800px; margin: 0 auto;'>
            Analyze your investment portfolio's risk, find optimization opportunities, and get actionable insights in minutes.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Start
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Get Started with Your Portfolio", type="primary", use_container_width=True):
            st.switch_page("pages/0_Portfolio_Dashboard.py")
        
        st.caption("No signup required • Free to use • Real market data")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # What You Get
    st.markdown("## What You'll Discover")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Optimization
        Find better allocation strategies that maximize returns for your risk level.
        
        **You'll see:**
        - Current vs optimal weights
        - Expected improvement in returns
        - Risk-adjusted performance (Sharpe ratio)
        """)
    
    with col2:
        st.markdown("""
        ### 🔥 Stress Testing
        See how your portfolio would perform in historical crisis scenarios like 2008 or COVID.
        
        **You'll see:**
        - Worst-case loss estimates
        - Resilience score (0-100)
        - Recovery time projections
        """)
    
    with col3:
        st.markdown("""
        ### 💡 Health Report
        Get an overall portfolio health score with prioritized actions to improve.
        
        **You'll see:**
        - Health score (0-100)
        - Top 3 priority fixes
        - Strengths to maintain
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # How It Works
    st.markdown("## How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: rgba(102, 126, 234, 0.05); border-radius: 12px;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>1️⃣</div>
            <h3>Enter Portfolio</h3>
            <p style='color: #808495;'>Load a sample or enter your own stocks and weights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: rgba(102, 126, 234, 0.05); border-radius: 12px;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>2️⃣</div>
            <h3>Run Analysis</h3>
            <p style='color: #808495;'>Click one button to analyze using real market data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: rgba(102, 126, 234, 0.05); border-radius: 12px;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>3️⃣</div>
            <h3>Get Insights</h3>
            <p style='color: #808495;'>Review results and take action on recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Sample Portfolios
    st.markdown("## Try a Sample Portfolio")
    st.caption("Not ready to enter your own? Start with an example:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Tech Growth", use_container_width=True):
            st.switch_page("pages/0_Portfolio_Dashboard.py")
    
    with col2:
        if st.button("⚖️ Balanced Mix", use_container_width=True):
            st.switch_page("pages/0_Portfolio_Dashboard.py")
    
    with col3:
        if st.button("🛡️ Conservative", use_container_width=True):
            st.switch_page("pages/0_Portfolio_Dashboard.py")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Trust Indicators
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔒 Private & Secure**  
        No data stored. Portfolio exists only in your browser session.
        """)
    
    with col2:
        st.markdown("""
        **📊 Real Market Data**  
        Analysis uses actual historical data from Financial Modeling Prep API.
        """)
    
    with col3:
        st.markdown("""
        **🆓 Free to Use**  
        No signup, no credit card, no limits. Just analysis.
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.caption("**Disclaimer:** This tool provides educational analysis only. Not investment advice. Always consult a financial advisor before making investment decisions.")

if __name__ == "__main__":
    main()