"""
Navigation helper with current page highlighting
"""

import streamlit as st

PAGES = [
    {
        'name': 'Dashboard',
        'icon': '📊',
        'file': 'Dashboard.py',
        'description': 'Overview & quick actions',
        'category': 'Core'
    },
    {
        'name': 'Portfolio Analysis',
        'icon': '📈',
        'file': '1_Portfolio_Analysis.py',
        'description': 'Optimize allocation',
        'category': 'Core'
    },
    {
        'name': 'Risk Analytics',
        'icon': '⚠️',
        'file': '2_Risk_Analytics.py',
        'description': 'VaR & stress tests',
        'category': 'Core'
    },
    {
        'name': 'Portfolio Insights',
        'icon': '💡',
        'file': '6_Portfolio_Insights.py',
        'description': 'Actionable recommendations',
        'category': 'Core'
    },
    {
        'name': 'Correlation Analysis',
        'icon': '🔗',
        'file': '3_Correlation_Analysis.py',
        'description': 'Diversification analysis',
        'category': 'Advanced'
    },
    {
        'name': 'Advanced Analytics',
        'icon': '📊',
        'file': '4_Advanced_Analytics.py',
        'description': 'Factor & attribution',
        'category': 'Advanced'
    },
    {
        'name': 'Behavioral Analysis',
        'icon': '🧠',
        'file': '5_Behavioral_Analysis.py',
        'description': 'Bias detection',
        'category': 'Advanced'
    }
]

def get_current_page():
    """Detect current page from script runner"""
    try:
        # This is a hack but works in Streamlit
        import inspect
        frame = inspect.currentframe()
        caller_frame = frame.f_back
        caller_file = caller_frame.f_code.co_filename
        
        for page in PAGES:
            if page['file'] in caller_file:
                return page['name']
    except:
        pass
    
    return 'Dashboard'

def render_navigation(current_page=None):
    """
    Render navigation in sidebar with current page highlighted
    
    Args:
        current_page: Name of current page (auto-detected if None)
    """
    if current_page is None:
        current_page = get_current_page()
    
    st.markdown("### 🧭 Navigation")
    
    # Breadcrumb
    st.markdown(f"""
    <div style='background: rgba(102, 126, 234, 0.1); padding: 0.5rem 1rem; 
                border-radius: 8px; margin-bottom: 1rem; font-size: 0.9rem;'>
        Portfolio Intelligence > <strong>{current_page}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Recommended workflow
    with st.expander("📋 Recommended Workflow", expanded=False):
        st.markdown("""
        1. **Dashboard** - Get overview
        2. **Portfolio Analysis** - Optimize weights
        3. **Risk Analytics** - Stress test
        4. **Portfolio Insights** - Get action items
        """)
    
    # Core pages
    st.markdown("#### Core Analysis")
    core_pages = [p for p in PAGES if p['category'] == 'Core']
    
    for page in core_pages:
        is_current = (page['name'] == current_page)
        
        # Visual indicator for current page
        if is_current:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem;'>
                <strong>{page['icon']} {page['name']}</strong><br>
                <small style='opacity: 0.9;'>{page['description']}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button(
                f"{page['icon']} {page['name']}",
                use_container_width=True,
                key=f"nav_{page['name']}",
                help=page['description']
            ):
                if page['name'] == 'Dashboard':
                    st.switch_page("Dashboard.py")
                else:
                    st.switch_page(f"pages/{page['file']}")
            st.caption(page['description'])
    
    # Advanced pages
    st.markdown("#### Advanced Analysis")
    advanced_pages = [p for p in PAGES if p['category'] == 'Advanced']
    
    for page in advanced_pages:
        is_current = (page['name'] == current_page)
        
        if is_current:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem;'>
                <strong>{page['icon']} {page['name']}</strong><br>
                <small style='opacity: 0.9;'>{page['description']}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button(
                f"{page['icon']} {page['name']}",
                use_container_width=True,
                key=f"nav_{page['name']}",
                help=page['description']
            ):
                st.switch_page(f"pages/{page['file']}")
            st.caption(page['description'])