import streamlit as st

def add_feedback_link():
    """Add feedback link to sidebar"""
    st.markdown("---")
    st.markdown("### 📋 Feedback")
    st.caption("Help us improve")
    
    feedback_url = "https://forms.google.com/your-form-id"
    st.markdown(f"[📝 Share Feedback]({feedback_url})")