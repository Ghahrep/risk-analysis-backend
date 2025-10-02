"""
Behavioral Analysis Page - Enhanced UX for User Testing
Key Improvements: Clear guidance, example prompts, integrated insights
"""

import streamlit as st
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from utils.portfolio_manager import get_portfolio, set_portfolio, normalize_weights, initialize_portfolio
from utils.api_client import get_behavioral_api_client
from utils.error_handler import safe_api_call
from utils.request_logger import request_logger
from utils.styling import (
    inject_custom_css, 
    add_page_header, 
    add_sidebar_branding,
    show_empty_state,
    add_footer_tip
)

initialize_portfolio()

st.set_page_config(page_title="Behavioral Analysis", page_icon="🧠", layout="wide")

api_client = get_behavioral_api_client()

def main():
    inject_custom_css()
    
    add_page_header(
        "Behavioral Analysis",
        "Identify cognitive biases affecting your investment decisions",
        "🧠"
    )
    
    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Sidebar with goal-based navigation
    with st.sidebar:
        add_sidebar_branding()
        
        st.markdown("### 🎯 What concerns you?")
        st.caption("Common investor questions")
        
        example_prompts = {
            "Am I being too emotional?": "bias",
            "Should I sell my losers?": "bias",
            "Am I chasing winners?": "bias",
            "Is my fear justified?": "sentiment"
        }
        
        for question, analysis in example_prompts.items():
            if st.button(question, use_container_width=True, key=f"prompt_{analysis}_{question[:10]}"):
                st.session_state['selected_prompt'] = question
        
        st.markdown("---")
        
        # Optional portfolio context
        st.markdown("### Portfolio Context")
        st.caption("Optional: helps with context")
        
        symbols, weights = get_portfolio()
        
        if symbols:
            st.success(f"✓ Using {len(symbols)} holdings")
            st.caption(", ".join(symbols[:3]) + ("..." if len(symbols) > 3 else ""))
        else:
            st.info("No portfolio loaded")
        
        st.markdown("---")

        st.markdown("---")
        st.markdown("### 📚 Help & Feedback")
        st.markdown("[Quick Start Guide](https://docs.google.com/document/d/1BX93dy0fOcFdeiiXxT3T7XlDgdC4CRv5Ehp72linzIc/view)")
        st.markdown("[Give Feedback](https://forms.gle/87hpD7gvPVQnsPfc7)")
        
        # Conversation management
        if st.session_state.conversation_history:
            msg_count = len([m for m in st.session_state.conversation_history if m['role'] == 'user'])
            st.markdown(f"**Messages:** {msg_count}")
            
            if st.button("🔄 Start Over", key="clear_conv", use_container_width=True):
                st.session_state.conversation_history = []
                for key in ['bias_result', 'sentiment_result', 'profile_result']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    # Main content - streamlined to bias detection (most useful)
    st.markdown("## Share Your Investment Concerns")
    
    # Show example prompts if no conversation yet
    if not st.session_state.conversation_history:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%); 
                    padding: 2rem; border-radius: 12px; margin-bottom: 2rem;'>
            <h3 style='margin-top: 0; color: #667eea;'>Get objective feedback on your thinking</h3>
            <p style='color: #808495; margin-bottom: 0;'>
                Share your investment thoughts, concerns, or recent decisions. I'll help identify any 
                cognitive biases that might be affecting your judgment.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Example prompts
        st.markdown("### 💭 Example Questions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Recent decisions:**
            - "I sold AAPL after it dropped 15%, was that emotional?"
            - "I'm holding onto my losers hoping they recover"
            - "I bought more after seeing it go up 20%"
            """)
        
        with col2:
            st.markdown("""
            **Current concerns:**
            - "Should I sell everything? Markets feel too high"
            - "I keep checking prices multiple times per day"
            - "I'm afraid to miss out on this AI rally"
            """)
    
    # Conversation input
    user_message = st.text_area(
        "What's on your mind?",
        height=150,
        placeholder="Example: I'm thinking about selling my tech stocks because they're down 20%. I bought them 6 months ago and can't stand watching them drop. Should I cut my losses or wait for them to recover?",
        help="Share specific situations, feelings, or decisions you're facing",
        key="user_input"
    )
    
    # Pre-fill from sidebar prompt if selected
    if 'selected_prompt' in st.session_state:
        user_message = st.session_state['selected_prompt']
        del st.session_state['selected_prompt']
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        analyze_button = st.button("🔍 Analyze for Biases", type="primary", use_container_width=True)
    
    with col2:
        st.caption("Takes 10-15 seconds")
    
    if analyze_button:
        if not user_message or len(user_message.strip()) < 10:
            st.warning("⚠️ Please share at least a sentence or two about your investment thinking")
        else:
            # Add to conversation
            st.session_state.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            with st.spinner("🧠 Analyzing for cognitive biases..."):
                symbols, weights = get_portfolio()
                
                # Validate and clean messages before sending
                valid_messages = []
                for msg in st.session_state.conversation_history:
                    if isinstance(msg, dict) and 'content' in msg:
                        valid_messages.append({
                            'role': msg.get('role', 'user'),
                            'content': str(msg['content'])
                        })
                
                if not valid_messages:
                    st.error("No valid messages to analyze")
                else:
                    result = safe_api_call(
                        lambda: api_client.analyze_biases(
                            valid_messages,
                            symbols if symbols else None
                        ),
                        error_context="bias detection"
                    )

                    if result:
                        st.session_state['bias_result'] = result
                        st.session_state.conversation_history.append({
                            "role": "assistant",
                            "content": "Analysis complete"
                        })
                        st.success("✓ Analysis complete!")
                        time.sleep(0.5)
                        st.rerun()
                    
    # Display results
    if 'bias_result' in st.session_state:
        result = st.session_state['bias_result']
        
        st.markdown("---")
        st.markdown("## Analysis Results")
        
        biases = result.get('biases_detected', [])
        risk_score = result.get('behavioral_risk_score', 0)
        
        # Overall assessment
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if risk_score > 70:
                st.markdown("""
                <div style='background: rgba(220, 53, 69, 0.1); padding: 2rem; border-radius: 12px; text-align: center;'>
                    <div style='font-size: 3rem;'>🔴</div>
                    <h3 style='color: #dc3545; margin: 0.5rem 0;'>{}/100</h3>
                    <p style='color: #808495; margin: 0;'>High Risk</p>
                </div>
                """.format(risk_score), unsafe_allow_html=True)
            elif risk_score > 40:
                st.markdown("""
                <div style='background: rgba(255, 193, 7, 0.1); padding: 2rem; border-radius: 12px; text-align: center;'>
                    <div style='font-size: 3rem;'>🟡</div>
                    <h3 style='color: #ffc107; margin: 0.5rem 0;'>{}/100</h3>
                    <p style='color: #808495; margin: 0;'>Moderate Risk</p>
                </div>
                """.format(risk_score), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: rgba(40, 167, 69, 0.1); padding: 2rem; border-radius: 12px; text-align: center;'>
                    <div style='font-size: 3rem;'>🟢</div>
                    <h3 style='color: #28a745; margin: 0.5rem 0;'>{}/100</h3>
                    <p style='color: #808495; margin: 0;'>Low Risk</p>
                </div>
                """.format(risk_score), unsafe_allow_html=True)
        
        with col2:
            if risk_score > 70:
                st.markdown("""
                <div style='background: rgba(220, 53, 69, 0.05); padding: 1.5rem; border-radius: 12px;'>
                    <h3 style='margin-top: 0;'>High Behavioral Risk Detected</h3>
                    <p>Your investment thinking shows strong signs of cognitive bias. Emotions or mental 
                    shortcuts may be significantly affecting your decisions.</p>
                    <p style='margin-bottom: 0;'><strong>Recommendation:</strong> Consider seeking a second 
                    opinion from a financial advisor or trusted friend before making major moves. Take time 
                    to cool off if you're feeling urgent pressure to act.</p>
                </div>
                """, unsafe_allow_html=True)
            elif risk_score > 40:
                st.markdown("""
                <div style='background: rgba(255, 193, 7, 0.05); padding: 1.5rem; border-radius: 12px;'>
                    <h3 style='margin-top: 0;'>Moderate Behavioral Risk</h3>
                    <p>Some cognitive biases are present in your thinking. This is normal - everyone has biases. 
                    The key is being aware of them.</p>
                    <p style='margin-bottom: 0;'><strong>Recommendation:</strong> Review the specific biases 
                    identified below. Consider writing down your reasoning before making decisions to spot 
                    these patterns.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: rgba(40, 167, 69, 0.05); padding: 1.5rem; border-radius: 12px;'>
                    <h3 style='margin-top: 0;'>Low Behavioral Risk</h3>
                    <p>Your investment thinking appears rational and well-balanced. No significant cognitive 
                    biases detected in your current reasoning.</p>
                    <p style='margin-bottom: 0;'><strong>Keep it up:</strong> Continue making decisions based 
                    on analysis rather than emotion. Stay vigilant for biases, especially during market stress.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Detected biases
        if biases and len(biases) > 0:
            st.markdown("### Identified Biases")
            
            for i, bias in enumerate(biases, 1):
                bias_type = bias.get('bias_type', 'Unknown Bias')
                severity = bias.get('severity', 'Unknown')
                description = bias.get('description', 'No description available')
                evidence = bias.get('evidence', 'No evidence provided')
                recommendation = bias.get('recommendation', '')
                
                severity_colors = {
                    'High': ('#dc3545', '🔴'),
                    'Medium': ('#ffc107', '🟡'),
                    'Low': ('#28a745', '🟢')
                }
                color, emoji = severity_colors.get(severity, ('#808495', '⚪'))
                
                with st.expander(f"{emoji} {bias_type} ({severity} severity)", expanded=(i==1)):
                    st.markdown(f"**What it is:** {description}")
                    st.markdown(f"**Where I see it:** {evidence}")
                    
                    if recommendation:
                        st.markdown(f"""
                        <div style='background: rgba(102, 126, 234, 0.05); padding: 1rem; 
                                    border-radius: 8px; border-left: 3px solid #667eea; margin-top: 1rem;'>
                            <strong>What to do:</strong><br>{recommendation}
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.success("✓ No significant biases detected in your current thinking")
        
        # Action buttons
        st.markdown("---")
        st.markdown("### Next Steps")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💬 Ask Follow-Up", use_container_width=True):
                st.session_state['show_follow_up'] = True
                st.rerun()
        
        with col2:
            if st.button("📊 View Portfolio", use_container_width=True):
                st.switch_page("pages/1_Portfolio_Analysis.py")
        
        with col3:
            if st.button("💡 Get Insights", use_container_width=True):
                st.switch_page("pages/6_Portfolio_Insights.py")
    
    # Follow-up input
    if st.session_state.get('show_follow_up'):
        st.markdown("### Follow-Up Question")
        
        follow_up = st.text_area(
            "Ask more about the biases or your situation",
            placeholder="Example: How can I avoid loss aversion in the future?",
            key="follow_up_input"
        )
        
        if st.button("Ask", type="primary"):
            if follow_up:
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": follow_up
                })
                st.session_state['show_follow_up'] = False
                st.rerun()
    
    # Conversation history
    if len(st.session_state.conversation_history) > 2:
        with st.expander("📝 Conversation History"):
            for msg in st.session_state.conversation_history[:-2]:  # Exclude latest
                if msg['role'] == 'user':
                    st.markdown(f"**You:** {msg['content']}")
    
    # Footer
    st.markdown("---")
    add_footer_tip("💡 Behavioral analysis complements financial analysis. Use it to check your thinking, not replace thorough research.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("An unexpected error occurred in Behavioral Analysis")
        request_logger.logger.exception("Unhandled exception in Behavioral Analysis")
        with st.expander("🔍 Error Details"):
            st.code(str(e))