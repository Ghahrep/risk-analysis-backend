"""
Behavioral Analysis Page
Cognitive bias detection, sentiment analysis, and behavioral risk assessment
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
from utils.portfolio_manager import get_portfolio, set_portfolio, normalize_weights, initialize_portfolio

# Initialize portfolio
initialize_portfolio()

sys.path.append(str(Path(__file__).parent.parent))
from utils.api_client import get_behavioral_api_client

st.set_page_config(page_title="Behavioral Analysis", page_icon="🧠", layout="wide")

api_client = get_behavioral_api_client()

def main():
    st.title("🧠 Behavioral Analysis")
    st.markdown("Analyze investment decisions for cognitive biases and behavioral patterns")
    
    # Initialize session state for conversation
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Analysis Configuration")
        
        # Portfolio context (optional)
        st.subheader("Portfolio Context (Optional)")
        symbols_input = st.text_area(
            "Your Holdings",
            value="",
            height=100,
            placeholder="Enter symbols you own\n(one per line)",
            help="Providing your holdings enables market-context analysis"
        )
        
        symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]
        
        if symbols:
            st.success(f"✓ {len(symbols)} holdings detected")
        
        st.markdown("---")
        
        # Analysis type
        st.subheader("Analysis Type")
        analysis_type = st.radio(
            "Select Analysis",
            ["Bias Detection", "Sentiment Analysis", "Risk Profile", "Comprehensive"]
        )
        
        # Clear conversation
        if st.button("🔄 Clear Conversation", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()
    
    # Main content area
    if analysis_type == "Bias Detection":
        st.header("Cognitive Bias Detection")
        st.markdown("Identify potential cognitive biases in your investment thinking")
        
        # Conversation input
        st.subheader("Share Your Investment Thoughts")
        
        user_message = st.text_area(
            "What are you thinking about your investments?",
            height=150,
            placeholder="Example: 'I'm thinking of selling my tech stocks because they've been going down lately, but I'm holding onto my losers hoping they'll bounce back...'",
            help="Share your current investment thoughts, concerns, or decisions"
        )
        
        if st.button("🔍 Analyze for Biases", type="primary"):
            if user_message:
                with st.spinner("Analyzing for cognitive biases..."):
                    # Add to conversation history
                    st.session_state.conversation_history.append({
                        "role": "user",
                        "content": user_message
                    })
                    
                    # Call bias detection API
                    result = api_client.analyze_biases(
                        st.session_state.conversation_history,
                        symbols if symbols else None
                    )
                    
                    if result:
                        st.session_state['bias_result'] = result
                        
                        # Add AI response to history
                        st.session_state.conversation_history.append({
                            "role": "assistant",
                            "content": "Analysis complete"
                        })
            else:
                st.warning("Please enter your investment thoughts to analyze")
        
        # Display results
        if 'bias_result' in st.session_state:
            result = st.session_state['bias_result']
            
            st.markdown("---")
            st.subheader("Detected Biases")
            
            biases = result.get('biases_detected', [])
            
            if biases and len(biases) > 0:
                for bias in biases:
                    with st.expander(f"⚠️ {bias.get('bias_type', 'Unknown Bias')} - Severity: {bias.get('severity', 'Unknown')}", expanded=True):
                        st.write(f"**Description:** {bias.get('description', 'No description available')}")
                        st.write(f"**Evidence:** {bias.get('evidence', 'No evidence provided')}")
                        
                        if bias.get('recommendation'):
                            st.info(f"**Recommendation:** {bias['recommendation']}")
                
                # Risk score
                risk_score = result.get('behavioral_risk_score', 0)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric("Behavioral Risk Score", f"{risk_score}/100")
                
                with col2:
                    if risk_score > 70:
                        st.error("High behavioral risk detected - consider seeking objective second opinion")
                    elif risk_score > 40:
                        st.warning("Moderate behavioral risk - be mindful of identified biases")
                    else:
                        st.success("Low behavioral risk - decision-making appears rational")
            else:
                st.success("✅ No significant biases detected in your analysis")
                st.info("Your investment thinking appears rational and well-balanced")
    
    elif analysis_type == "Sentiment Analysis":
        st.header("Market Sentiment Analysis")
        st.markdown("Analyze your emotional state and market sentiment")
        
        # Sentiment input
        st.subheader("Express Your Market Sentiment")
        
        user_message = st.text_area(
            "How do you feel about the market?",
            height=150,
            placeholder="Example: 'I'm really worried about inflation and rising rates. Everything seems overvalued and I think we're heading for a crash...'",
            help="Share your feelings and concerns about market conditions"
        )
        
        if st.button("📊 Analyze Sentiment", type="primary"):
            if user_message:
                with st.spinner("Analyzing sentiment..."):
                    # Add to conversation
                    st.session_state.conversation_history.append({
                        "role": "user",
                        "content": user_message
                    })
                    
                    # Call sentiment API
                    result = api_client.analyze_sentiment(
                        st.session_state.conversation_history,
                        symbols if symbols else None
                    )
                    
                    if result:
                        st.session_state['sentiment_result'] = result
            else:
                st.warning("Please share your market sentiment to analyze")
        
        # Display results
        if 'sentiment_result' in st.session_state:
            result = st.session_state['sentiment_result']
            
            st.markdown("---")
            st.subheader("Sentiment Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sentiment = result.get('overall_sentiment', 'neutral')
                emoji = {"bullish": "📈", "bearish": "📉", "neutral": "➡️"}.get(sentiment, "➡️")
                st.metric("Overall Sentiment", f"{emoji} {sentiment.capitalize()}")
            
            with col2:
                confidence = result.get('sentiment_confidence', 0)
                st.metric("Confidence", f"{confidence:.0%}")
            
            with col3:
                alignment = result.get('market_alignment', 'unknown')
                st.metric("Market Alignment", alignment.capitalize())
            
            # Emotional indicators
            if 'emotional_indicators' in result:
                st.subheader("Emotional Indicators")
                
                emotions = result['emotional_indicators']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Detected Emotions:**")
                    for emotion in emotions.get('primary_emotions', []):
                        st.write(f"• {emotion}")
                
                with col2:
                    st.write("**Risk Level:**")
                    risk = emotions.get('risk_level', 'unknown')
                    if risk == 'high':
                        st.error(f"⚠️ {risk.upper()} - Strong emotional influence detected")
                    elif risk == 'moderate':
                        st.warning(f"⚡ {risk.upper()} - Some emotional influence")
                    else:
                        st.success(f"✓ {risk.upper()} - Rational sentiment")
            
            # Recommendations
            if 'recommendations' in result:
                st.subheader("Recommendations")
                for rec in result['recommendations']:
                    st.info(f"💡 {rec}")
    
    elif analysis_type == "Risk Profile":
        st.header("Behavioral Risk Profile Assessment")
        st.markdown("Assess your behavioral risk tolerance and investment personality")
        
        st.subheader("Share Your Investment Approach")
        
        user_message = st.text_area(
            "Describe your investment style and risk approach",
            height=150,
            placeholder="Example: 'I usually buy stocks after doing research but sometimes I sell when I see them drop 10%. I like growth stocks but get nervous when the market is volatile...'",
            help="Describe how you make investment decisions and handle risk"
        )
        
        if st.button("🎯 Assess Profile", type="primary"):
            if user_message:
                with st.spinner("Assessing behavioral profile..."):
                    # Add to conversation
                    st.session_state.conversation_history.append({
                        "role": "user",
                        "content": user_message
                    })
                    
                    # Call profile assessment API
                    result = api_client.assess_profile(
                        st.session_state.conversation_history
                    )
                    
                    if result:
                        st.session_state['profile_result'] = result
            else:
                st.warning("Please describe your investment approach")
        
        # Display results
        if 'profile_result' in st.session_state:
            result = st.session_state['profile_result']
            
            st.markdown("---")
            st.subheader("Your Behavioral Risk Profile")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_tolerance = result.get('risk_tolerance', 'moderate')
                st.metric("Risk Tolerance", risk_tolerance.capitalize())
            
            with col2:
                consistency = result.get('consistency_score', 0)
                st.metric("Consistency Score", f"{consistency:.0f}/100")
            
            with col3:
                maturity = result.get('behavioral_maturity', 'developing')
                st.metric("Behavioral Maturity", maturity.capitalize())
            
            # Profile characteristics
            if 'profile_characteristics' in result:
                st.subheader("Profile Characteristics")
                
                chars = result['profile_characteristics']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Strengths:**")
                    for strength in chars.get('strengths', []):
                        st.write(f"✅ {strength}")
                
                with col2:
                    st.write("**Areas for Improvement:**")
                    for area in chars.get('areas_for_improvement', []):
                        st.write(f"⚠️ {area}")
            
            # Development recommendations
            if 'development_recommendations' in result:
                st.subheader("Development Recommendations")
                
                for rec in result['development_recommendations']:
                    st.info(f"📚 {rec}")
    
    elif analysis_type == "Comprehensive":
        st.header("Comprehensive Behavioral Analysis")
        st.markdown("Complete analysis combining bias detection, sentiment, and risk profile")
        
        st.subheader("Comprehensive Investment Discussion")
        
        user_message = st.text_area(
            "Tell us about your current investment situation",
            height=200,
            placeholder="Example: 'I've been holding AAPL and TSLA for 2 years. They're both down and I'm not sure what to do. I feel anxious when I check my portfolio. I'm considering selling to avoid further losses but I also don't want to miss out if they recover...'",
            help="Share comprehensive details about your investments, feelings, and decision-making"
        )
        
        if st.button("🔬 Run Comprehensive Analysis", type="primary"):
            if user_message:
                with st.spinner("Running comprehensive behavioral analysis..."):
                    # Add to conversation
                    st.session_state.conversation_history.append({
                        "role": "user",
                        "content": user_message
                    })
                    
                    # Run all analyses
                    bias_result = api_client.analyze_biases(
                        st.session_state.conversation_history,
                        symbols if symbols else None
                    )
                    
                    sentiment_result = api_client.analyze_sentiment(
                        st.session_state.conversation_history,
                        symbols if symbols else None
                    )
                    
                    profile_result = api_client.assess_profile(
                        st.session_state.conversation_history
                    )
                    
                    st.session_state['comprehensive_result'] = {
                        'bias': bias_result,
                        'sentiment': sentiment_result,
                        'profile': profile_result
                    }
            else:
                st.warning("Please provide details about your investment situation")
        
        # Display comprehensive results
        if 'comprehensive_result' in st.session_state:
            result = st.session_state['comprehensive_result']
            
            st.markdown("---")
            st.success("✅ Comprehensive Analysis Complete")
            
            # Summary metrics
            st.subheader("Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                bias_count = len(result.get('bias', {}).get('biases_detected', []))
                st.metric("Biases Detected", bias_count)
            
            with col2:
                sentiment = result.get('sentiment', {}).get('overall_sentiment', 'neutral')
                st.metric("Sentiment", sentiment.capitalize())
            
            with col3:
                risk_score = result.get('bias', {}).get('behavioral_risk_score', 0)
                st.metric("Risk Score", f"{risk_score}/100")
            
            with col4:
                risk_tolerance = result.get('profile', {}).get('risk_tolerance', 'moderate')
                st.metric("Risk Tolerance", risk_tolerance.capitalize())
            
            # Detailed sections
            tab1, tab2, tab3 = st.tabs(["🎯 Biases", "📊 Sentiment", "👤 Profile"])
            
            with tab1:
                biases = result.get('bias', {}).get('biases_detected', [])
                if biases:
                    for bias in biases:
                        st.warning(f"**{bias.get('bias_type')}** - {bias.get('description')}")
                else:
                    st.success("No significant biases detected")
            
            with tab2:
                sent_result = result.get('sentiment', {})
                st.write(f"**Overall Sentiment:** {sent_result.get('overall_sentiment', 'N/A').capitalize()}")
                st.write(f"**Confidence:** {sent_result.get('sentiment_confidence', 0):.0%}")
                st.write(f"**Market Alignment:** {sent_result.get('market_alignment', 'N/A').capitalize()}")
            
            with tab3:
                prof_result = result.get('profile', {})
                st.write(f"**Risk Tolerance:** {prof_result.get('risk_tolerance', 'N/A').capitalize()}")
                st.write(f"**Consistency Score:** {prof_result.get('consistency_score', 0):.0f}/100")
                st.write(f"**Behavioral Maturity:** {prof_result.get('behavioral_maturity', 'N/A').capitalize()}")
    
    # Conversation history display
    if st.session_state.conversation_history:
        with st.expander("📝 Conversation History"):
            for msg in st.session_state.conversation_history:
                if msg['role'] == 'user':
                    st.write(f"**You:** {msg['content']}")
                else:
                    st.write(f"**Analysis:** {msg['content']}")
    
    # Footer
    st.markdown("---")
    st.caption("""
    💡 **Note:** Behavioral analysis is designed to help identify potential biases and improve decision-making. 
    It should complement, not replace, thorough financial analysis and professional advice.
    """)

if __name__ == "__main__":
    main()