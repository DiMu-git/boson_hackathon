"""
Voice Lock Streamlit App

A user-friendly interface for the Voice Lock API system.
Provides voice enrollment, verification, and analysis capabilities.
"""

import streamlit as st
import requests
import json
import io
import base64
from datetime import datetime
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Voice Lock System",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def make_api_request(method, endpoint, data=None, files=None):
    """Make API request with error handling."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method.upper() == "GET":
            response = requests.get(url, timeout=30)
        elif method.upper() == "POST":
            if files:
                # For multipart requests with files, send data as form data
                response = requests.post(url, data=data, files=files, timeout=30)
            else:
                response = requests.post(url, json=data, timeout=30)
        elif method.upper() == "DELETE":
            response = requests.delete(url, timeout=30)
        
        if response.status_code < 400:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"API Error {response.status_code}: {response.text}"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to API. Make sure the backend is running on localhost:8000"}
    except Exception as e:
        return {"success": False, "error": f"Request failed: {str(e)}"}

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🎤 Voice Lock System")
    st.markdown("Secure voice authentication and analysis platform")
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ Backend API is not running. Please start the backend server first.")
        st.code("cd app && python app.py")
        st.stop()
    
    st.success("✅ Backend API is running")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🎯 Voice Enrollment", "🔐 Voice Verification", "🔍 Voice Analysis", "👤 Profile Management", "🛡️ Security Monitor"]
    )
    
    # Home page
    if page == "🏠 Home":
        show_home_page()
    
    # Voice Enrollment page
    elif page == "🎯 Voice Enrollment":
        show_enrollment_page()
    
    # Voice Verification page
    elif page == "🔐 Voice Verification":
        show_verification_page()
    
    # Voice Analysis page
    elif page == "🔍 Voice Analysis":
        show_analysis_page()
    
    # Profile Management page
    elif page == "👤 Profile Management":
        show_profile_page()
    
    # Security Monitor page
    elif page == "🛡️ Security Monitor":
        show_security_page()

def show_home_page():
    """Display the home page."""
    st.header("Welcome to Voice Lock System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Features")
        st.markdown("""
        - **Voice Enrollment**: Register your voice for authentication
        - **Voice Verification**: Verify your identity using voice
        - **Voice Analysis**: Analyze voice characteristics and detect attacks
        - **Profile Management**: Manage your voice profiles
        - **Security Monitoring**: Monitor security events and threats
        """)
    
    with col2:
        st.subheader("🚀 Quick Start")
        st.markdown("""
        1. **Enroll**: Upload a voice sample to create your profile
        2. **Verify**: Test authentication with a new voice sample
        3. **Analyze**: Get detailed voice analysis and security assessment
        4. **Monitor**: Check security events and system status
        """)
    
    # API Status
    st.subheader("📊 System Status")
    status_response = make_api_request("GET", "/")
    if status_response["success"]:
        status_data = status_response["data"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Service", status_data.get("service", "Unknown"))
        with col2:
            st.metric("Version", status_data.get("version", "Unknown"))
        with col3:
            st.metric("Status", status_data.get("status", "Unknown"))

def show_enrollment_page():
    """Display the voice enrollment page."""
    st.header("🎯 Voice Enrollment")
    st.markdown("Register your voice for authentication. Upload a clear voice sample for best results.")
    
    with st.form("enrollment_form"):
        user_id = st.text_input("User ID", placeholder="Enter unique user identifier")
        voice_name = st.text_input("Voice Name (Optional)", placeholder="Enter a name for your voice profile")
        
        col1, col2 = st.columns(2)
        with col1:
            security_level = st.selectbox("Security Level", ["low", "medium", "high"], index=1)
        with col2:
            max_attempts = st.number_input("Max Attempts", min_value=1, max_value=10, value=3)
        
        audio_file = st.file_uploader(
            "Upload Voice Sample", 
            type=['wav', 'mp3', 'm4a'],
            help="Upload a clear voice recording (WAV, MP3, or M4A format)"
        )
        
        submitted = st.form_submit_button("Enroll Voice", type="primary")
        
        if submitted:
            if not user_id:
                st.error("Please enter a User ID")
            elif not audio_file:
                st.error("Please upload a voice sample")
            else:
                with st.spinner("Processing enrollment..."):
                    # Prepare enrollment data as form data
                    enrollment_data = {
                        "voice_name": voice_name if voice_name else None,
                        "security_level": security_level,
                        "max_attempts": max_attempts
                    }
                    
                    # Prepare files for API
                    files = {"audio_file": (audio_file.name, audio_file.getvalue(), audio_file.type)}
                    
                    # Make API request with form data
                    response = make_api_request("POST", f"/enroll?user_id={user_id}", 
                                              data=enrollment_data, files=files)
                    
                    if response["success"]:
                        st.success("✅ Voice enrollment successful!")
                        st.json(response["data"])
                    else:
                        st.error(f"❌ Enrollment failed: {response['error']}")

def show_verification_page():
    """Display the voice verification page."""
    st.header("🔐 Voice Verification")
    st.markdown("Verify your identity using your voice. Upload a voice sample to authenticate.")
    
    with st.form("verification_form"):
        user_id = st.text_input("User ID", placeholder="Enter your user identifier")
        confidence_threshold = st.slider(
            "Confidence Threshold (Optional)", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.75, 
            step=0.05,
            help="Custom confidence threshold for verification"
        )
        
        audio_file = st.file_uploader(
            "Upload Voice Sample for Verification", 
            type=['wav', 'mp3', 'm4a'],
            help="Upload a voice recording to verify against your enrolled profile"
        )
        
        submitted = st.form_submit_button("Verify Voice", type="primary")
        
        if submitted:
            if not user_id:
                st.error("Please enter a User ID")
            elif not audio_file:
                st.error("Please upload a voice sample")
            else:
                with st.spinner("Processing verification..."):
                    # Prepare verification data as form data
                    verification_data = {
                        "confidence_threshold": confidence_threshold
                    }
                    
                    # Prepare files for API
                    files = {"audio_file": (audio_file.name, audio_file.getvalue(), audio_file.type)}
                    
                    # Make API request with form data
                    response = make_api_request("POST", f"/verify?user_id={user_id}", 
                                              data=verification_data, files=files)
                    
                    if response["success"]:
                        result = response["data"]
                        
                        # Display verification result
                        col1, col2 = st.columns(2)
                        with col1:
                            if result["verified"]:
                                st.success("✅ Voice verification successful!")
                            else:
                                st.error("❌ Voice verification failed")
                        
                        with col2:
                            st.metric("Confidence Score", f"{result['confidence']:.3f}")
                        
                        # Display detailed results
                        st.subheader("📊 Verification Details")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Similarity Scores:**")
                            for key, value in result["similarity_scores"].items():
                                st.write(f"- {key.replace('_', ' ').title()}: {value:.3f}")
                        
                        with col2:
                            st.write("**Security Analysis:**")
                            security = result["security_analysis"]
                            st.write(f"- Attack Detection: {security['attack_detection']['is_attack']}")
                            st.write(f"- Security Level: {security['security_level']}")
                            st.write(f"- Threshold Used: {security['threshold_used']:.3f}")
                        
                        # Show full result
                        with st.expander("View Full Result"):
                            st.json(result)
                    else:
                        st.error(f"❌ Verification failed: {response['error']}")

def show_analysis_page():
    """Display the voice analysis page."""
    st.header("🔍 Voice Analysis")
    st.markdown("Analyze voice characteristics and detect potential security threats.")
    
    with st.form("analysis_form"):
        analysis_type = st.selectbox(
            "Analysis Type", 
            ["basic", "full", "security"], 
            index=1,
            help="Choose the level of analysis to perform"
        )
        include_attack_detection = st.checkbox(
            "Include Attack Detection", 
            value=True,
            help="Include security analysis and attack detection"
        )
        
        audio_file = st.file_uploader(
            "Upload Voice Sample for Analysis", 
            type=['wav', 'mp3', 'm4a'],
            help="Upload a voice recording to analyze"
        )
        
        submitted = st.form_submit_button("Analyze Voice", type="primary")
        
        if submitted:
            if not audio_file:
                st.error("Please upload a voice sample")
            else:
                with st.spinner("Processing analysis..."):
                    # Prepare analysis data as form data
                    analysis_data = {
                        "analysis_type": analysis_type,
                        "include_attack_detection": include_attack_detection
                    }
                    
                    # Prepare files for API
                    files = {"audio_file": (audio_file.name, audio_file.getvalue(), audio_file.type)}
                    
                    # Make API request with form data
                    response = make_api_request("POST", "/analyze", 
                                              data=analysis_data, files=files)
                    
                    if response["success"]:
                        result = response["data"]
                        
                        st.success("✅ Voice analysis completed!")
                        
                        # Display basic analysis
                        if "basic_analysis" in result:
                            st.subheader("📊 Basic Analysis")
                            basic = result["basic_analysis"]
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Duration (seconds)", f"{basic['duration']:.2f}")
                            with col2:
                                st.metric("Sample Rate", f"{basic['sample_rate']} Hz")
                        
                        # Display voice characteristics
                        if "voice_characteristics" in result:
                            st.subheader("🎵 Voice Characteristics")
                            characteristics = result["voice_characteristics"]
                            
                            # Create tabs for different characteristics
                            tab1, tab2, tab3, tab4 = st.tabs(["Pitch", "Spectral", "Energy", "Formants"])
                            
                            with tab1:
                                if "pitch" in characteristics:
                                    pitch = characteristics["pitch"]
                                    st.write(f"**Mean F0:** {pitch.get('mean_f0', 'N/A')}")
                                    st.write(f"**F0 Range:** {pitch.get('f0_range', 'N/A')}")
                                    st.write(f"**F0 Std:** {pitch.get('std_f0', 'N/A')}")
                            
                            with tab2:
                                if "spectral_centroid" in characteristics:
                                    spectral = characteristics["spectral_centroid"]
                                    st.write(f"**Mean:** {spectral.get('mean', 'N/A')}")
                                    st.write(f"**Std:** {spectral.get('std', 'N/A')}")
                            
                            with tab3:
                                if "energy" in characteristics:
                                    energy = characteristics["energy"]
                                    st.write(f"**Mean:** {energy.get('mean', 'N/A')}")
                                    st.write(f"**Std:** {energy.get('std', 'N/A')}")
                            
                            with tab4:
                                if "formants" in characteristics:
                                    formants = characteristics["formants"]
                                    st.write("**Formant Analysis:**")
                                    for key, value in formants.items():
                                        st.write(f"- {key}: {value}")
                        
                        # Display embedding analysis
                        if "embedding_analysis" in result:
                            st.subheader("🧠 Embedding Analysis")
                            embedding = result["embedding_analysis"]
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Dimension", embedding["embedding_dimension"])
                                st.metric("Norm", f"{embedding['embedding_norm']:.3f}")
                            with col2:
                                summary = embedding["embedding_summary"]
                                st.write("**Statistics:**")
                                st.write(f"- Mean: {summary['mean']:.3f}")
                                st.write(f"- Std: {summary['std']:.3f}")
                                st.write(f"- Min: {summary['min']:.3f}")
                                st.write(f"- Max: {summary['max']:.3f}")
                        
                        # Display security analysis
                        if "security_analysis" in result:
                            st.subheader("🛡️ Security Analysis")
                            security = result["security_analysis"]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if security["is_attack"]:
                                    st.error("⚠️ Potential attack detected!")
                                else:
                                    st.success("✅ No attack detected")
                                st.write(f"**Attack Probability:** {security['attack_probability']:.3f}")
                            
                            with col2:
                                if security["attack_type"]:
                                    st.write(f"**Attack Type:** {security['attack_type']}")
                                else:
                                    st.write("**Attack Type:** None")
                            
                            if security["security_recommendations"]:
                                st.write("**Recommendations:**")
                                for rec in security["security_recommendations"]:
                                    st.write(f"- {rec}")
                        
                        # Show full result
                        with st.expander("View Full Analysis Result"):
                            st.json(result)
                    else:
                        st.error(f"❌ Analysis failed: {response['error']}")

def show_profile_page():
    """Display the profile management page."""
    st.header("👤 Profile Management")
    st.markdown("Manage your voice profiles and view profile information.")
    
    tab1, tab2 = st.tabs(["View Profile", "Delete Profile"])
    
    with tab1:
        st.subheader("View Voice Profile")
        with st.form("view_profile_form"):
            user_id = st.text_input("User ID", placeholder="Enter user identifier")
            submitted = st.form_submit_button("Get Profile", type="primary")
            
            if submitted:
                if not user_id:
                    st.error("Please enter a User ID")
                else:
                    with st.spinner("Fetching profile..."):
                        response = make_api_request("GET", f"/profiles/{user_id}")
                        
                        if response["success"]:
                            profile = response["data"]
                            
                            st.success("✅ Profile found!")
                            
                            # Display profile information
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**User ID:** {profile['user_id']}")
                                st.write(f"**Voice Name:** {profile.get('voice_name', 'N/A')}")
                                st.write(f"**Security Level:** {profile['security_level']}")
                                st.write(f"**Max Attempts:** {profile['max_attempts']}")
                            
                            with col2:
                                st.write(f"**Active:** {'Yes' if profile['is_active'] else 'No'}")
                                st.write(f"**Enrollment Date:** {profile['enrollment_date']}")
                                st.write(f"**Verification Count:** {profile['verification_count']}")
                                st.write(f"**Failed Attempts:** {profile['failed_attempts']}")
                            
                            # Voice characteristics
                            if profile.get('voice_characteristics'):
                                st.subheader("🎵 Voice Characteristics")
                                characteristics = profile['voice_characteristics']
                                st.json(characteristics)
                            
                            # Show full profile
                            with st.expander("View Full Profile"):
                                st.json(profile)
                        else:
                            st.error(f"❌ Failed to get profile: {response['error']}")
    
    with tab2:
        st.subheader("Delete Voice Profile")
        st.warning("⚠️ This action will deactivate the voice profile. This cannot be undone.")
        
        with st.form("delete_profile_form"):
            user_id = st.text_input("User ID", placeholder="Enter user identifier to delete")
            confirm = st.checkbox("I understand this action cannot be undone")
            submitted = st.form_submit_button("Delete Profile", type="primary")
            
            if submitted:
                if not user_id:
                    st.error("Please enter a User ID")
                elif not confirm:
                    st.error("Please confirm you understand this action cannot be undone")
                else:
                    with st.spinner("Deleting profile..."):
                        response = make_api_request("DELETE", f"/profiles/{user_id}")
                        
                        if response["success"]:
                            st.success("✅ Profile deleted successfully!")
                            st.json(response["data"])
                        else:
                            st.error(f"❌ Failed to delete profile: {response['error']}")

def show_security_page():
    """Display the security monitoring page."""
    st.header("🛡️ Security Monitor")
    st.markdown("Monitor security events and system status.")
    
    with st.form("security_form"):
        user_id = st.text_input("User ID", placeholder="Enter user identifier")
        limit = st.number_input("Event Limit", min_value=1, max_value=100, value=50)
        submitted = st.form_submit_button("Get Security Events", type="primary")
        
        if submitted:
            if not user_id:
                st.error("Please enter a User ID")
            else:
                with st.spinner("Fetching security events..."):
                    response = make_api_request("GET", f"/security/events/{user_id}?limit={limit}")
                    
                    if response["success"]:
                        result = response["data"]
                        events = result.get("events", [])
                        
                        if events:
                            st.success(f"✅ Found {len(events)} security events")
                            
                            # Create DataFrame for better display
                            df_data = []
                            for event in events:
                                df_data.append({
                                    "Event Type": event["event_type"],
                                    "Severity": event["severity"],
                                    "Description": event["description"],
                                    "Timestamp": event["timestamp"],
                                    "Metadata": json.dumps(event.get("metadata", {}))
                                })
                            
                            df = pd.DataFrame(df_data)
                            
                            # Display events
                            st.dataframe(df, use_container_width=True)
                            
                            # Severity distribution
                            st.subheader("📊 Event Severity Distribution")
                            severity_counts = df["Severity"].value_counts()
                            st.bar_chart(severity_counts)
                            
                            # Event type distribution
                            st.subheader("📈 Event Type Distribution")
                            type_counts = df["Event Type"].value_counts()
                            st.bar_chart(type_counts)
                        else:
                            st.info("ℹ️ No security events found for this user")
                    else:
                        st.error(f"❌ Failed to get security events: {response['error']}")

if __name__ == "__main__":
    main()
