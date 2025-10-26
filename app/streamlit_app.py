"""
Voice Lock Streamlit App - Redesigned

A user-friendly interface for the Voice Lock API system with improved UX flow.
Provides voice enrollment, verification, and analysis capabilities.
"""

import streamlit as st
import requests
import json
import io
import base64
from datetime import datetime
import pandas as pd
from st_audiorec import st_audiorec
import tempfile
import os

# Configure page
st.set_page_config(
    page_title="Voice Lock System",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if 'user_authenticated' not in st.session_state:
    st.session_state.user_authenticated = False
if 'current_user_id' not in st.session_state:
    st.session_state.current_user_id = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "home"

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

def save_audio_to_temp_file(audio_bytes, file_extension="wav"):
    """Save audio bytes to a temporary file and return the file path."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}")
    temp_file.write(audio_bytes)
    temp_file.close()
    return temp_file.name

def create_audio_file_from_bytes(audio_bytes, filename="recording.wav", content_type="audio/wav"):
    """Create a file-like object from audio bytes for API requests."""
    return (filename, audio_bytes, content_type)

def logout():
    """Logout user and reset session state."""
    st.session_state.user_authenticated = False
    st.session_state.current_user_id = None
    st.session_state.current_page = "home"
    st.rerun()

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
    
    # Show logout button if user is authenticated
    if st.session_state.user_authenticated:
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🚪 Logout", type="secondary"):
                logout()
    
    # Main navigation logic
    if not st.session_state.user_authenticated:
        show_landing_page()
    else:
        show_user_dashboard()

def show_landing_page():
    """Display the landing page with two main options."""
    st.header("Welcome to Voice Lock System")
    st.markdown("Choose an option to get started:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🆕 Enroll a New User")
        st.markdown("""
        Create a new account and enroll your voice for authentication.
        
        **What you'll do:**
        - Create a user account
        - Upload voice samples
        - Generate your voice print
        - Set up security preferences
        """)
        
        if st.button("Start Enrollment", type="primary", use_container_width=True):
            st.session_state.current_page = "enroll"
            st.rerun()
    
    with col2:
        st.subheader("🔐 Authenticate Existing User")
        st.markdown("""
        Already have an account? Verify your identity using your voice.
        
        **What you'll do:**
        - Enter your user ID
        - Record a voice sample
        - Get authenticated
        - Access your dashboard
        """)
        
        if st.button("Start Authentication", type="primary", use_container_width=True):
            st.session_state.current_page = "authenticate"
            st.rerun()
    
    # Show current page content
    if st.session_state.current_page == "enroll":
        show_enrollment_page()
    elif st.session_state.current_page == "authenticate":
        show_authentication_page()

def show_enrollment_page():
    """Display the voice enrollment page."""
    st.header("🆕 User Enrollment")
    st.markdown("Create your account and enroll your voice for authentication.")
    
    # Back button
    if st.button("← Back to Home"):
        st.session_state.current_page = "home"
        st.rerun()
    
    with st.form("enrollment_form"):
        st.subheader("Account Information")
        user_id = st.text_input("User ID", placeholder="Enter unique user identifier", help="This will be your login ID")
        voice_name = st.text_input("Voice Name (Optional)", placeholder="Enter a name for your voice profile")
        
        col1, col2 = st.columns(2)
        with col1:
            security_level = st.selectbox("Security Level", ["low", "medium", "high"], index=1, help="Higher security requires more precise voice matching")
        with col2:
            max_attempts = st.number_input("Max Attempts", min_value=1, max_value=10, value=3, help="Maximum failed attempts before account lockout")
        
        st.subheader("Voice Sample")
        st.markdown("Record your voice directly or upload a file. For best results, speak naturally and clearly.")
        
        # Recording option (primary)
        st.markdown("**🎤 Record Your Voice (Recommended)**")
        st.markdown("*Click the microphone to start recording. Speak clearly for 3-10 seconds.*")
        
        audio_bytes = st_audiorec()
        
        # Show audio playback if recording exists
        if audio_bytes is not None:
            st.audio(audio_bytes, format='audio/wav')
        
        # File upload option (secondary)
        st.markdown("**📁 Or Upload a File**")
        audio_file = st.file_uploader(
            "Upload Voice Sample", 
            type=['wav', 'mp3', 'm4a'],
            help="Upload a clear voice recording (WAV, MP3, or M4A format). Recommended: 3-10 seconds of speech.",
            label_visibility="collapsed"
        )
        
        submitted = st.form_submit_button("Complete Enrollment", type="primary")
        
        if submitted:
            if not user_id:
                st.error("Please enter a User ID")
            elif not audio_bytes and not audio_file:
                st.error("Please record your voice or upload a voice sample")
            else:
                with st.spinner("Processing enrollment..."):
                    # Prepare enrollment data as form data
                    enrollment_data = {
                        "voice_name": voice_name if voice_name else None,
                        "security_level": security_level,
                        "max_attempts": max_attempts
                    }
                    
                    # Prepare files for API - prioritize recording over file upload
                    if audio_bytes:
                        # Use recorded audio
                        files = {"audio_file": create_audio_file_from_bytes(audio_bytes)}
                        st.info("🎤 Using recorded voice sample")
                    else:
                        # Use uploaded file
                        files = {"audio_file": (audio_file.name, audio_file.getvalue(), audio_file.type)}
                        st.info("📁 Using uploaded voice sample")
                    
                    # Make API request with form data
                    response = make_api_request("POST", f"/enroll?user_id={user_id}", 
                                              data=enrollment_data, files=files)
                    
                    if response["success"]:
                        st.success("✅ Voice enrollment successful!")
                        st.balloons()
                        
                        # Show enrollment details
                        result = response["data"]
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("User ID", user_id)
                            st.metric("Security Level", security_level.title())
                        with col2:
                            st.metric("Voice Name", voice_name if voice_name else "Default")
                            st.metric("Max Attempts", max_attempts)
                        
                        st.info("🎉 Your account has been created! You can now authenticate using your voice.")
                        
                        # Auto-login after successful enrollment
                        st.session_state.user_authenticated = True
                        st.session_state.current_user_id = user_id
                        st.session_state.current_page = "dashboard"
                        st.rerun()
                    else:
                        st.error(f"❌ Enrollment failed: {response['error']}")
    

def show_authentication_page():
    """Display the voice authentication page."""
    st.header("🔐 User Authentication")
    st.markdown("Verify your identity using your voice.")
    
    # Back button
    if st.button("← Back to Home"):
        st.session_state.current_page = "home"
        st.rerun()
    
    with st.form("authentication_form"):
        user_id = st.text_input("User ID", placeholder="Enter your user identifier")
        
        st.markdown("Record a voice sample to authenticate. Speak clearly and naturally.")
        
        # Recording option (primary)
        st.markdown("**🎤 Record Your Voice (Recommended)**")
        st.markdown("*Click the microphone to start recording. Speak clearly for 3-10 seconds.*")
        
        audio_bytes = st_audiorec()
        
        # Show audio playback if recording exists
        if audio_bytes is not None:
            st.audio(audio_bytes, format='audio/wav')
        
        # File upload option (secondary)
        st.markdown("**📁 Or Upload a File**")
        audio_file = st.file_uploader(
            "Upload Voice Sample for Authentication", 
            type=['wav', 'mp3', 'm4a'],
            help="Upload a voice recording to verify against your enrolled profile",
            label_visibility="collapsed"
        )
        
        submitted = st.form_submit_button("Authenticate", type="primary")
        
        if submitted:
            if not user_id:
                st.error("Please enter a User ID")
            elif not audio_bytes and not audio_file:
                st.error("Please record your voice or upload a voice sample")
            else:
                with st.spinner("Processing authentication..."):
                    # Prepare files for API - prioritize recording over file upload
                    if audio_bytes:
                        # Use recorded audio
                        files = {"audio_file": create_audio_file_from_bytes(audio_bytes)}
                        st.info("🎤 Using recorded voice sample")
                    else:
                        # Use uploaded file
                        files = {"audio_file": (audio_file.name, audio_file.getvalue(), audio_file.type)}
                        st.info("📁 Using uploaded voice sample")
                    
                    # Make API request with form data
                    response = make_api_request("POST", f"/verify?user_id={user_id}", files=files)
                    
                    if response["success"]:
                        result = response["data"]
                        
                        if result["verified"]:
                            st.success("✅ Authentication successful!")
                            st.balloons()
                            
                            # Show detailed authentication results
                            st.subheader("🔍 Authentication Results")
                            
                            # Main metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Overall Confidence", f"{result['confidence']:.3f}")
                            with col2:
                                st.metric("Status", "✅ Verified")
                            with col3:
                                security = result["security_analysis"]
                                st.metric("Security Level", security['security_level'].title())
                            
                            # Similarity scores breakdown
                            st.subheader("📊 Similarity Analysis")
                            similarity_scores = result.get("similarity_scores", {})
                            
                            if similarity_scores:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Similarity Scores:**")
                                    for key, value in similarity_scores.items():
                                        st.write(f"- {key.replace('_', ' ').title()}: {value:.3f}")
                                
                                with col2:
                                    st.write("**Security Analysis:**")
                                    st.write(f"- Attack Detection: {'❌ No' if not security['attack_detection']['is_attack'] else '⚠️ Yes'}")
                                    st.write(f"- Threshold Used: {security['threshold_used']:.3f}")
                            
                            # Show success message with delay
                            st.success("🎉 Welcome back! Redirecting to dashboard...")
                            
                            # Add a small delay to show the results before redirecting
                            import time
                            time.sleep(2)
                            
                            # Auto-login after successful authentication
                            st.session_state.user_authenticated = True
                            st.session_state.current_user_id = user_id
                            st.session_state.current_page = "dashboard"
                            st.rerun()
                        else:
                            st.error("❌ Authentication failed")
                            st.metric("Confidence Score", f"{result['confidence']:.3f}")
                            
                            # Show security analysis
                            security = result["security_analysis"]
                            if security['attack_detection']['is_attack']:
                                st.warning("⚠️ Potential attack detected!")
                            else:
                                st.info("Voice sample didn't match your enrolled profile. Please try again.")
                    else:
                        st.error(f"❌ Authentication failed: {response['error']}")
    

def show_user_dashboard():
    """Display the user dashboard with personalized features."""
    st.header(f"Welcome back, {st.session_state.current_user_id}!")
    st.markdown("Manage your voice profile and explore advanced features.")
    
    # Navigation tabs
    tab1, tab2 = st.tabs(["🏠 Dashboard", "👤 Profile Management"])
    
    with tab1:
        show_dashboard_home()
    
    with tab2:
        show_profile_management()

def show_dashboard_home():
    """Display the dashboard home content."""
    st.subheader("Your Voice Lock Dashboard")
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("User ID", st.session_state.current_user_id)
    
    with col2:
        st.metric("Status", "🟢 Active")
    
    with col3:
        st.metric("Last Login", datetime.now().strftime("%H:%M"))
    
    # Quick actions
    st.subheader("Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**👤 Profile Management**")
        st.markdown("View and manage your voice profile settings.")
        if st.button("Manage Profile", key="quick_profile"):
            st.session_state.current_tab = "profile"
            st.rerun()
    
    with col2:
        st.markdown("**🔐 Voice Authentication**")
        st.markdown("Test your voice authentication with a new sample.")
        if st.button("Test Authentication", key="quick_auth"):
            st.info("Use the main authentication page to test your voice.")
    
    # Account management
    st.subheader("Account Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**👤 Profile Settings**")
        st.markdown("View and manage your voice profile.")
        if st.button("Manage Profile", key="dashboard_profile"):
            st.session_state.current_tab = "profile"
            st.rerun()
    
    with col2:
        st.markdown("**🗑️ Delete Account**")
        st.markdown("⚠️ Permanently delete your voice profile and account.")
        if st.button("Delete Account", key="dashboard_delete", type="secondary"):
            st.session_state.show_delete_confirmation = True
            st.rerun()
    
    # Recent activity (placeholder)
    st.subheader("Recent Activity")
    st.info("No recent activity to display.")
    
    # Delete confirmation dialog
    if st.session_state.get('show_delete_confirmation', False):
        st.warning("⚠️ **Are you sure you want to delete your account?**")
        st.markdown("This action will:")
        st.markdown("- Permanently delete your voice profile")
        st.markdown("- Remove all authentication data")
        st.markdown("- Log you out of the system")
        
        col_confirm1, col_confirm2 = st.columns(2)
        with col_confirm1:
            if st.button("✅ Yes, Delete Account", type="primary", key="dashboard_confirm_delete"):
                with st.spinner("Deleting account..."):
                    response = make_api_request("DELETE", f"/profiles/{st.session_state.current_user_id}")
                    
                    if response["success"]:
                        st.success("✅ Account deleted successfully!")
                        st.balloons()
                        
                        # Logout user and redirect to landing page
                        st.session_state.user_authenticated = False
                        st.session_state.current_user_id = None
                        st.session_state.current_page = "home"
                        st.session_state.show_delete_confirmation = False
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to delete account: {response['error']}")
                        st.session_state.show_delete_confirmation = False
        
        with col_confirm2:
            if st.button("❌ Cancel", type="secondary", key="dashboard_cancel_delete"):
                st.session_state.show_delete_confirmation = False
                st.rerun()

def show_profile_management():
    """Display the profile management page."""
    st.subheader("👤 Profile Management")
    st.markdown("Manage your voice profile and view profile information.")
    
    # Get current user profile
    with st.spinner("Loading profile..."):
        response = make_api_request("GET", f"/profiles/{st.session_state.current_user_id}")
        
        if response["success"]:
            profile = response["data"]
            
            # Display profile information
            st.subheader("Your Profile Information")
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
            st.error(f"❌ Failed to load profile: {response['error']}")
    
    # Profile actions
    st.subheader("Profile Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔄 Re-enroll Voice**")
        st.markdown("Update your voice sample for better authentication.")
        if st.button("Re-enroll Voice", type="secondary"):
            st.info("Re-enrollment feature coming soon!")
    
    with col2:
        st.markdown("**🗑️ Delete Profile**")
        st.markdown("⚠️ This will permanently delete your voice profile.")
        
        # Confirmation dialog for profile deletion
        if st.button("Delete Profile", type="secondary"):
            st.session_state.show_delete_confirmation = True
        
        if st.session_state.get('show_delete_confirmation', False):
            st.warning("⚠️ **Are you sure you want to delete your profile?**")
            st.markdown("This action will:")
            st.markdown("- Permanently delete your voice profile")
            st.markdown("- Remove all authentication data")
            st.markdown("- Log you out of the system")
            
            col_confirm1, col_confirm2 = st.columns(2)
            with col_confirm1:
                if st.button("✅ Yes, Delete Profile", type="primary"):
                    with st.spinner("Deleting profile..."):
                        response = make_api_request("DELETE", f"/profiles/{st.session_state.current_user_id}")
                        
                        if response["success"]:
                            st.success("✅ Profile deleted successfully!")
                            st.balloons()
                            
                            # Logout user and redirect to landing page
                            st.session_state.user_authenticated = False
                            st.session_state.current_user_id = None
                            st.session_state.current_page = "home"
                            st.session_state.show_delete_confirmation = False
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to delete profile: {response['error']}")
                            st.session_state.show_delete_confirmation = False
            
            with col_confirm2:
                if st.button("❌ Cancel", type="secondary"):
                    st.session_state.show_delete_confirmation = False
                    st.rerun()

if __name__ == "__main__":
    main()