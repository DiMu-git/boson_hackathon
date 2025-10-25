"""
Streamlit app for voice generation and manipulation using Boson API.
"""

import os
import streamlit as st
import tempfile
import numpy as np
import av
from pathlib import Path
from st_audiorec import st_audiorec
from src.voice_generator import VoiceGenerator
from audio_utils import AudioUtils

# Set page config
st.set_page_config(
    page_title="Boson Voice Generator",
    page_icon="🎙️",
    layout="wide"
)

# Initialize session state
if 'voice_generator' not in st.session_state:
    api_key = os.getenv("BOSON_API_KEY")
    base_url = os.getenv("BOSON_BASE_URL", "https://hackathon.boson.ai/v1")
    st.session_state.voice_generator = VoiceGenerator(api_key=api_key, base_url=base_url)

if 'audio_utils' not in st.session_state:
    st.session_state.audio_utils = AudioUtils()

# Title and description
st.title("🎙️ Boson Voice Generator")
st.markdown("""
This app demonstrates the capabilities of the Boson Voice Generation API. You can:
- Generate speech using different pre-defined voices
- Clone voices using reference audio
- Compare different voice generations
""")

# Sidebar for API status
with st.sidebar:
    st.header("API Status")
    if st.session_state.voice_generator.test_api_connection():
        st.success("✅ Connected to Boson API")
    else:
        st.error("❌ Failed to connect to Boson API")
    
    st.header("Available Voices")
    st.write(", ".join(st.session_state.voice_generator.get_available_voices()))

# Main content tabs
tab1, tab2, tab3 = st.tabs(["Basic Voice Generation", "Voice Cloning", "Voice Comparison"])

# Tab 1: Basic Voice Generation
with tab1:
    st.header("Basic Voice Generation")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter text to generate speech",
            value="Hello! This is a test of the Boson voice generation system.",
            height=100
        )
    
    with col2:
        voice = st.selectbox(
            "Select voice",
            options=st.session_state.voice_generator.get_available_voices()
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
    
    if st.button("Generate Voice", key="generate_basic"):
        with st.spinner("Generating voice..."):
            try:
                # Generate audio
                audio_data = st.session_state.voice_generator.generate_simple_voice(
                    text=text_input,
                    voice=voice,
                    temperature=temperature
                )
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    st.session_state.voice_generator.save_audio(
                        audio_data,
                        tmp_file.name
                    )
                    st.audio(tmp_file.name)
                    
                st.success("Voice generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating voice: {str(e)}")

# Tab 2: Voice Cloning
with tab2:
    st.header("Voice Cloning")
    st.info("Either record your voice or upload a reference audio file to clone a voice.")
    
    # Audio source selection
    audio_source = st.radio(
        "Choose audio source",
        ["Record Audio", "Upload Audio"],
        horizontal=True
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if audio_source == "Record Audio":
            st.markdown("**🎤 Record Your Voice (Recommended)**")
            st.markdown("*Click the microphone to start recording. Speak clearly for 3-10 seconds.*")
            
            # Use st_audiorec for simple recording
            audio_bytes = st_audiorec()
            
            # Show audio playback if recording exists
            if audio_bytes is not None:
                st.audio(audio_bytes, format='audio/wav')
                
                # Save the recorded audio to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_bytes)
                    st.session_state.recorded_audio_path = tmp_file.name
                
                st.success("✅ Audio recorded successfully!")
            
            # File upload option (secondary)
            st.markdown("**📁 Or Upload a File**")
            uploaded_audio = st.file_uploader(
                "Upload reference audio (WAV format)", 
                type=['wav'],
                help="Upload a reference audio file for voice cloning",
                label_visibility="collapsed"
            )
            
            # Determine which audio source to use
            if audio_bytes is not None:
                reference_audio = open(st.session_state.recorded_audio_path, 'rb')
            elif uploaded_audio is not None:
                reference_audio = uploaded_audio
            else:
                reference_audio = None
            
        else:  # Upload Audio
            reference_audio = st.file_uploader(
                "Upload reference audio (WAV format)",
                type=['wav']
            )
        
        reference_text = st.text_area(
            "Enter reference audio transcript",
            height=100
        )
        
        target_text = st.text_area(
            "Enter text to generate with cloned voice",
            height=100
        )
    
    with col2:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            key="clone_temp"
        )
        
        top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.05
        )
        
        top_k = st.slider(
            "Top K",
            min_value=1,
            max_value=100,
            value=50
        )
    
    if st.button("Clone Voice", key="generate_clone"):
        if reference_audio is None:
            st.warning("Please record your voice or upload a reference audio file.")
        elif not reference_text:
            st.warning("Please enter the reference audio transcript.")
        elif not target_text:
            st.warning("Please enter text to generate.")
        else:
            with st.spinner("Cloning voice..."):
                try:
                    # Handle different audio sources
                    if hasattr(reference_audio, 'read'):  # File upload
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_ref_file:
                            tmp_ref_file.write(reference_audio.read())
                            ref_path = tmp_ref_file.name
                    else:  # Recorded audio (already saved to temp file)
                        ref_path = st.session_state.recorded_audio_path
                    
                    # Generate cloned voice
                    audio_data = st.session_state.voice_generator.generate_impersonation(
                        target_voice_path=ref_path,
                        text=target_text,
                        strategy="direct_cloning",
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k
                    )
                    
                    # Save generated audio
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_out_file:
                        st.session_state.voice_generator.save_audio(
                            audio_data,
                            tmp_out_file.name
                        )
                        st.audio(tmp_out_file.name)
                    
                    st.success("Voice cloned successfully!")
                    
                    # Cleanup uploaded file if it was temporary
                    if hasattr(reference_audio, 'read'):
                        os.unlink(ref_path)
                    
                except Exception as e:
                    st.error(f"Error cloning voice: {str(e)}")

# Tab 3: Voice Comparison
with tab3:
    st.header("Voice Comparison")
    st.info("Generate the same text with different voices to compare them.")
    
    text_input = st.text_area(
        "Enter text to generate with multiple voices",
        value="Hello! This is a test of the Boson voice generation system.",
        height=100,
        key="compare_text"
    )
    
    voices = st.multiselect(
        "Select voices to compare",
        options=st.session_state.voice_generator.get_available_voices(),
        default=["belinda", "en_man", "en_woman"]
    )
    
    if st.button("Generate and Compare", key="generate_compare"):
        if len(voices) < 2:
            st.warning("Please select at least 2 voices to compare.")
        else:
            with st.spinner("Generating voices..."):
                try:
                    # Generate audio for all selected voices
                    results = st.session_state.voice_generator.generate_multiple_voices(
                        text=text_input,
                        voices=voices
                    )
                    
                    # Display results
                    for voice, audio_data in results.items():
                        if audio_data is not None:
                            st.subheader(f"Voice: {voice}")
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                                st.session_state.voice_generator.save_audio(
                                    audio_data,
                                    tmp_file.name
                                )
                                st.audio(tmp_file.name)
                        else:
                            st.error(f"Failed to generate audio for voice: {voice}")
                    
                    st.success("Voice comparison completed!")
                    
                except Exception as e:
                    st.error(f"Error generating voices: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Boson API and Streamlit")
