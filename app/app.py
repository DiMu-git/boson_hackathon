"""
Streamlit app for voice generation and manipulation using Boson API.
"""

import os
import streamlit as st
import tempfile
import numpy as np
import av
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from app.voice_generator import VoiceGenerator
from app.audio_utils import AudioUtils

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
            st.write("Click 'START' to begin recording. Click 'STOP' when you're done:")
            
            # Initialize recording state if not exists
            if 'audio_buffer' not in st.session_state:
                st.session_state.audio_buffer = None

            class AudioProcessor:
                def __init__(self):
                    self.audio_buffer = None

                def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
                    # Convert audio frame to numpy array
                    audio_data = frame.to_ndarray()
                    if st.session_state.audio_buffer is None:
                        st.session_state.audio_buffer = audio_data
                    else:
                        st.session_state.audio_buffer = np.concatenate([st.session_state.audio_buffer, audio_data])
                    return frame

            ctx = webrtc_streamer(
                key="voice-recorder",
                mode=WebRtcMode.SENDONLY,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": False, "audio": True},
                audio_processor_factory=AudioProcessor,
            )
            
            # Save button
            if st.button("Save Recording") and st.session_state.audio_buffer is not None:
                # Create recordings directory if it doesn't exist
                recordings_dir = Path("app/recordings")
                recordings_dir.mkdir(exist_ok=True)
                
                # Generate filename with timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                wav_path = recordings_dir / f"recording_{timestamp}.wav"
                
                # Convert numpy array to WAV format and save
                audio_data = st.session_state.audio_buffer.tobytes()
                st.session_state.audio_utils.save_pcm_to_wav(
                    audio_data,
                    str(wav_path),
                    sample_rate=24000
                )
                
                # Store the path in session state
                st.session_state.recorded_audio_path = str(wav_path)
                
                # Show success message and preview
                st.success(f"Audio recorded successfully! Saved to: {wav_path}")
                st.audio(str(wav_path))
                
                # Clear the buffer
                st.session_state.audio_buffer = None
            
            # Clear recording button
            if st.button("Clear Recording"):
                st.session_state.audio_buffer = None
                if hasattr(st.session_state, 'recorded_audio_path'):
                    try:
                        # Don't delete the file, just clear the reference
                        delattr(st.session_state, 'recorded_audio_path')
                    except:
                        pass
                st.success("Recording buffer cleared! Previous recordings are still saved in the recordings directory.")
            
            reference_audio = None if not hasattr(st.session_state, 'recorded_audio_path') else open(st.session_state.recorded_audio_path, 'rb')
            
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
            st.warning("Please upload a reference audio file.")
        elif not reference_text:
            st.warning("Please enter the reference audio transcript.")
        elif not target_text:
            st.warning("Please enter text to generate.")
        else:
            with st.spinner("Cloning voice..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_ref_file:
                        tmp_ref_file.write(reference_audio.read())
                    
                    # Generate cloned voice
                    audio_data = st.session_state.voice_generator.generate_voice_with_cloning(
                        text=target_text,
                        reference_audio_path=tmp_ref_file.name,
                        reference_transcript=reference_text,
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
                    
                    # Cleanup
                    os.unlink(tmp_ref_file.name)
                    
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
