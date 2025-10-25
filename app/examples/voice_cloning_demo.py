"""
Voice cloning demonstration example.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.voice_generator import VoiceGenerator


def main():
    """
    Voice cloning demonstration.
    """
    print("🎭 Voice Cloning Demonstration")
    print("=" * 50)
    
    # Initialize voice generator
    try:
        generator = VoiceGenerator()
        print("✅ Voice generator initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize voice generator: {e}")
        return
    
    # Check for reference audio
    reference_audio_path = "../../hackathon-msac-public/ref-audio/belinda.wav"
    if not os.path.exists(reference_audio_path):
        print(f"❌ Reference audio not found at {reference_audio_path}")
        print("Please ensure you have the reference audio files.")
        return
    
    print(f"✅ Found reference audio: {reference_audio_path}")
    
    # Voice cloning parameters
    reference_transcript = "Hello, my name is Belinda. I'm here to demonstrate voice cloning capabilities."
    target_text = "Welcome to Boson AI's voice generation system. This is a cloned voice speaking."
    
    print(f"\n📝 Reference transcript: '{reference_transcript}'")
    print(f"🎯 Target text: '{target_text}'")
    
    try:
        print("\n🔄 Generating cloned voice...")
        cloned_audio = generator.generate_impersonation(
            target_voice_path=reference_audio_path,
            text=target_text,
            strategy="direct_cloning",
            temperature=1.0,
            top_p=0.95,
            top_k=50
        )
        
        # Save cloned voice
        output_path = "generated_voices/cloned_voice.wav"
        generator.save_audio(cloned_audio, output_path)
        print(f"✅ Cloned voice saved to {output_path}")
        
        # Generate comparison with original voice
        print("\n🔄 Generating original voice for comparison...")
        original_audio = generator.generate_simple_voice(
            text=target_text,
            voice="belinda",
            temperature=0.7
        )
        
        original_output_path = "generated_voices/original_voice.wav"
        generator.save_audio(original_audio, original_output_path)
        print(f"✅ Original voice saved to {original_output_path}")
        
        print("\n🎉 Voice cloning demonstration completed!")
        print("Compare the two audio files to hear the difference between")
        print("original voice generation and voice cloning.")
        
    except Exception as e:
        print(f"❌ Voice cloning failed: {e}")


if __name__ == "__main__":
    main()
