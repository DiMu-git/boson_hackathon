"""
Basic voice generation example for the playground.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from playground.voice_generator import VoiceGenerator


def main():
    """
    Basic voice generation example.
    """
    print("🎤 Basic Voice Generation Example")
    print("=" * 50)
    
    # Initialize voice generator
    try:
        generator = VoiceGenerator()
        print("✅ Voice generator initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize voice generator: {e}")
        return
    
    # Test API connection
    print("\n🔗 Testing API connection...")
    if not generator.test_api_connection():
        print("❌ API connection failed. Please check your API key.")
        return
    print("✅ API connection successful")
    
    # Generate voice with different voices
    text = "Hello, this is a test of the voice generation system. How does this sound?"
    voices = ["belinda", "en_woman", "en_man"]
    
    print(f"\n🎵 Generating voices for text: '{text}'")
    print("-" * 50)
    
    for voice in voices:
        try:
            print(f"Generating voice: {voice}")
            audio_data = generator.generate_simple_voice(text, voice=voice)
            
            # Save audio
            output_path = f"generated_voices/output_{voice}.wav"
            generator.save_audio(audio_data, output_path)
            print(f"✅ Saved to {output_path}")
            
        except Exception as e:
            print(f"❌ Failed to generate voice '{voice}': {e}")
    
    print("\n🎉 Basic voice generation example completed!")


if __name__ == "__main__":
    main()
