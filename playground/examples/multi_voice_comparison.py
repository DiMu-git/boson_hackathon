"""
Multi-voice comparison example.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from playground.voice_generator import VoiceGenerator
from playground.audio_utils import AudioUtils


def main():
    """
    Multi-voice comparison demonstration.
    """
    print("🎵 Multi-Voice Comparison Example")
    print("=" * 50)
    
    # Initialize components
    try:
        generator = VoiceGenerator()
        audio_utils = AudioUtils()
        print("✅ Components initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        return
    
    # Test API connection
    if not generator.test_api_connection():
        print("❌ API connection failed. Please check your API key.")
        return
    print("✅ API connection successful")
    
    # Generate voices with multiple options
    text = "This is a comparison of different voices. Listen to the variations in tone and style."
    voices = ["belinda", "en_woman", "en_man", "mabel", "vex"]
    
    print(f"\n🎤 Generating voices for text: '{text}'")
    print("-" * 50)
    
    generated_voices = {}
    
    for voice in voices:
        try:
            print(f"Generating voice: {voice}")
            audio_data = generator.generate_simple_voice(text, voice=voice)
            
            # Save audio
            output_path = f"comparison_{voice}.wav"
            generator.save_audio(audio_data, output_path)
            generated_voices[voice] = output_path
            print(f"✅ Saved to {output_path}")
            
        except Exception as e:
            print(f"❌ Failed to generate voice '{voice}': {e}")
    
    # Compare voices
    if len(generated_voices) >= 2:
        print(f"\n🔍 Comparing generated voices...")
        print("-" * 50)
        
        voice_list = list(generated_voices.items())
        for i in range(len(voice_list)):
            for j in range(i + 1, len(voice_list)):
                voice1_name, voice1_path = voice_list[i]
                voice2_name, voice2_path = voice_list[j]
                
                try:
                    similarities = audio_utils.compare_audio_files(voice1_path, voice2_path)
                    print(f"{voice1_name} vs {voice2_name}:")
                    print(f"  Overall similarity: {similarities['overall_similarity']:.3f}")
                    print(f"  MFCC similarity: {similarities['mfcc_similarity']:.3f}")
                    print(f"  Spectral similarity: {similarities['spectral_similarity']:.3f}")
                    print()
                    
                except Exception as e:
                    print(f"❌ Failed to compare {voice1_name} vs {voice2_name}: {e}")
    
    # Generate ensemble voice (average of all voices)
    print("🎼 Generating ensemble voice...")
    try:
        # This is a simplified ensemble - in practice, you'd use more sophisticated methods
        ensemble_audio = generator.generate_simple_voice(
            text=text,
            voice="belinda",  # Use one voice as ensemble representative
            temperature=1.0
        )
        
        ensemble_output_path = "ensemble_voice.wav"
        generator.save_audio(ensemble_audio, ensemble_output_path)
        print(f"✅ Ensemble voice saved to {ensemble_output_path}")
        
    except Exception as e:
        print(f"❌ Failed to generate ensemble voice: {e}")
    
    print("\n🎉 Multi-voice comparison completed!")
    print("Listen to the generated audio files to compare different voices.")
    print("Each voice should have distinct characteristics and style.")


if __name__ == "__main__":
    main()
