"""
Voice generation utilities for testing and experimentation.
"""

import os
import base64
from typing import Optional, List, Dict, Any
import openai
from .audio_utils import AudioUtils


class VoiceGenerator:
    """
    Voice generation utilities for testing Boson API functionality.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the voice generator.
        
        Args:
            api_key: Boson API key
            base_url: Boson API base URL
        """
        self.api_key = api_key or os.getenv("BOSON_API_KEY")
        self.base_url = base_url or os.getenv("BOSON_BASE_URL", "https://hackathon.boson.ai/v1")
        
        if not self.api_key:
            raise ValueError("Boson API key is required. Set BOSON_API_KEY environment variable.")
        
        self.client = openai.Client(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        self.audio_utils = AudioUtils()
        self.available_voices = [
            "belinda", "broom_salesman", "chadwick", "en_man", "en_woman", 
            "mabel", "vex", "zh_man_sichuan"
        ]
    
    def generate_simple_voice(
        self,
        text: str,
        voice: str = "belinda",
        temperature: float = 0.7,
        response_format: str = "pcm"
    ) -> bytes:
        """
        Generate voice using simple parameters.
        
        Args:
            text: Text to generate speech for
            voice: Voice to use
            temperature: Generation temperature
            response_format: Audio format
            
        Returns:
            Generated audio data
        """
        if voice not in self.available_voices:
            raise ValueError(f"Voice '{voice}' not available. Choose from: {self.available_voices}")
        
        response = self.client.audio.speech.create(
            model="higgs-audio-generation-Hackathon",
            voice=voice,
            input=text,
            response_format=response_format
        )
        
        return response.content
    
    def generate_voice_with_cloning(
        self,
        text: str,
        reference_audio_path: str,
        reference_transcript: str,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50
    ) -> bytes:
        """
        Generate voice using reference audio for cloning.
        
        Args:
            text: Text to generate speech for
            reference_audio_path: Path to reference audio
            reference_transcript: Transcript of reference audio
            temperature: Generation temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated audio data
        """
        # Encode reference audio to base64
        with open(reference_audio_path, "rb") as f:
            reference_audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        # Generate voice using chat completions with reference audio
        response = self.client.chat.completions.create(
            model="higgs-audio-generation-Hackathon",
            messages=[
                {"role": "user", "content": reference_transcript},
                {
                    "role": "assistant",
                    "content": [{
                        "type": "input_audio",
                        "input_audio": {
                            "data": reference_audio_b64,
                            "format": "wav"
                        }
                    }],
                },
                {"role": "user", "content": text},
            ],
            modalities=["text", "audio"],
            max_completion_tokens=4096,
            temperature=temperature,
            top_p=top_p,
            stream=False,
            stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
            extra_body={"top_k": top_k},
        )
        
        # Extract and decode audio data
        audio_b64 = response.choices[0].message.audio.data
        return base64.b64decode(audio_b64)
    
    def generate_multiple_voices(
        self,
        text: str,
        voices: List[str],
        **kwargs
    ) -> Dict[str, bytes]:
        """
        Generate voices with multiple voice options.
        
        Args:
            text: Text to generate speech for
            voices: List of voices to use
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping voice names to audio data
        """
        results = {}
        
        for voice in voices:
            if voice in self.available_voices:
                try:
                    audio_data = self.generate_simple_voice(text, voice, **kwargs)
                    results[voice] = audio_data
                except Exception as e:
                    print(f"Error generating voice '{voice}': {e}")
                    results[voice] = None
            else:
                print(f"Voice '{voice}' not available")
                results[voice] = None
        
        return results
    
    def save_audio(self, audio_data: bytes, output_path: str, sample_rate: int = 24000):
        """
        Save generated audio data to file.
        
        Args:
            audio_data: Audio data as bytes
            output_path: Output file path
            sample_rate: Audio sample rate
        """
        self.audio_utils.save_pcm_to_wav(audio_data, output_path, sample_rate)
    
    def get_available_voices(self) -> List[str]:
        """
        Get list of available voices.
        
        Returns:
            List of available voice names
        """
        return self.available_voices.copy()
    
    def test_api_connection(self) -> bool:
        """
        Test connection to Boson API.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try a simple text generation to test connection
            response = self.client.chat.completions.create(
                model="Qwen3-14B-Hackathon",
                messages=[
                    {"role": "user", "content": "Hello, this is a test."}
                ],
                max_tokens=10
            )
            return True
        except Exception as e:
            print(f"API connection test failed: {e}")
            return False
