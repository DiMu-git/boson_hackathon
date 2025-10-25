"""
Main voice impersonation engine for generating voices that can fool speaker recognition systems.
"""

import os
import base64
from typing import Optional, Dict, Any, List
from pathlib import Path
import openai
from .attack_strategies import AttackStrategy
from .voice_analyzer import VoiceAnalyzer


class VoiceImpersonator:
    """
    Main engine for voice impersonation attacks using Higgs Audio v2.
    
    This class orchestrates the process of generating voices that can fool
    speaker recognition systems by combining various attack strategies with
    the capabilities of Boson's Higgs Audio v2 model.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the voice impersonator.
        
        Args:
            api_key: Boson API key. If None, will use BOSON_API_KEY environment variable.
            base_url: Boson API base URL. If None, will use default hackathon URL.
        """
        self.api_key = api_key or os.getenv("BOSON_API_KEY")
        self.base_url = base_url or os.getenv("BOSON_BASE_URL", "https://hackathon.boson.ai/v1")
        
        if not self.api_key:
            raise ValueError("Boson API key is required. Set BOSON_API_KEY environment variable.")
        
        self.client = openai.Client(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        self.voice_analyzer = VoiceAnalyzer()
        self.available_voices = [
            "belinda", "broom_salesman", "chadwick", "en_man", "en_woman", 
            "mabel", "vex", "zh_man_sichuan"
        ]
    
    def generate_impersonation(
        self,
        target_voice_path: str,
        text: str,
        strategy: str = "direct_cloning",
        **kwargs
    ) -> bytes:
        """
        Generate an impersonated voice using the specified strategy.
        
        Args:
            target_voice_path: Path to the target voice audio file
            text: Text to generate speech for
            strategy: Attack strategy to use
            **kwargs: Additional parameters for the strategy
            
        Returns:
            Generated audio data as bytes
        """
        if strategy == "direct_cloning":
            return self._direct_cloning_attack(target_voice_path, text, **kwargs)
        elif strategy == "characteristic_manipulation":
            return self._characteristic_manipulation_attack(target_voice_path, text, **kwargs)
        elif strategy == "adversarial_generation":
            return self._adversarial_generation_attack(target_voice_path, text, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _direct_cloning_attack(
        self,
        target_voice_path: str,
        text: str,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50
    ) -> bytes:
        """
        Direct voice cloning attack using reference audio.
        
        Args:
            target_voice_path: Path to reference audio
            text: Text to generate
            temperature: Generation temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated audio data
        """
        # Encode reference audio to base64
        with open(target_voice_path, "rb") as f:
            reference_audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        # Get reference transcript (simplified - in practice, you'd use ASR)
        reference_transcript = "This is a reference audio sample for voice cloning."
        BOSON_API_KEY = os.getenv("BOSON_API_KEY")
        client = openai.Client(
            api_key=BOSON_API_KEY,
            base_url="https://hackathon.boson.ai/v1"
        )
        
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
    
    def _characteristic_manipulation_attack(
        self,
        target_voice_path: str,
        text: str,
        **kwargs
    ) -> bytes:
        """
        Voice characteristic manipulation attack.
        
        Args:
            target_voice_path: Path to target voice
            text: Text to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated audio data
        """
        # Analyze target voice characteristics
        target_characteristics = self.voice_analyzer.analyze_voice(target_voice_path)
        
        # Generate voice with manipulated characteristics
        # This is a simplified implementation - in practice, you'd use more sophisticated
        # voice characteristic manipulation techniques
        
        # For now, use direct cloning as a baseline
        return self._direct_cloning_attack(target_voice_path, text, **kwargs)
    
    def _adversarial_generation_attack(
        self,
        target_voice_path: str,
        text: str,
        **kwargs
    ) -> bytes:
        """
        Adversarial voice generation attack.
        
        Args:
            target_voice_path: Path to target voice
            text: Text to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated audio data
        """
        # This would implement adversarial generation techniques
        # For now, use direct cloning as a baseline
        return self._direct_cloning_attack(target_voice_path, text, **kwargs)
    
    def generate_voice_with_parameters(
        self,
        text: str,
        voice: str = "belinda",
        temperature: float = 0.7,
        response_format: str = "pcm"
    ) -> bytes:
        """
        Generate voice using simple parameters (non-impersonation).
        
        Args:
            text: Text to generate speech for
            voice: Voice to use (from available voices)
            temperature: Generation temperature
            response_format: Audio format (pcm, wav, etc.)
            
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
    
    def save_audio(self, audio_data: bytes, output_path: str, sample_rate: int = 24000):
        """
        Save generated audio data to file.
        
        Args:
            audio_data: Audio data as bytes
            output_path: Output file path
            sample_rate: Audio sample rate
        """
        import wave
        
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
    
    def get_available_voices(self) -> List[str]:
        """
        Get list of available voices.
        
        Returns:
            List of available voice names
        """
        return self.available_voices.copy()
