"""
Boson API testing utilities for the playground module.
"""

import os
import base64
from typing import Optional, Dict, Any, List
import openai


class BosonAPITester:
    """
    Test Boson API functionality and endpoints.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the API tester.
        
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
    
    def test_text_completion(self, model: str = "Qwen3-14B-Hackathon") -> Dict[str, Any]:
        """
        Test text completion endpoint.
        
        Args:
            model: Model to use for testing
            
        Returns:
            Test results
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                max_tokens=50,
                temperature=0.7
            )
            
            return {
                "success": True,
                "model": model,
                "response": response.choices[0].message.content,
                "usage": response.usage
            }
        except Exception as e:
            return {
                "success": False,
                "model": model,
                "error": str(e)
            }
    
    def test_audio_generation(self, voice: str = "belinda") -> Dict[str, Any]:
        """
        Test audio generation endpoint.
        
        Args:
            voice: Voice to use for testing
            
        Returns:
            Test results
        """
        try:
            response = self.client.audio.speech.create(
                model="higgs-audio-generation-Hackathon",
                voice=voice,
                input="Hello, this is a test of the audio generation system.",
                response_format="pcm"
            )
            
            return {
                "success": True,
                "voice": voice,
                "audio_size": len(response.content),
                "content_type": "audio/pcm"
            }
        except Exception as e:
            return {
                "success": False,
                "voice": voice,
                "error": str(e)
            }
    
    def test_audio_understanding(self, audio_path: str) -> Dict[str, Any]:
        """
        Test audio understanding endpoint.
        
        Args:
            audio_path: Path to audio file for testing
            
        Returns:
            Test results
        """
        try:
            # Encode audio to base64
            with open(audio_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode("utf-8")
            
            file_format = audio_path.split(".")[-1]
            
            response = self.client.chat.completions.create(
                model="higgs-audio-understanding-Hackathon",
                messages=[
                    {"role": "system", "content": "Transcribe this audio for me."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_base64,
                                    "format": file_format,
                                },
                            },
                        ],
                    },
                ],
                max_completion_tokens=256,
                temperature=0.0,
            )
            
            return {
                "success": True,
                "transcription": response.choices[0].message.content,
                "usage": response.usage
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_voice_cloning(self, reference_audio_path: str, reference_transcript: str) -> Dict[str, Any]:
        """
        Test voice cloning functionality.
        
        Args:
            reference_audio_path: Path to reference audio
            reference_transcript: Transcript of reference audio
            
        Returns:
            Test results
        """
        try:
            # Encode reference audio to base64
            with open(reference_audio_path, "rb") as f:
                reference_audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            
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
                    {"role": "user", "content": "Welcome to Boson AI's voice generation system."},
                ],
                modalities=["text", "audio"],
                max_completion_tokens=4096,
                temperature=1.0,
                top_p=0.95,
                stream=False,
                stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
                extra_body={"top_k": 50},
            )
            
            # Extract audio data
            audio_b64 = response.choices[0].message.audio.data
            audio_data = base64.b64decode(audio_b64)
            
            return {
                "success": True,
                "audio_size": len(audio_data),
                "content_type": "audio/wav"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_multimodal(self) -> Dict[str, Any]:
        """
        Test multimodal capabilities.
        
        Returns:
            Test results
        """
        try:
            response = self.client.chat.completions.create(
                model="Qwen3-Omni-30B-A3B-Thinking-Hackathon",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Describe what you see and what you hear in one sentence."}
                    ]}
                ],
                max_tokens=256,
                temperature=0.2
            )
            
            return {
                "success": True,
                "response": response.choices[0].message.content,
                "usage": response.usage
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Run comprehensive API tests.
        
        Returns:
            Comprehensive test results
        """
        results = {
            "api_connection": self.test_connection(),
            "text_completion": self.test_text_completion(),
            "audio_generation": self.test_audio_generation(),
            "multimodal": self.test_multimodal()
        }
        
        # Calculate overall success rate
        successful_tests = sum(1 for result in results.values() if result.get("success", False))
        total_tests = len(results)
        results["overall_success_rate"] = successful_tests / total_tests
        
        return results
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test basic API connection.
        
        Returns:
            Connection test results
        """
        try:
            # Simple test to verify API key and connection
            response = self.client.chat.completions.create(
                model="Qwen3-14B-Hackathon",
                messages=[
                    {"role": "user", "content": "Test"}
                ],
                max_tokens=1
            )
            
            return {
                "success": True,
                "message": "API connection successful"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "API connection failed"
            }
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models.
        
        Returns:
            List of available model names
        """
        return [
            "higgs-audio-generation-Hackathon",
            "Qwen3-32B-thinking-Hackathon",
            "Qwen3-32B-non-thinking-Hackathon",
            "Qwen3-14B-Hackathon",
            "higgs-audio-understanding-Hackathon",
            "Qwen3-Omni-30B-A3B-Thinking-Hackathon"
        ]
    
    def get_available_voices(self) -> List[str]:
        """
        Get list of available voices.
        
        Returns:
            List of available voice names
        """
        return [
            "belinda", "broom_salesman", "chadwick", "en_man", "en_woman", 
            "mabel", "vex", "zh_man_sichuan"
        ]
