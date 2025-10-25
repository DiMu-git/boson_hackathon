"""
Voice characteristic analysis for impersonation attacks.
"""

import librosa
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


class VoiceAnalyzer:
    """
    Analyzes voice characteristics for impersonation attacks.
    
    This class provides methods to extract and analyze various voice
    characteristics that can be used to improve impersonation attacks.
    """
    
    def __init__(self, sample_rate: int = 24000):
        """
        Initialize the voice analyzer.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
    
    def analyze_voice(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze voice characteristics from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary of voice characteristics
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract characteristics
        characteristics = {
            "pitch": self._extract_pitch(y, sr),
            "formants": self._extract_formants(y, sr),
            "mfcc": self._extract_mfcc(y, sr),
            "spectral_centroid": self._extract_spectral_centroid(y, sr),
            "spectral_rolloff": self._extract_spectral_rolloff(y, sr),
            "zero_crossing_rate": self._extract_zero_crossing_rate(y, sr),
            "energy": self._extract_energy(y, sr),
            "duration": len(y) / sr
        }
        
        return characteristics
    
    def _extract_pitch(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract pitch characteristics.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of pitch characteristics
        """
        # Extract fundamental frequency
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        
        # Remove unvoiced frames
        f0_voiced = f0[voiced_flag]
        
        if len(f0_voiced) > 0:
            return {
                "mean_f0": float(np.mean(f0_voiced)),
                "std_f0": float(np.std(f0_voiced)),
                "min_f0": float(np.min(f0_voiced)),
                "max_f0": float(np.max(f0_voiced)),
                "f0_range": float(np.max(f0_voiced) - np.min(f0_voiced))
            }
        else:
            return {
                "mean_f0": 0.0,
                "std_f0": 0.0,
                "min_f0": 0.0,
                "max_f0": 0.0,
                "f0_range": 0.0
            }
    
    def _extract_formants(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract formant characteristics.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of formant characteristics
        """
        # This is a simplified formant extraction
        # In practice, you'd use more sophisticated methods like LPC
        
        # Get spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        return {
            "mean_spectral_centroid": float(np.mean(spectral_centroids)),
            "std_spectral_centroid": float(np.std(spectral_centroids))
        }
    
    def _extract_mfcc(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract MFCC features.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            MFCC features
        """
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs, axis=1)
    
    def _extract_spectral_centroid(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract spectral centroid characteristics.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of spectral centroid characteristics
        """
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        return {
            "mean": float(np.mean(spectral_centroids)),
            "std": float(np.std(spectral_centroids)),
            "min": float(np.min(spectral_centroids)),
            "max": float(np.max(spectral_centroids))
        }
    
    def _extract_spectral_rolloff(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract spectral rolloff characteristics.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of spectral rolloff characteristics
        """
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        return {
            "mean": float(np.mean(spectral_rolloff)),
            "std": float(np.std(spectral_rolloff)),
            "min": float(np.min(spectral_rolloff)),
            "max": float(np.max(spectral_rolloff))
        }
    
    def _extract_zero_crossing_rate(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract zero crossing rate characteristics.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of zero crossing rate characteristics
        """
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        return {
            "mean": float(np.mean(zcr)),
            "std": float(np.std(zcr)),
            "min": float(np.min(zcr)),
            "max": float(np.max(zcr))
        }
    
    def _extract_energy(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract energy characteristics.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of energy characteristics
        """
        rms = librosa.feature.rms(y=y)[0]
        
        return {
            "mean": float(np.mean(rms)),
            "std": float(np.std(rms)),
            "min": float(np.min(rms)),
            "max": float(np.max(rms))
        }
    
    def compare_voices(
        self,
        voice1_path: str,
        voice2_path: str
    ) -> Dict[str, float]:
        """
        Compare two voices and return similarity scores.
        
        Args:
            voice1_path: Path to first voice
            voice2_path: Path to second voice
            
        Returns:
            Dictionary of similarity scores
        """
        # Analyze both voices
        char1 = self.analyze_voice(voice1_path)
        char2 = self.analyze_voice(voice2_path)
        
        # Calculate similarity scores
        similarities = {}
        
        # Pitch similarity
        if char1["pitch"]["mean_f0"] > 0 and char2["pitch"]["mean_f0"] > 0:
            pitch_sim = 1.0 - abs(char1["pitch"]["mean_f0"] - char2["pitch"]["mean_f0"]) / max(
                char1["pitch"]["mean_f0"], char2["pitch"]["mean_f0"]
            )
            similarities["pitch_similarity"] = max(0.0, pitch_sim)
        else:
            similarities["pitch_similarity"] = 0.0
        
        # Spectral centroid similarity
        sc_sim = 1.0 - abs(
            char1["spectral_centroid"]["mean"] - char2["spectral_centroid"]["mean"]
        ) / max(
            char1["spectral_centroid"]["mean"], char2["spectral_centroid"]["mean"]
        )
        similarities["spectral_similarity"] = max(0.0, sc_sim)
        
        # MFCC cosine similarity
        mfcc1 = char1["mfcc"]
        mfcc2 = char2["mfcc"]
        mfcc_sim = np.dot(mfcc1, mfcc2) / (np.linalg.norm(mfcc1) * np.linalg.norm(mfcc2))
        similarities["mfcc_similarity"] = float(mfcc_sim)
        
        # Overall similarity (weighted average)
        similarities["overall_similarity"] = (
            similarities["pitch_similarity"] * 0.4 +
            similarities["spectral_similarity"] * 0.3 +
            similarities["mfcc_similarity"] * 0.3
        )
        
        return similarities
