"""
Audio processing utilities for the playground module.
"""

import wave
import numpy as np
import librosa
from typing import Optional, Tuple
from pathlib import Path


class AudioUtils:
    """
    Audio processing utilities for voice generation and analysis.
    """
    
    def __init__(self, sample_rate: int = 24000):
        """
        Initialize audio utilities.
        
        Args:
            sample_rate: Default sample rate for audio processing
        """
        self.sample_rate = sample_rate
    
    def save_pcm_to_wav(
        self,
        pcm_data: bytes,
        output_path: str,
        sample_rate: int = 24000,
        channels: int = 1,
        sample_width: int = 2
    ):
        """
        Save PCM data to WAV file.
        
        Args:
            pcm_data: PCM audio data as bytes
            output_path: Output file path
            sample_rate: Audio sample rate
            channels: Number of audio channels
            sample_width: Sample width in bytes
        """
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
    
    def load_audio(self, file_path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Load audio file using librosa.
        
        Args:
            file_path: Path to audio file
            sr: Sample rate (if None, uses original)
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        return librosa.load(file_path, sr=sr)
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio: Audio data
            
        Returns:
            Normalized audio data
        """
        if np.max(np.abs(audio)) > 0:
            return audio / np.max(np.abs(audio))
        return audio
    
    def trim_silence(
        self,
        audio: np.ndarray,
        sr: int,
        top_db: float = 20.0,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        Trim silence from audio.
        
        Args:
            audio: Audio data
            sr: Sample rate
            top_db: Silence threshold in dB
            frame_length: Frame length for analysis
            hop_length: Hop length for analysis
            
        Returns:
            Trimmed audio data
        """
        return librosa.effects.trim(
            audio,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
    
    def get_audio_duration(self, file_path: str) -> float:
        """
        Get duration of audio file in seconds.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        y, sr = self.load_audio(file_path)
        return len(y) / sr
    
    def convert_sample_rate(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        Convert audio sample rate.
        
        Args:
            audio: Audio data
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Audio data with converted sample rate
        """
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    
    def extract_audio_features(
        self,
        audio: np.ndarray,
        sr: int,
        n_mfcc: int = 13
    ) -> dict:
        """
        Extract basic audio features.
        
        Args:
            audio: Audio data
            sr: Sample rate
            n_mfcc: Number of MFCC coefficients
            
        Returns:
            Dictionary of audio features
        """
        features = {}
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        features['mfcc'] = np.mean(mfccs, axis=1)
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features['spectral_centroid'] = np.mean(spectral_centroids)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features['zero_crossing_rate'] = np.mean(zcr)
        
        # RMS energy
        rms = librosa.feature.rms(y=audio)
        features['rms_energy'] = np.mean(rms)
        
        return features
    
    def compare_audio_files(
        self,
        file1: str,
        file2: str
    ) -> dict:
        """
        Compare two audio files and return similarity metrics.
        
        Args:
            file1: Path to first audio file
            file2: Path to second audio file
            
        Returns:
            Dictionary of similarity metrics
        """
        # Load both audio files
        y1, sr1 = self.load_audio(file1)
        y2, sr2 = self.load_audio(file2)
        
        # Ensure same sample rate
        if sr1 != sr2:
            y2 = self.convert_sample_rate(y2, sr2, sr1)
            sr2 = sr1
        
        # Extract features for both files
        features1 = self.extract_audio_features(y1, sr1)
        features2 = self.extract_audio_features(y2, sr2)
        
        # Calculate similarities
        similarities = {}
        
        # MFCC cosine similarity
        mfcc_sim = np.dot(features1['mfcc'], features2['mfcc']) / (
            np.linalg.norm(features1['mfcc']) * np.linalg.norm(features2['mfcc'])
        )
        similarities['mfcc_similarity'] = float(mfcc_sim)
        
        # Spectral centroid similarity
        sc_sim = 1.0 - abs(features1['spectral_centroid'] - features2['spectral_centroid']) / max(
            features1['spectral_centroid'], features2['spectral_centroid']
        )
        similarities['spectral_similarity'] = max(0.0, sc_sim)
        
        # Overall similarity
        similarities['overall_similarity'] = (
            similarities['mfcc_similarity'] * 0.7 +
            similarities['spectral_similarity'] * 0.3
        )
        
        return similarities
