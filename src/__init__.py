"""
Voice Impersonation Attack Framework (VIAF)

A comprehensive framework for testing speaker recognition system vulnerabilities
using AI-generated voices from Boson's Higgs Audio v2 model.
"""

__version__ = "0.1.0"
__author__ = "Boson Hackathon Team"
__email__ = "team@boson.ai"

from .core.voice_impersonator import VoiceImpersonator
from .core.attack_strategies import AttackStrategy
from .core.voice_analyzer import VoiceAnalyzer

__all__ = [
    "VoiceImpersonator",
    "AttackStrategy", 
    "VoiceAnalyzer",
]
