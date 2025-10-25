"""
Core voice impersonation framework components.
"""

from .voice_generator import VoiceGenerator
from .attack_strategies import AttackStrategy, DirectCloningStrategy, CharacteristicManipulationStrategy
from .voice_analyzer import VoiceAnalyzer

__all__ = [
    "VoiceGenerator",
    "AttackStrategy",
    "DirectCloningStrategy", 
    "CharacteristicManipulationStrategy",
    "VoiceAnalyzer",
]
