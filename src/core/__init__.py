"""
Core voice impersonation framework components.
"""

from .voice_impersonator import VoiceImpersonator
from .attack_strategies import AttackStrategy, DirectCloningStrategy, CharacteristicManipulationStrategy
from .voice_analyzer import VoiceAnalyzer

__all__ = [
    "VoiceImpersonator",
    "AttackStrategy",
    "DirectCloningStrategy", 
    "CharacteristicManipulationStrategy",
    "VoiceAnalyzer",
]
