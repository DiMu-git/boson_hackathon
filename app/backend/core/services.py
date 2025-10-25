"""
Core services initialization and management.
"""

from pathlib import Path
from typing import Dict, Any

from ..database import VoiceLockDatabase
from ..security import SecurityManager
import sys
from pathlib import Path

# Add the project root to Python path to access src modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.voice_analyzer import VoiceAnalyzer
from src.embedding_scorer import SpeakerEmbedder
from src.voice_generator import VoiceGenerator

# Global services container
_services: Dict[str, Any] = {}


def initialize_services():
    """Initialize all core services."""
    global _services
    
    # Initialize database
    _services["db"] = VoiceLockDatabase()
    
    # Initialize security manager
    _services["security_manager"] = SecurityManager(_services["db"])
    
    # Initialize voice analysis services with error handling
    try:
        _services["voice_analyzer"] = VoiceAnalyzer()
    except ImportError as e:
        if "numpy" in str(e).lower():
            raise ImportError("NumPy is not available. Please install it with: pip install numpy>=1.24.0")
        else:
            raise e
    
    try:
        _services["embedder"] = SpeakerEmbedder(cache_dir=Path("voice_embeddings"))
    except ImportError as e:
        if "numpy" in str(e).lower():
            raise ImportError("NumPy is not available. Please install it with: pip install numpy>=1.24.0")
        else:
            raise e
    
    try:
        _services["voice_generator"] = VoiceGenerator()
    except ImportError as e:
        if "numpy" in str(e).lower():
            raise ImportError("NumPy is not available. Please install it with: pip install numpy>=1.24.0")
        else:
            raise e


def get_services() -> Dict[str, Any]:
    """Get initialized services."""
    if not _services:
        initialize_services()
    return _services
