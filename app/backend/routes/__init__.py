"""
API routes for Voice Lock API.
"""

from .auth_routes import router as auth_router
from .voice_routes import router as voice_router
from .profile_routes import router as profile_router
from .security_routes import router as security_router

__all__ = ["auth_router", "voice_router", "profile_router", "security_router"]
