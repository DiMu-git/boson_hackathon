"""
Authentication and basic service routes.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Voice Lock API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "enrollment": "/enroll",
            "verification": "/verify",
            "analysis": "/analyze",
            "profiles": "/profiles",
            "security": "/security"
        }
    }


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    from datetime import datetime
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": "operational",
            "voice_analyzer": "operational",
            "embedder": "operational",
            "security": "operational"
        }
    }
