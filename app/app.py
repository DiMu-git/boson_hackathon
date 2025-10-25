"""
Voice Lock API Service - Main Entry Point

This is the main entry point for the Voice Lock API service.
It imports and configures the modular backend.
"""

from backend.app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )