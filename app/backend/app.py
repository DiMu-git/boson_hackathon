"""
Voice Lock API Service

A comprehensive voice authentication service built on top of the voice analysis
and generation framework. Provides secure voice enrollment, verification, and
advanced security features.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .core import initialize_services
from .routes import auth_router, voice_router, profile_router, security_router

# Initialize services
initialize_services()

# Create FastAPI application
app = FastAPI(
    title="Voice Lock API",
    description="Secure voice authentication service with advanced security features",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(voice_router)
app.include_router(profile_router)
app.include_router(security_router)


if __name__ == "__main__":
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
