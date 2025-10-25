"""
Authentication utilities for Voice Lock API.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


# Security
security = HTTPBearer(auto_error=False)


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """Simple token-based authentication (implement proper JWT in production)."""
    if not credentials:
        return None
    # In production, implement proper JWT validation
    return credentials.credentials


def require_auth(current_user: Optional[str] = Depends(get_current_user)) -> str:
    """Require authentication for protected endpoints."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return current_user
