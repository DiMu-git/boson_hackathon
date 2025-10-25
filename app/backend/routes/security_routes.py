"""
Security and monitoring routes.
"""

from typing import Optional
from fastapi import APIRouter, Depends

from ..security.auth import get_current_user
from ..core.services import get_services

router = APIRouter()


@router.get("/security/events/{user_id}")
async def get_security_events(
    user_id: str,
    limit: int = 50,
    current_user: Optional[str] = Depends(get_current_user)
):
    """Get security events for a user."""
    services = get_services()
    db = services["db"]
    
    events = db.get_security_events(user_id, limit)
    return {"user_id": user_id, "events": events}
