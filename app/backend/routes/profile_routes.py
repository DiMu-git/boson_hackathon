"""
Voice profile management routes.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, status

from ..models.schemas import VoiceProfile
from ..security.auth import get_current_user
from ..core.services import get_services

router = APIRouter()


@router.get("/profiles/{user_id}", response_model=VoiceProfile)
async def get_voice_profile(
    user_id: str,
    current_user: Optional[str] = Depends(get_current_user)
):
    """Get voice profile information."""
    services = get_services()
    db = services["db"]
    
    profile = db.get_voice_profile(user_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice profile for user {user_id} not found"
        )
    return profile


@router.delete("/profiles/{user_id}")
async def delete_voice_profile(
    user_id: str,
    current_user: Optional[str] = Depends(get_current_user)
):
    """Delete voice profile."""
    services = get_services()
    db = services["db"]
    
    profile = db.get_voice_profile(user_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice profile for user {user_id} not found"
        )
    
    # Deactivate the profile
    success = db.deactivate_voice_profile(user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deactivate voice profile"
        )
    
    return {"success": True, "message": f"Voice profile for user {user_id} deactivated"}
