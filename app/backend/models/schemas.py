"""
Pydantic schemas for request/response models.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class VoiceEnrollmentRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    voice_name: Optional[str] = Field(None, description="Optional name for the voice profile")
    security_level: str = Field("medium", description="Security level: low, medium, high")
    max_attempts: int = Field(3, description="Maximum verification attempts per session")


class VoiceVerificationRequest(BaseModel):
    user_id: str = Field(..., description="User identifier to verify")
    confidence_threshold: Optional[float] = Field(None, description="Custom confidence threshold")


class VoiceProfile(BaseModel):
    user_id: str
    voice_name: Optional[str]
    enrollment_date: datetime
    security_level: str
    max_attempts: int
    is_active: bool
    voice_characteristics: Dict[str, Any]
    embedding_data: List[float]
    audio_file_path: Optional[str]  # Path to stored audio file
    last_verification: Optional[datetime]
    verification_count: int
    failed_attempts: int


class VerificationResult(BaseModel):
    verified: bool
    confidence: float
    similarity_scores: Dict[str, float]
    security_analysis: Dict[str, Any]
    timestamp: datetime
    attempt_count: int


class AttackDetectionResult(BaseModel):
    is_attack: bool
    attack_probability: float
    attack_type: Optional[str]
    security_recommendations: List[str]
    analysis_details: Dict[str, Any]
