"""
Voice processing routes (enrollment, verification, analysis).
"""

import os
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, status
from pydantic import BaseModel

from ..models.schemas import (
    VoiceEnrollmentRequest,
    VoiceVerificationRequest,
    VoiceAnalysisRequest,
    VerificationResult
)
from ..security.auth import get_current_user
from ..core.services import get_services

router = APIRouter()


@router.post("/enroll", response_model=Dict[str, Any])
async def enroll_voice(
    user_id: str,
    voice_name: Optional[str] = None,
    security_level: str = "medium",
    max_attempts: int = 3,
    audio_file: UploadFile = File(...),
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    Enroll a new voice profile for authentication.
    
    This endpoint allows users to register their voice for authentication.
    The system analyzes the voice characteristics and stores them securely.
    """
    try:
        services = get_services()
        db = services["db"]
        voice_analyzer = services["voice_analyzer"]
        embedder = services["embedder"]
        
        # Check if user already exists
        existing_profile = db.get_voice_profile(user_id)
        if existing_profile:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Voice profile for user {user_id} already exists"
            )
        
        # Save uploaded audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Analyze voice characteristics
            voice_characteristics = voice_analyzer.analyze_voice(temp_file_path)
            
            # Generate speaker embedding
            embedding = embedder.embed_file(temp_file_path)
            
            # Create voice profile
            from ..models.schemas import VoiceProfile
            profile = VoiceProfile(
                user_id=user_id,
                voice_name=voice_name,
                enrollment_date=datetime.now(),
                security_level=security_level,
                max_attempts=max_attempts,
                is_active=True,
                voice_characteristics=voice_characteristics,
                embedding_data=embedding.tolist(),
                last_verification=None,
                verification_count=0,
                failed_attempts=0
            )
            
            # Save to database
            if not db.create_voice_profile(profile):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create voice profile"
                )
            
            # Log security event
            db.log_security_event(
                user_id=user_id,
                event_type="enrollment",
                severity="info",
                description="Voice profile enrolled successfully",
                metadata={"security_level": security_level}
            )
            
            return {
                "success": True,
                "message": f"Voice profile for user {user_id} enrolled successfully",
                "profile_id": user_id,
                "enrollment_date": profile.enrollment_date.isoformat(),
                "security_level": security_level,
                "voice_characteristics": {
                    "duration": voice_characteristics.get("duration", 0),
                    "pitch_range": voice_characteristics.get("pitch", {}).get("f0_range", 0),
                    "energy_level": voice_characteristics.get("energy", {}).get("mean", 0)
                }
            }
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enrollment failed: {str(e)}"
        )


@router.post("/verify", response_model=VerificationResult)
async def verify_voice(
    user_id: str,
    confidence_threshold: Optional[float] = None,
    audio_file: UploadFile = File(...),
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    Verify a voice against enrolled profile.
    
    This endpoint compares the provided voice sample against the enrolled
    voice profile and returns verification results with confidence scores.
    """
    try:
        services = get_services()
        db = services["db"]
        voice_analyzer = services["voice_analyzer"]
        embedder = services["embedder"]
        security_manager = services["security_manager"]
        
        # Get voice profile
        profile = db.get_voice_profile(user_id)
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice profile for user {user_id} not found"
            )
        
        if not profile.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Voice profile is deactivated"
            )
        
        # Check failed attempts
        if profile.failed_attempts >= profile.max_attempts:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Maximum verification attempts exceeded"
            )
        
        # Save uploaded audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Analyze verification audio
            verification_analysis = voice_analyzer.analyze_voice(temp_file_path)
            
            # Generate embedding for verification audio
            verification_embedding = embedder.embed_file(temp_file_path)
            
            # Compare with enrolled profile
            enrolled_embedding = profile.embedding_data
            
            # Calculate similarity scores
            embedding_similarity = embedder.cosine_similarity(
                verification_embedding, 
                enrolled_embedding
            )
            
            # Calculate voice characteristic similarities
            voice_similarity = voice_analyzer.compare_voices(
                temp_file_path, 
                temp_file_path  # This would be the enrolled voice file in practice
            )
            
            # Determine confidence threshold based on security level
            thresholds = {
                "low": 0.5,
                "medium": 0.75,
                "high": 0.85
            }
            threshold = confidence_threshold or thresholds.get(profile.security_level, 0.75)
            
            # Calculate overall confidence
            overall_confidence = (embedding_similarity * 0.7 + 
                                voice_similarity.get("overall_similarity", 0) * 0.3)
            
            verified = overall_confidence >= threshold
            
            # Security analysis
            attack_detection = security_manager.detect_attack(verification_analysis, profile)
            
            # Log verification attempt
            db.update_verification_log(
                user_id=user_id,
                verified=verified,
                confidence=overall_confidence
            )
            
            # Log security events if needed
            if attack_detection.is_attack:
                db.log_security_event(
                    user_id=user_id,
                    event_type="attack_detected",
                    severity="high",
                    description=f"Potential voice attack detected: {attack_detection.attack_type}",
                    metadata=attack_detection.analysis_details
                )
            
            return VerificationResult(
                verified=verified,
                confidence=overall_confidence,
                similarity_scores={
                    "embedding_similarity": embedding_similarity,
                    "voice_similarity": voice_similarity.get("overall_similarity", 0),
                    "pitch_similarity": voice_similarity.get("pitch_similarity", 0),
                    "spectral_similarity": voice_similarity.get("spectral_similarity", 0)
                },
                security_analysis={
                    "attack_detection": attack_detection.dict(),
                    "security_level": profile.security_level,
                    "threshold_used": threshold
                },
                timestamp=datetime.now(),
                attempt_count=profile.verification_count + 1
            )
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification failed: {str(e)}"
        )


@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_voice(
    analysis_type: str = "full",
    include_attack_detection: bool = True,
    audio_file: UploadFile = File(...),
    current_user: Optional[str] = Depends(get_current_user)
):
    """
    Analyze voice characteristics and detect potential attacks.
    
    This endpoint provides detailed voice analysis including security assessment
    and attack detection capabilities.
    """
    try:
        services = get_services()
        voice_analyzer = services["voice_analyzer"]
        embedder = services["embedder"]
        security_manager = services["security_manager"]
        
        # Save uploaded audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Perform voice analysis
            analysis = voice_analyzer.analyze_voice(temp_file_path)
            
            result = {
                "basic_analysis": {
                    "duration": analysis.get("duration", 0),
                    "sample_rate": voice_analyzer.sample_rate
                },
                "voice_characteristics": {
                    "pitch": analysis.get("pitch", {}),
                    "spectral_centroid": analysis.get("spectral_centroid", {}),
                    "spectral_rolloff": analysis.get("spectral_rolloff", {}),
                    "zero_crossing_rate": analysis.get("zero_crossing_rate", {}),
                    "energy": analysis.get("energy", {}),
                    "formants": analysis.get("formants", {})
                }
            }
            
            if analysis_type in ["full", "security"]:
                # Generate embedding
                embedding = embedder.embed_file(temp_file_path)
                result["embedding_analysis"] = {
                    "embedding_dimension": len(embedding),
                    "embedding_norm": float(embedding.norm()),
                    "embedding_summary": {
                        "mean": float(embedding.mean()),
                        "std": float(embedding.std()),
                        "min": float(embedding.min()),
                        "max": float(embedding.max())
                    }
                }
            
            if include_attack_detection:
                # Create a dummy profile for attack detection
                from ..models.schemas import VoiceProfile
                dummy_profile = VoiceProfile(
                    user_id="dummy",
                    voice_name="dummy",
                    enrollment_date=datetime.now(),
                    security_level="medium",
                    max_attempts=3,
                    is_active=True,
                    voice_characteristics={},
                    embedding_data=[],
                    last_verification=None,
                    verification_count=0,
                    failed_attempts=0
                )
                
                attack_detection = security_manager.detect_attack(analysis, dummy_profile)
                result["security_analysis"] = attack_detection.dict()
            
            return result
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )
