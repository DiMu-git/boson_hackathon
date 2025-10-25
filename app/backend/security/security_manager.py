"""
Security manager for attack detection and rate limiting.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional
from ..models.schemas import VoiceProfile, AttackDetectionResult


class SecurityManager:
    """Handles security features and attack detection."""
    
    def __init__(self, db):
        self.db = db
        self.rate_limits = {}  # Simple in-memory rate limiting
        self.blocked_ips = set()
    
    def check_rate_limit(self, ip_address: str, user_id: str = None) -> bool:
        """Check if request is within rate limits."""
        key = f"{ip_address}:{user_id}" if user_id else ip_address
        now = datetime.now()
        
        if key not in self.rate_limits:
            self.rate_limits[key] = []
        
        # Remove old entries (older than 1 hour)
        self.rate_limits[key] = [
            timestamp for timestamp in self.rate_limits[key]
            if now - timestamp < timedelta(hours=1)
        ]
        
        # Check limits (max 10 requests per hour per IP, 5 per user)
        max_requests = 5 if user_id else 10
        if len(self.rate_limits[key]) >= max_requests:
            return False
        
        self.rate_limits[key].append(now)
        return True
    
    def detect_attack(self, audio_analysis: Dict, voice_profile: VoiceProfile) -> AttackDetectionResult:
        """Detect potential voice attacks."""
        attack_indicators = []
        attack_probability = 0.0
        
        # Check for synthetic voice characteristics
        if audio_analysis.get("spectral_centroid", {}).get("std", 0) < 0.1:
            attack_indicators.append("Low spectral variation (possible synthetic voice)")
            attack_probability += 0.3
        
        # Check for unusual pitch patterns
        pitch_analysis = audio_analysis.get("pitch", {})
        if pitch_analysis.get("std_f0", 0) < 10:
            attack_indicators.append("Unnaturally stable pitch (possible synthetic voice)")
            attack_probability += 0.2
        
        # Check for energy patterns
        energy_analysis = audio_analysis.get("energy", {})
        if energy_analysis.get("std", 0) < 0.01:
            attack_indicators.append("Unnaturally consistent energy levels")
            attack_probability += 0.2
        
        # Check for zero crossing rate anomalies
        zcr_analysis = audio_analysis.get("zero_crossing_rate", {})
        if zcr_analysis.get("std", 0) < 0.01:
            attack_indicators.append("Unnaturally consistent zero crossing rate")
            attack_probability += 0.1
        
        is_attack = attack_probability > 0.5
        attack_type = "synthetic_voice" if is_attack else None
        
        recommendations = []
        if is_attack:
            recommendations.extend([
                "Request additional authentication",
                "Flag for manual review",
                "Consider implementing liveness detection"
            ])
        
        return AttackDetectionResult(
            is_attack=is_attack,
            attack_probability=attack_probability,
            attack_type=attack_type,
            security_recommendations=recommendations,
            analysis_details={
                "attack_indicators": attack_indicators,
                "confidence_factors": {
                    "spectral_consistency": audio_analysis.get("spectral_centroid", {}).get("std", 0),
                    "pitch_stability": pitch_analysis.get("std_f0", 0),
                    "energy_consistency": energy_analysis.get("std", 0),
                    "zcr_consistency": zcr_analysis.get("std", 0)
                }
            }
        )
