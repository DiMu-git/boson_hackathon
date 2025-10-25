"""
Database operations for Voice Lock API.
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, Optional
from ..models.schemas import VoiceProfile


class VoiceLockDatabase:
    """SQLite database for voice lock service."""
    
    def __init__(self, db_path: str = "voice_lock.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS voice_profiles (
                    user_id TEXT PRIMARY KEY,
                    voice_name TEXT,
                    enrollment_date TIMESTAMP,
                    security_level TEXT,
                    max_attempts INTEGER,
                    is_active BOOLEAN,
                    voice_characteristics TEXT,
                    embedding_data TEXT,
                    last_verification TIMESTAMP,
                    verification_count INTEGER,
                    failed_attempts INTEGER
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS verification_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    verified BOOLEAN,
                    confidence REAL,
                    timestamp TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT,
                    FOREIGN KEY (user_id) REFERENCES voice_profiles (user_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    event_type TEXT,
                    severity TEXT,
                    description TEXT,
                    timestamp TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (user_id) REFERENCES voice_profiles (user_id)
                )
            """)
    
    def create_voice_profile(self, profile: VoiceProfile) -> bool:
        """Create a new voice profile."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO voice_profiles 
                    (user_id, voice_name, enrollment_date, security_level, max_attempts,
                     is_active, voice_characteristics, embedding_data, last_verification,
                     verification_count, failed_attempts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    profile.user_id,
                    profile.voice_name,
                    profile.enrollment_date,
                    profile.security_level,
                    profile.max_attempts,
                    profile.is_active,
                    json.dumps(profile.voice_characteristics),
                    json.dumps(profile.embedding_data),
                    profile.last_verification,
                    profile.verification_count,
                    profile.failed_attempts
                ))
                return True
        except sqlite3.IntegrityError:
            return False
    
    def get_voice_profile(self, user_id: str) -> Optional[VoiceProfile]:
        """Get voice profile by user ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM voice_profiles WHERE user_id = ?", (user_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return VoiceProfile(
                user_id=row[0],
                voice_name=row[1],
                enrollment_date=datetime.fromisoformat(row[2]),
                security_level=row[3],
                max_attempts=row[4],
                is_active=bool(row[5]),
                voice_characteristics=json.loads(row[6]),
                embedding_data=json.loads(row[7]),
                last_verification=datetime.fromisoformat(row[8]) if row[8] else None,
                verification_count=row[9],
                failed_attempts=row[10]
            )
    
    def update_verification_log(self, user_id: str, verified: bool, confidence: float, 
                               ip_address: str = None, user_agent: str = None):
        """Log verification attempt."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO verification_logs 
                (user_id, verified, confidence, timestamp, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, verified, confidence, datetime.now(), ip_address, user_agent))
            
            # Update profile statistics
            if verified:
                conn.execute("""
                    UPDATE voice_profiles 
                    SET verification_count = verification_count + 1,
                        last_verification = ?,
                        failed_attempts = 0
                    WHERE user_id = ?
                """, (datetime.now(), user_id))
            else:
                conn.execute("""
                    UPDATE voice_profiles 
                    SET failed_attempts = failed_attempts + 1
                    WHERE user_id = ?
                """, (user_id,))
    
    def log_security_event(self, user_id: str, event_type: str, severity: str, 
                          description: str, metadata: Dict = None):
        """Log security event."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO security_events 
                (user_id, event_type, severity, description, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, event_type, severity, description, datetime.now(), 
                  json.dumps(metadata or {})))
    
    def get_security_events(self, user_id: str, limit: int = 50) -> list:
        """Get security events for a user."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT event_type, severity, description, timestamp, metadata
                FROM security_events 
                WHERE user_id = ? 
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, limit))
            
            events = []
            for row in cursor.fetchall():
                events.append({
                    "event_type": row[0],
                    "severity": row[1],
                    "description": row[2],
                    "timestamp": row[3],
                    "metadata": json.loads(row[4]) if row[4] else {}
                })
            
            return events
    
    def deactivate_voice_profile(self, user_id: str) -> bool:
        """Deactivate a voice profile."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE voice_profiles SET is_active = FALSE WHERE user_id = ?",
                (user_id,)
            )
            return cursor.rowcount > 0
