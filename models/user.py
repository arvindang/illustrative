"""User model with encrypted API key storage."""
import os
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, DateTime, LargeBinary
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from cryptography.fernet import Fernet
import uuid

from models.base import Base


def get_fernet() -> Optional[Fernet]:
    """Get Fernet cipher for encryption/decryption."""
    key = os.getenv("ENCRYPTION_KEY")
    if key:
        return Fernet(key.encode() if isinstance(key, str) else key)
    return None


class User(Base):
    """User model with encrypted Gemini API key storage."""

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    gemini_api_key_encrypted = Column(LargeBinary, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationship to graphic novels
    novels = relationship("GraphicNovel", back_populates="user", cascade="all, delete-orphan")

    def set_gemini_api_key(self, api_key: str) -> None:
        """Encrypt and store the Gemini API key."""
        fernet = get_fernet()
        if fernet is None:
            raise RuntimeError("ENCRYPTION_KEY not configured")
        self.gemini_api_key_encrypted = fernet.encrypt(api_key.encode())

    def get_gemini_api_key(self) -> Optional[str]:
        """Decrypt and return the Gemini API key."""
        if self.gemini_api_key_encrypted is None:
            return None
        fernet = get_fernet()
        if fernet is None:
            raise RuntimeError("ENCRYPTION_KEY not configured")
        return fernet.decrypt(self.gemini_api_key_encrypted).decode()

    def has_api_key(self) -> bool:
        """Check if user has a stored API key."""
        return self.gemini_api_key_encrypted is not None
