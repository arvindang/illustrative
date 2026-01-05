"""Pydantic schemas for API requests and responses."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr


# ==================== Auth Schemas ====================

class UserRegister(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    """User login request."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User response (public info only)."""
    id: str
    email: str
    has_api_key: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"


class ApiKeyRequest(BaseModel):
    """API key storage request."""
    api_key: str


class ApiKeyStatus(BaseModel):
    """API key status response."""
    has_api_key: bool


# ==================== Novel Schemas ====================

class NovelCreate(BaseModel):
    """Create novel request."""
    title: str
    source_filename: Optional[str] = None
    art_style: Optional[str] = None
    narrative_tone: Optional[str] = None
    page_count: Optional[int] = None


class NovelUpdate(BaseModel):
    """Update novel request."""
    status: Optional[str] = None
    page_count: Optional[int] = None
    pdf_storage_key: Optional[str] = None
    epub_storage_key: Optional[str] = None


class NovelResponse(BaseModel):
    """Novel response."""
    id: str
    title: str
    source_filename: Optional[str]
    art_style: Optional[str]
    narrative_tone: Optional[str]
    page_count: Optional[int]
    status: str
    has_pdf: bool
    has_epub: bool
    created_at: Optional[datetime]
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class NovelListResponse(BaseModel):
    """List of novels response."""
    novels: list[NovelResponse]
    total: int


class DownloadUrlResponse(BaseModel):
    """Presigned download URL response."""
    download_url: str
    expires_in: int = 3600


class MessageResponse(BaseModel):
    """Generic message response."""
    message: str
