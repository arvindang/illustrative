"""GraphicNovel model for tracking user's generated novels."""
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from models.base import Base


class GraphicNovel(Base):
    """Graphic novel record linked to a user."""

    __tablename__ = "graphic_novels"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # Novel metadata
    title = Column(String(255), nullable=False)
    source_filename = Column(String(255), nullable=True)
    art_style = Column(String(100), nullable=True)
    narrative_tone = Column(String(100), nullable=True)
    page_count = Column(Integer, nullable=True)

    # Status: processing, completed, failed
    status = Column(String(50), default="processing", index=True)
    error_message = Column(String(2000), nullable=True)

    # Progress tracking
    current_stage = Column(String(50), nullable=True)  # 'scripting', 'illustrating', 'compositing', 'exporting'
    pages_completed = Column(Integer, default=0)
    pages_total = Column(Integer, nullable=True)
    panels_completed = Column(Integer, default=0)
    panels_total = Column(Integer, nullable=True)

    # Resume capability
    can_resume = Column(Boolean, default=False)
    manifest_path = Column(String(512), nullable=True)

    # Storage keys for S3 bucket
    pdf_storage_key = Column(String(512), nullable=True)
    epub_storage_key = Column(String(512), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationship to user
    user = relationship("User", back_populates="novels")

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "title": self.title,
            "source_filename": self.source_filename,
            "art_style": self.art_style,
            "narrative_tone": self.narrative_tone,
            "page_count": self.page_count,
            "status": self.status,
            "error_message": self.error_message,
            # Progress tracking
            "current_stage": self.current_stage,
            "pages_completed": self.pages_completed,
            "pages_total": self.pages_total,
            "panels_completed": self.panels_completed,
            "panels_total": self.panels_total,
            "can_resume": self.can_resume,
            "manifest_path": self.manifest_path,
            # Storage
            "has_pdf": self.pdf_storage_key is not None,
            "has_epub": self.epub_storage_key is not None,
            "pdf_storage_key": self.pdf_storage_key,
            "epub_storage_key": self.epub_storage_key,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
