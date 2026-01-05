"""Graphic novel CRUD routes."""
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.schemas import (
    NovelCreate,
    NovelUpdate,
    NovelResponse,
    NovelListResponse,
    DownloadUrlResponse,
    MessageResponse,
)
from api.dependencies import get_db, get_current_user
from models.user import User
from models.novel import GraphicNovel
from storage.bucket import get_storage

router = APIRouter(prefix="/novels", tags=["novels"])


def novel_to_response(novel: GraphicNovel) -> NovelResponse:
    """Convert novel model to response schema."""
    return NovelResponse(
        id=str(novel.id),
        title=novel.title,
        source_filename=novel.source_filename,
        art_style=novel.art_style,
        narrative_tone=novel.narrative_tone,
        page_count=novel.page_count,
        status=novel.status,
        has_pdf=novel.pdf_storage_key is not None,
        has_epub=novel.epub_storage_key is not None,
        created_at=novel.created_at,
        completed_at=novel.completed_at,
    )


@router.get("/", response_model=NovelListResponse)
async def list_novels(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    status_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """List user's graphic novels."""
    query = db.query(GraphicNovel).filter(GraphicNovel.user_id == current_user.id)

    if status_filter:
        query = query.filter(GraphicNovel.status == status_filter)

    total = query.count()
    novels = query.order_by(GraphicNovel.created_at.desc()).offset(offset).limit(limit).all()

    return NovelListResponse(
        novels=[novel_to_response(n) for n in novels],
        total=total,
    )


@router.post("/", response_model=NovelResponse, status_code=status.HTTP_201_CREATED)
async def create_novel(
    request: NovelCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new graphic novel record."""
    novel = GraphicNovel(
        user_id=current_user.id,
        title=request.title,
        source_filename=request.source_filename,
        art_style=request.art_style,
        narrative_tone=request.narrative_tone,
        page_count=request.page_count,
        status="processing",
    )
    db.add(novel)
    db.commit()
    db.refresh(novel)

    return novel_to_response(novel)


@router.get("/{novel_id}", response_model=NovelResponse)
async def get_novel(
    novel_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a specific graphic novel."""
    novel = db.query(GraphicNovel).filter(
        GraphicNovel.id == novel_id,
        GraphicNovel.user_id == current_user.id,
    ).first()

    if not novel:
        raise HTTPException(status_code=404, detail="Novel not found")

    return novel_to_response(novel)


@router.patch("/{novel_id}", response_model=NovelResponse)
async def update_novel(
    novel_id: str,
    request: NovelUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update a graphic novel."""
    novel = db.query(GraphicNovel).filter(
        GraphicNovel.id == novel_id,
        GraphicNovel.user_id == current_user.id,
    ).first()

    if not novel:
        raise HTTPException(status_code=404, detail="Novel not found")

    # Update fields if provided
    if request.status is not None:
        novel.status = request.status
        if request.status == "completed":
            novel.completed_at = datetime.utcnow()
    if request.page_count is not None:
        novel.page_count = request.page_count
    if request.pdf_storage_key is not None:
        novel.pdf_storage_key = request.pdf_storage_key
    if request.epub_storage_key is not None:
        novel.epub_storage_key = request.epub_storage_key

    db.commit()
    db.refresh(novel)

    return novel_to_response(novel)


@router.delete("/{novel_id}", response_model=MessageResponse)
async def delete_novel(
    novel_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a graphic novel and its files from storage."""
    novel = db.query(GraphicNovel).filter(
        GraphicNovel.id == novel_id,
        GraphicNovel.user_id == current_user.id,
    ).first()

    if not novel:
        raise HTTPException(status_code=404, detail="Novel not found")

    # Delete files from bucket if they exist
    storage = get_storage()
    if storage.is_configured():
        keys_to_delete = []
        if novel.pdf_storage_key:
            keys_to_delete.append(novel.pdf_storage_key)
        if novel.epub_storage_key:
            keys_to_delete.append(novel.epub_storage_key)
        if keys_to_delete:
            try:
                storage.delete_files(keys_to_delete)
            except Exception:
                pass  # Continue even if bucket deletion fails

    # Delete from database
    db.delete(novel)
    db.commit()

    return MessageResponse(message="Novel deleted successfully")


@router.get("/{novel_id}/download/{format}", response_model=DownloadUrlResponse)
async def get_download_url(
    novel_id: str,
    format: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a presigned download URL for PDF or EPUB."""
    if format not in ("pdf", "epub"):
        raise HTTPException(status_code=400, detail="Format must be 'pdf' or 'epub'")

    novel = db.query(GraphicNovel).filter(
        GraphicNovel.id == novel_id,
        GraphicNovel.user_id == current_user.id,
    ).first()

    if not novel:
        raise HTTPException(status_code=404, detail="Novel not found")

    storage_key = novel.pdf_storage_key if format == "pdf" else novel.epub_storage_key
    if not storage_key:
        raise HTTPException(status_code=404, detail=f"{format.upper()} file not available")

    storage = get_storage()
    if not storage.is_configured():
        raise HTTPException(status_code=503, detail="Storage not configured")

    download_url = storage.generate_presigned_url(storage_key, expiration=3600)

    return DownloadUrlResponse(download_url=download_url, expires_in=3600)
