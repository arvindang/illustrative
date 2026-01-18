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
    PrescanRequest,
    PrescanResponse,
    CostEstimateRequest,
    CostEstimateResponse,
    CostBreakdownResponse,
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


# ==================== Cost Estimation Endpoints ====================

@router.post("/prescan", response_model=PrescanResponse)
async def prescan_document(
    request: PrescanRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Quick scan to extract characters and objects for cost estimation.

    This uses a lightweight LLM call (~$0.01) to identify named characters
    and key objects from the document. More accurate than heuristics.
    """
    from utils import calculate_page_count, get_client
    from cost_calculator import PRESCAN_COST
    from google import genai

    content = request.content[:request.max_chars]  # Limit for token savings
    word_count = len(content.split())

    # Calculate recommended pages
    page_info = calculate_page_count(word_count)
    recommended_pages = page_info['recommended']

    # Quick extraction prompt
    prompt = f"""Analyze this text excerpt and extract:
1. CHARACTER_NAMES: List all named characters (people with names, not generic descriptions)
2. KEY_OBJECTS: List important recurring objects, vehicles, or locations that appear multiple times

Text (first {len(content)} characters):
---
{content}
---

Respond in this exact format:
CHARACTERS: Name1, Name2, Name3
OBJECTS: Object1, Object2, Object3

If none found, write "CHARACTERS: None" or "OBJECTS: None"."""

    try:
        client = get_client()
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",  # Use fast model for prescan
            contents=prompt,
        )

        # Parse response
        text = response.text.strip()
        characters = []
        objects = []

        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('CHARACTERS:'):
                chars_str = line.replace('CHARACTERS:', '').strip()
                if chars_str.lower() != 'none':
                    characters = [c.strip() for c in chars_str.split(',') if c.strip()]
            elif line.startswith('OBJECTS:'):
                objs_str = line.replace('OBJECTS:', '').strip()
                if objs_str.lower() != 'none':
                    objects = [o.strip() for o in objs_str.split(',') if o.strip()]

        return PrescanResponse(
            word_count=word_count,
            character_names=characters,
            object_names=objects,
            recommended_pages=recommended_pages,
            prescan_cost=PRESCAN_COST,
        )

    except Exception as e:
        # Fallback to heuristics if LLM fails
        from cost_calculator import (
            estimate_characters_from_word_count,
            estimate_objects_from_word_count,
        )

        num_chars = estimate_characters_from_word_count(word_count)
        num_objs = estimate_objects_from_word_count(word_count)

        return PrescanResponse(
            word_count=word_count,
            character_names=[f"Character {i+1}" for i in range(num_chars)],
            object_names=[f"Object {i+1}" for i in range(num_objs)],
            recommended_pages=recommended_pages,
            prescan_cost=0.0,  # No cost if LLM failed
        )


@router.post("/estimate-cost", response_model=CostEstimateResponse)
async def estimate_cost(
    request: CostEstimateRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Calculate estimated processing cost with detailed breakdown.

    Returns the estimated Vertex AI cost plus a margin fee.
    Use /prescan first for more accurate character/object counts,
    or provide them directly if known.
    """
    from cost_calculator import (
        estimate_total_cost,
        estimate_characters_from_word_count,
        estimate_objects_from_word_count,
        DEFAULT_MARGIN_PERCENT,
    )
    from utils import calculate_page_count

    # Determine page count
    if request.page_count:
        page_count = request.page_count
    else:
        page_info = calculate_page_count(request.word_count)
        page_count = page_info['recommended']

    # Determine character/object counts (use heuristics if not provided)
    num_characters = request.num_characters
    if num_characters is None:
        num_characters = estimate_characters_from_word_count(request.word_count)

    num_objects = request.num_objects
    if num_objects is None:
        num_objects = estimate_objects_from_word_count(request.word_count)

    margin = request.margin_percent if request.margin_percent is not None else DEFAULT_MARGIN_PERCENT

    # Calculate estimate
    estimate = estimate_total_cost(
        word_count=request.word_count,
        page_count=page_count,
        num_characters=num_characters,
        num_objects=num_objects,
        margin_percent=margin,
        include_prescan=True,
    )

    return CostEstimateResponse(
        word_count=estimate.word_count,
        estimated_pages=estimate.pages,
        estimated_panels=estimate.panels,
        characters=estimate.characters,
        objects=estimate.objects,
        breakdown=CostBreakdownResponse(
            prescan=estimate.breakdown.prescan,
            scripting_input=estimate.breakdown.scripting_input,
            scripting_output=estimate.breakdown.scripting_output,
            cache_savings=estimate.breakdown.cache_savings,
            image_generation=estimate.breakdown.image_generation,
            character_refs=estimate.breakdown.character_refs,
            object_refs=estimate.breakdown.object_refs,
            composition_analysis=estimate.breakdown.composition_analysis,
        ),
        subtotal_vertex=estimate.subtotal_vertex,
        margin_fee=estimate.margin_fee,
        margin_percent=estimate.margin_percent,
        total_cost=estimate.total_cost,
    )


@router.get("/estimate-cost/quick", response_model=CostEstimateResponse)
async def quick_estimate_cost(
    word_count: int,
    page_count: Optional[int] = None,
    current_user: User = Depends(get_current_user),
):
    """
    Quick cost estimate using heuristics only (no prescan needed).

    Faster but less accurate than POST /estimate-cost with prescan data.
    """
    from cost_calculator import quick_estimate, CostBreakdownResponse

    estimate = quick_estimate(word_count=word_count, page_count=page_count)

    return CostEstimateResponse(
        word_count=estimate.word_count,
        estimated_pages=estimate.pages,
        estimated_panels=estimate.panels,
        characters=estimate.characters,
        objects=estimate.objects,
        breakdown=CostBreakdownResponse(
            prescan=estimate.breakdown.prescan,
            scripting_input=estimate.breakdown.scripting_input,
            scripting_output=estimate.breakdown.scripting_output,
            cache_savings=estimate.breakdown.cache_savings,
            image_generation=estimate.breakdown.image_generation,
            character_refs=estimate.breakdown.character_refs,
            object_refs=estimate.breakdown.object_refs,
            composition_analysis=estimate.breakdown.composition_analysis,
        ),
        subtotal_vertex=estimate.subtotal_vertex,
        margin_fee=estimate.margin_fee,
        margin_percent=estimate.margin_percent,
        total_cost=estimate.total_cost,
    )
