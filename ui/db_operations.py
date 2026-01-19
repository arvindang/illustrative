"""Database operations for users and novels."""
from typing import Optional

from sqlalchemy.orm import Session

from models.base import SessionLocal
from models.user import User
from models.novel import GraphicNovel
from auth_utils import hash_password, verify_password
from storage.bucket import BucketStorage


def get_db() -> Optional[Session]:
    """Get database session if configured."""
    if SessionLocal is None:
        return None
    return SessionLocal()


def db_register(email: str, password: str) -> tuple[bool, str]:
    """Register a new user."""
    db = get_db()
    if db is None:
        return False, "Database not configured"

    try:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            db.close()
            return False, "Email already registered"

        user = User(email=email, password_hash=hash_password(password))
        db.add(user)
        db.commit()
        db.close()
        return True, "Account created"
    except Exception as e:
        db.close()
        return False, str(e)


def db_login(email: str, password: str) -> tuple[bool, str]:
    """Login user."""
    db = get_db()
    if db is None:
        return False, "Database not configured"

    try:
        user = db.query(User).filter(User.email == email).first()
        if not user or not verify_password(password, user.password_hash):
            db.close()
            return False, "Invalid email or password"

        user_id = str(user.id)
        db.close()
        return True, user_id
    except Exception as e:
        db.close()
        return False, str(e)


def db_has_api_key(user_id: str) -> bool:
    """Check if user has saved API key."""
    db = get_db()
    if db is None:
        return False
    try:
        user = db.query(User).filter(User.id == user_id).first()
        result = user.has_api_key() if user else False
        db.close()
        return result
    except Exception:
        db.close()
        return False


def db_get_api_key(user_id: str) -> Optional[str]:
    """Get decrypted API key."""
    db = get_db()
    if db is None:
        return None
    try:
        user = db.query(User).filter(User.id == user_id).first()
        key = user.get_gemini_api_key() if user and user.has_api_key() else None
        db.close()
        return key
    except Exception:
        db.close()
        return None


def db_save_api_key(user_id: str, api_key: str) -> bool:
    """Save encrypted API key."""
    db = get_db()
    if db is None:
        return False
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.set_gemini_api_key(api_key)
            db.commit()
        db.close()
        return True
    except Exception:
        db.close()
        return False


def db_delete_api_key(user_id: str) -> bool:
    """Delete API key."""
    db = get_db()
    if db is None:
        return False
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.gemini_api_key_encrypted = None
            db.commit()
        db.close()
        return True
    except Exception:
        db.close()
        return False


def db_list_novels(user_id: str) -> list[dict]:
    """List user's novels."""
    db = get_db()
    if db is None:
        return []
    try:
        novels = db.query(GraphicNovel).filter(
            GraphicNovel.user_id == user_id
        ).order_by(GraphicNovel.created_at.desc()).all()
        result = [n.to_dict() for n in novels]
        db.close()
        return result
    except Exception:
        db.close()
        return []


def db_create_novel(
    user_id: str,
    title: str,
    source_filename: str,
    art_style: str,
    tone: str,
    page_count: int,
    estimated_cost: Optional[float] = None
) -> Optional[str]:
    """Create novel record with optional cost estimate."""
    db = get_db()
    if db is None:
        return None
    try:
        novel = GraphicNovel(
            user_id=user_id,
            title=title,
            source_filename=source_filename,
            art_style=art_style,
            narrative_tone=tone,
            page_count=page_count,
            status="processing",
            estimated_cost=estimated_cost,
        )
        db.add(novel)
        db.commit()
        db.refresh(novel)
        novel_id = str(novel.id)
        db.close()
        return novel_id
    except Exception:
        db.close()
        return None


def db_update_novel(novel_id: str, **kwargs) -> bool:
    """Update novel."""
    db = get_db()
    if db is None:
        return False
    try:
        novel = db.query(GraphicNovel).filter(GraphicNovel.id == novel_id).first()
        if novel:
            for k, v in kwargs.items():
                if hasattr(novel, k) and v is not None:
                    setattr(novel, k, v)
            db.commit()
        db.close()
        return True
    except Exception:
        db.close()
        return False


def db_delete_novel(novel_id: str, user_id: str) -> bool:
    """Delete novel."""
    db = get_db()
    if db is None:
        return False
    try:
        novel = db.query(GraphicNovel).filter(
            GraphicNovel.id == novel_id, GraphicNovel.user_id == user_id
        ).first()
        if novel:
            db.delete(novel)
            db.commit()
        db.close()
        return True
    except Exception:
        db.close()
        return False


def db_get_novel(novel_id: str, user_id: str) -> Optional[dict]:
    """Get a single novel by ID."""
    db = get_db()
    if db is None:
        return None
    try:
        novel = db.query(GraphicNovel).filter(
            GraphicNovel.id == novel_id, GraphicNovel.user_id == user_id
        ).first()
        result = novel.to_dict() if novel else None
        db.close()
        return result
    except Exception:
        db.close()
        return None


def get_download_url(storage_key: str) -> Optional[str]:
    """Generate a presigned download URL for a storage key."""
    try:
        storage = BucketStorage()
        return storage.generate_presigned_url(storage_key, expiration=3600)
    except Exception:
        return None
