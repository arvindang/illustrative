"""Authentication routes."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from api.schemas import (
    UserRegister,
    UserResponse,
    TokenResponse,
    ApiKeyRequest,
    ApiKeyStatus,
    MessageResponse,
)
from api.dependencies import (
    get_db,
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
)
from models.user import User

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(request: UserRegister, db: Session = Depends(get_db)):
    """Register a new user."""
    # Check if email already exists
    existing = db.query(User).filter(User.email == request.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create user
    user = User(
        email=request.email,
        password_hash=hash_password(request.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return UserResponse(
        id=str(user.id),
        email=user.email,
        has_api_key=user.has_api_key(),
        created_at=user.created_at,
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """Login and get access token."""
    user = db.query(User).filter(User.email == form_data.username).first()

    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(str(user.id))
    return TokenResponse(access_token=access_token)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user info."""
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        has_api_key=current_user.has_api_key(),
        created_at=current_user.created_at,
    )


@router.put("/api-key", response_model=MessageResponse)
async def save_api_key(
    request: ApiKeyRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Save encrypted Gemini API key for user."""
    current_user.set_gemini_api_key(request.api_key)
    db.commit()
    return MessageResponse(message="API key saved successfully")


@router.get("/api-key", response_model=ApiKeyStatus)
async def check_api_key(current_user: User = Depends(get_current_user)):
    """Check if user has a saved API key."""
    return ApiKeyStatus(has_api_key=current_user.has_api_key())


@router.delete("/api-key", response_model=MessageResponse)
async def delete_api_key(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete user's saved API key."""
    current_user.gemini_api_key_encrypted = None
    db.commit()
    return MessageResponse(message="API key deleted")
