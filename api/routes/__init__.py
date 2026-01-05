"""API routes."""
from api.routes.auth import router as auth_router
from api.routes.novels import router as novels_router

__all__ = ["auth_router", "novels_router"]
