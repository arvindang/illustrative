"""FastAPI application for Illustrate AI backend."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import auth_router, novels_router

app = FastAPI(
    title="Illustrate AI API",
    description="Backend API for graphic novel generation and user management",
    version="1.0.0",
)

# CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/api")
app.include_router(novels_router, prefix="/api")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Illustrate AI API", "docs": "/docs"}
