"""
FastAPI Main Application
"""
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.config import CORS_ORIGINS, ADMIN_EMAIL, ADMIN_PASSWORD
from api.database import init_db, SessionLocal
from api.auth import create_user, get_password_hash
from api.models import User
from api.routes import auth, users, ocr


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print("[*] Starting OCR API Server...")
    
    # Initialize database
    init_db()
    
    # Create default admin user if not exists
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.email == ADMIN_EMAIL).first()
        if not admin:
            admin = User(
                email=ADMIN_EMAIL,
                username="admin",
                hashed_password=get_password_hash(ADMIN_PASSWORD),
                is_admin=True,
                is_active=True
            )
            db.add(admin)
            db.commit()
            print(f"[+] Default admin user created: {ADMIN_EMAIL}")
        else:
            print(f"[+] Admin user exists: {ADMIN_EMAIL}")
    finally:
        db.close()
    
    print("[+] API Server ready!")
    print("[i] Swagger docs: http://localhost:8000/docs")
    
    yield
    
    # Shutdown
    print("[*] Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Document Intelligence OCR API",
    description="Production-ready OCR and LLM extraction API with multi-user support",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(ocr.router)


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Document Intelligence OCR API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "ok"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
