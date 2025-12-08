"""
API Configuration for Production Backend
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
DATABASE_DIR = BASE_DIR / "database"

# Ensure directories exist
UPLOADS_DIR.mkdir(exist_ok=True)
DATABASE_DIR.mkdir(exist_ok=True)

# Database - SQLite for development, PostgreSQL for production
# Use SQLite by default if DATABASE_URL not set
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    f"sqlite:///{DATABASE_DIR}/ocr.db"
)

# JWT Settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Admin settings
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@example.com")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")  # Change in production!

# Gemini API (shared, managed by admin)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "models/gemini-2.5-flash")

# Quota settings (per user, per day)
# Admin gets unlimited (999,999,999), regular users get 100,000
DEFAULT_DAILY_TOKEN_LIMIT = 100_000  # Regular users
ADMIN_DAILY_TOKEN_LIMIT = 999_999_999  # Admin users
DEFAULT_DAILY_DOCUMENT_LIMIT = 50

# CORS
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:8501",  # Streamlit
]
