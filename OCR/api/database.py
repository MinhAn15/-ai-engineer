"""
Database setup with SQLAlchemy (SQLite or PostgreSQL)
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from api.config import DATABASE_URL

# Create engine with appropriate settings
if DATABASE_URL.startswith("sqlite"):
    # SQLite needs check_same_thread=False for FastAPI
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    # PostgreSQL
    engine = create_engine(DATABASE_URL)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    from api.models import User, Document, UsageLog, Quota
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created")
