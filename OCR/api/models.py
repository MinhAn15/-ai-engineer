"""
Database ORM Models
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship

from api.database import Base


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    documents = relationship("Document", back_populates="user")
    usage_logs = relationship("UsageLog", back_populates="user")
    quota = relationship("Quota", back_populates="user", uselist=False)


class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # File info
    filename = Column(String(255), nullable=False)
    original_path = Column(String(500))
    output_txt_path = Column(String(500))
    output_json_path = Column(String(500))
    output_toon_path = Column(String(500))
    
    # OCR results
    ocr_confidence = Column(Float)
    num_text_blocks = Column(Integer)
    processing_time_ms = Column(Float)
    
    # LLM extraction
    schema_prompt = Column(Text)
    num_entities = Column(Integer)
    num_relations = Column(Integer)
    
    # Status
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="documents")
    usage_logs = relationship("UsageLog", back_populates="document")


class UsageLog(Base):
    __tablename__ = "usage_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id"))
    
    model = Column(String(100))
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="usage_logs")
    document = relationship("Document", back_populates="usage_logs")


class Quota(Base):
    __tablename__ = "quotas"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    
    # Limits
    max_tokens_daily = Column(Integer, default=100000)
    max_documents_daily = Column(Integer, default=50)
    
    # Current usage
    tokens_used_today = Column(Integer, default=0)
    documents_today = Column(Integer, default=0)
    
    # Reset tracking
    last_reset = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="quota")
