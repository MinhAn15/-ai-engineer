"""
Pydantic Schemas for API validation
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr


# ============ Token Schemas ============

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[int] = None


# ============ User Schemas ============

class UserBase(BaseModel):
    email: EmailStr
    username: str


class UserCreate(UserBase):
    password: str


class UserResponse(UserBase):
    id: int
    is_active: bool
    is_admin: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    username: str
    password: str


# ============ Document Schemas ============

class DocumentBase(BaseModel):
    filename: str


class DocumentCreate(DocumentBase):
    pass


class DocumentResponse(DocumentBase):
    id: int
    user_id: int
    status: str
    ocr_confidence: Optional[float] = None
    num_text_blocks: Optional[int] = None
    processing_time_ms: Optional[float] = None
    schema_prompt: Optional[str] = None
    num_entities: Optional[int] = None
    num_relations: Optional[int] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class DocumentList(BaseModel):
    documents: List[DocumentResponse]
    total: int


# ============ OCR Schemas ============

class OCRRequest(BaseModel):
    language: str = "eng"
    preprocess: bool = False
    use_hybrid: bool = False  # Enable hybrid OCR with Vision LLM for low-confidence regions
    min_confidence: float = 0.75  # Threshold for hybrid OCR


class OCRResponse(BaseModel):
    document_id: int
    status: str
    ocr_confidence: Optional[float] = None
    num_text_blocks: Optional[int] = None
    processing_time_ms: Optional[float] = None
    text_preview: Optional[str] = None


class ExtractRequest(BaseModel):
    schema_prompt: str
    min_confidence: float = 0.75



class ExtractResponse(BaseModel):
    document_id: int
    status: str
    toon_output: Optional[str] = None
    fields: Optional[List[dict]] = None  # Structured extraction with refs
    num_entities: int = 0
    num_relations: int = 0
    tokens_used: int = 0


# ============ Quota Schemas ============

class QuotaResponse(BaseModel):
    max_tokens_daily: int
    max_documents_daily: int
    tokens_used_today: int
    documents_today: int
    tokens_remaining: int
    documents_remaining: int
    
    class Config:
        from_attributes = True


# ============ Usage Schemas ============

class UsageLogResponse(BaseModel):
    id: int
    document_id: Optional[int] = None
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class UsageSummary(BaseModel):
    total_requests: int
    total_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float
