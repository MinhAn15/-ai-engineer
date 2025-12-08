"""
OCR Routes
"""
import os
import sys
import shutil
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.database import get_db
from api.dependencies import get_current_user, check_quota, update_quota_usage
from api.schemas import (
    DocumentResponse, DocumentList, 
    OCRRequest, OCRResponse,
    ExtractRequest, ExtractResponse
)
from api.models import User, Document, UsageLog
from api.config import UPLOADS_DIR, GEMINI_API_KEY, LLM_MODEL

router = APIRouter(prefix="/ocr", tags=["OCR"])


def get_user_upload_dir(user_id: int, document_id: int) -> Path:
    """Get user-specific upload directory"""
    user_dir = UPLOADS_DIR / str(user_id) / str(document_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload a document (PDF or image) for OCR processing.
    """
    # Check quota
    check_quota(current_user, db)
    
    # Validate file type
    allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Create document record
    doc = Document(
        user_id=current_user.id,
        filename=file.filename,
        status="pending"
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    
    # Save file to user directory
    user_dir = get_user_upload_dir(current_user.id, doc.id)
    file_path = user_dir / f"original{file_ext}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    doc.original_path = str(file_path)
    db.commit()
    
    # Update quota
    update_quota_usage(current_user, db, tokens_used=0, document_count=1)
    
    return doc


@router.post("/process/{document_id}", response_model=OCRResponse)
async def process_document(
    document_id: int,
    request: OCRRequest = OCRRequest(),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Run OCR on uploaded document.
    
    Options:
    - use_hybrid=False: Standard Tesseract OCR (FREE)
    - use_hybrid=True: Hybrid OCR with Vision LLM for low-confidence regions (PAID)
    """
    # Get document
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()
    
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    if not doc.original_path or not os.path.exists(doc.original_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Original file not found"
        )
    
    doc.status = "processing"
    db.commit()
    
    try:
        import json
        from PIL import Image
        
        user_dir = Path(doc.original_path).parent
        
        if request.use_hybrid:
            # ========== HYBRID OCR MODE ==========
            from src.hybrid_ocr import HybridOCR
            
            # Check if it's an image file
            file_ext = Path(doc.original_path).suffix.lower()
            
            if file_ext == '.pdf':
                # For PDFs, convert to images first
                from src.pdf_processor import pdf_to_images
                images = pdf_to_images(doc.original_path)
                
                all_text = []
                all_blocks = []
                total_tokens = 0
                
                ocr = HybridOCR(
                    min_confidence=request.min_confidence,
                    enable_vision_llm=True
                )
                
                for i, img in enumerate(images):
                    result = ocr.process(img, lang=request.language)
                    all_text.append(result.full_text)
                    all_blocks.extend([
                        {"text": b.text, "confidence": b.confidence, "source": b.source}
                        for b in result.blocks
                    ])
                    total_tokens += result.stats.get("tokens_used", 0)
                
                full_text = "\n\n--- Page Break ---\n\n".join(all_text)
                avg_confidence = sum(b["confidence"] for b in all_blocks) / len(all_blocks) if all_blocks else 0
                
            else:
                # For images, process directly
                img = Image.open(doc.original_path)
                
                ocr = HybridOCR(
                    min_confidence=request.min_confidence,
                    enable_vision_llm=True
                )
                result = ocr.process(img, lang=request.language)
                
                full_text = result.full_text
                all_blocks = [
                    {"text": b.text, "confidence": b.confidence, "source": b.source}
                    for b in result.blocks
                ]
                avg_confidence = sum(b["confidence"] for b in all_blocks) / len(all_blocks) if all_blocks else 0
                total_tokens = result.stats.get("tokens_used", 0)
            
            # Save text
            txt_path = user_dir / "output.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            
            # Save metadata JSON
            json_path = user_dir / "output.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "ocr_mode": "hybrid",
                    "avg_confidence": avg_confidence,
                    "tokens_used": total_tokens,
                    "min_confidence_threshold": request.min_confidence,
                    "text_blocks": all_blocks
                }, f, indent=2, ensure_ascii=False)
            
            # Log usage if tokens were used
            if total_tokens > 0:
                usage_log = UsageLog(
                    user_id=current_user.id,
                    document_id=doc.id,
                    model="hybrid-vision",
                    input_tokens=int(total_tokens * 0.7),
                    output_tokens=int(total_tokens * 0.3),
                    total_tokens=total_tokens
                )
                db.add(usage_log)
                update_quota_usage(current_user, db, tokens_used=total_tokens)
            
            # Update document
            doc.output_txt_path = str(txt_path)
            doc.output_json_path = str(json_path)
            doc.ocr_confidence = avg_confidence
            doc.num_text_blocks = len(all_blocks)
            doc.processing_time_ms = 0  # Not tracked in hybrid mode
            doc.status = "completed"
            db.commit()
            
        else:
            # ========== STANDARD OCR MODE ==========
            from src.extractor import process_document as run_ocr
            
            result = run_ocr(
                doc.original_path,
                lang=request.language,
                preprocess=request.preprocess
            )
            
            # Save text
            txt_path = user_dir / "output.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(result.full_text)
            
            # Save metadata JSON
            json_path = user_dir / "output.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "ocr_mode": "standard",
                    "ocr_score": result.ocr_score,
                    "avg_confidence": result.avg_confidence,
                    "num_pages": result.num_pages,
                    "latency_ms": result.latency_ms,
                    "models_used": result.models_used,
                    "text_blocks": [
                        {"text": b.text, "confidence": b.confidence}
                        for b in result.text_blocks
                    ]
                }, f, indent=2, ensure_ascii=False)
            
            # Update document
            doc.output_txt_path = str(txt_path)
            doc.output_json_path = str(json_path)
            doc.ocr_confidence = result.avg_confidence
            doc.num_text_blocks = len(result.text_blocks)
            doc.processing_time_ms = result.latency_ms
            doc.status = "completed"
            db.commit()
        
        return OCRResponse(
            document_id=doc.id,
            status=doc.status,
            ocr_confidence=doc.ocr_confidence,
            num_text_blocks=doc.num_text_blocks,
            processing_time_ms=doc.processing_time_ms,
            text_preview=full_text[:500] if request.use_hybrid else result.full_text[:500] if result.full_text else None
        )
        
    except Exception as e:
        doc.status = "failed"
        doc.error_message = str(e)
        db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(e)}"
        )


@router.post("/extract/{document_id}", response_model=ExtractResponse)
async def extract_schema(
    document_id: int,
    request: ExtractRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Extract structured data from OCR text using LLM.
    """
    # Check quota
    check_quota(current_user, db, tokens_needed=1000)  # Estimate
    
    # Get document
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()
    
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    if doc.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document must be processed with OCR first"
        )
    
    if not doc.output_txt_path or not os.path.exists(doc.output_txt_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OCR output not found"
        )
    
    try:
        # Read OCR text
        with open(doc.output_txt_path, "r", encoding="utf-8") as f:
            ocr_text = f.read()
        
        # Read OCR blocks for confidence filtering
        import json
        with open(doc.output_json_path, "r", encoding="utf-8") as f:
            ocr_data = json.load(f)
        ocr_blocks = ocr_data.get("text_blocks", [])
        
        # Import LLM extractor
        from src.llm_extractor import extract_with_llm, parse_toon
        
        # Run LLM extraction
        toon_result = extract_with_llm(
            ocr_text,
            request.schema_prompt,
            api_key=GEMINI_API_KEY,
            model=LLM_MODEL,
            min_confidence=request.min_confidence,
            ocr_blocks=ocr_blocks
        )
        
        if toon_result.startswith("# Error"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=toon_result
            )
        
        # Parse TOON
        parsed = parse_toon(toon_result)
        
        # Save TOON output
        user_dir = Path(doc.original_path).parent
        toon_path = user_dir / "output.toon"
        with open(toon_path, "w", encoding="utf-8") as f:
            f.write(toon_result)
        
        # Estimate tokens used
        input_tokens = int(len(ocr_text.split()) * 1.3)
        output_tokens = int(len(toon_result.split()) * 1.3)
        total_tokens = input_tokens + output_tokens
        
        # Log usage
        usage_log = UsageLog(
            user_id=current_user.id,
            document_id=doc.id,
            model=LLM_MODEL,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens
        )
        db.add(usage_log)
        
        # Update document
        doc.output_toon_path = str(toon_path)
        doc.schema_prompt = request.schema_prompt
        doc.num_entities = len(parsed.get("entities", []))
        doc.num_relations = len(parsed.get("relations", []))
        db.commit()
        
        # Update quota
        update_quota_usage(current_user, db, tokens_used=total_tokens)
        
        return ExtractResponse(
            document_id=doc.id,
            status="extracted",
            toon_output=toon_result,
            num_entities=doc.num_entities,
            num_relations=doc.num_relations,
            tokens_used=total_tokens
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM extraction failed: {str(e)}"
        )


@router.get("/documents", response_model=DocumentList)
async def list_documents(
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List user's documents.
    """
    query = db.query(Document).filter(Document.user_id == current_user.id)
    total = query.count()
    documents = query.order_by(Document.created_at.desc()).offset(skip).limit(limit).all()
    
    return DocumentList(documents=documents, total=total)


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get document details.
    """
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()
    
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    return doc


@router.get("/documents/{document_id}/text")
async def get_document_text(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get OCR text output.
    """
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()
    
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    if not doc.output_txt_path or not os.path.exists(doc.output_txt_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="OCR text not available"
        )
    
    with open(doc.output_txt_path, "r", encoding="utf-8") as f:
        return {"text": f.read()}


@router.get("/documents/{document_id}/toon")
async def get_document_toon(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get TOON extraction output.
    """
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()
    
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    if not doc.output_toon_path or not os.path.exists(doc.output_toon_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="TOON extraction not available"
        )
    
    with open(doc.output_toon_path, "r", encoding="utf-8") as f:
        return {"toon": f.read()}


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a document and its files.
    """
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()
    
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Delete files
    if doc.original_path:
        user_dir = Path(doc.original_path).parent
        if user_dir.exists():
            shutil.rmtree(user_dir)
    
    # Delete database record
    db.delete(doc)
    db.commit()


@router.get("/documents/{document_id}/usage")
async def get_document_usage(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get token usage for a document.
    """
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()
    
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Get usage logs for this document
    usage_logs = db.query(UsageLog).filter(
        UsageLog.document_id == document_id
    ).all()
    
    total_tokens = sum(log.total_tokens for log in usage_logs)
    
    return {
        "document_id": document_id,
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_tokens * 0.000001, 6),  # Estimate $1/1M tokens
        "extractions": [
            {
                "model": log.model,
                "input_tokens": log.input_tokens,
                "output_tokens": log.output_tokens,
                "total_tokens": log.total_tokens,
                "created_at": log.created_at.isoformat()
            }
            for log in usage_logs
        ],
        "pipeline": {
            "upload": {
                "status": "completed" if doc.original_path else "pending",
                "filename": doc.filename
            },
            "ocr": {
                "status": "completed" if doc.output_txt_path else ("failed" if doc.status == "failed" else "pending"),
                "confidence": doc.ocr_confidence,
                "text_blocks": doc.num_text_blocks,
                "processing_time_ms": doc.processing_time_ms
            },
            "extraction": {
                "status": "completed" if doc.output_toon_path else "pending",
                "entities": doc.num_entities,
                "relations": doc.num_relations,
                "tokens_used": total_tokens
            }
        }
    }
