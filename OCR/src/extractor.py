"""
Main extraction pipeline for Document Intelligence.
"""

import sys
import time
from pathlib import Path
from typing import List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pdf_processor import (
    pdf_to_images,
    ocr_image_to_text,
    ocr_image_to_blocks
)
from src.models import TextBlock, DocumentExtractionResult
from config import DEFAULT_LANG, DEFAULT_DPI


def process_pdf_with_tesseract(
    pdf_path: str,
    lang: str = DEFAULT_LANG,
    dpi: int = DEFAULT_DPI
) -> DocumentExtractionResult:
    '''
    Extract text and blocks from PDF using Tesseract OCR.
    '''
    print(f"📄 Processing: {pdf_path}")
    print(f"   Language: {lang}, DPI: {dpi}")
    
    start_time = time.time()
    
    # Convert PDF to images
    images = pdf_to_images(pdf_path, dpi=dpi)
    if not images:
        print("✗ Failed to convert PDF to images")
        return DocumentExtractionResult(
            full_text="",
            latency_ms=(time.time() - start_time) * 1000.0,
            models_used=["tesseract"]
        )
    
    print(f"✓ Converted to {len(images)} page(s)")
    
    all_blocks: List[TextBlock] = []
    full_text_parts: List[str] = []
    
    # Process each page
    for page_idx, img in enumerate(images):
        print(f"   Processing page {page_idx + 1}/{len(images)}...", end=" ")
        
        # Extract plain text
        page_text = ocr_image_to_text(img, lang=lang)
        full_text_parts.append(page_text)
        
        # Extract blocks with bounding boxes
        blocks_raw = ocr_image_to_blocks(img, lang=lang)
        for block_dict in blocks_raw:
            text_block = TextBlock(
                text=block_dict["text"],
                confidence=block_dict["confidence"],
                bounding_box=block_dict["bbox"],
                source_model="tesseract",
                page_number=page_idx
            )
            all_blocks.append(text_block)
        
        print(f"✓ {len(blocks_raw)} blocks")
    
    # Combine results
    full_text = "\n".join(full_text_parts)
    avg_confidence = (
        sum(b.confidence for b in all_blocks) / len(all_blocks)
        if all_blocks else 0.0
    )
    latency_ms = (time.time() - start_time) * 1000.0
    
    result = DocumentExtractionResult(
        full_text=full_text,
        text_blocks=all_blocks,
        ocr_score=avg_confidence,
        avg_confidence=avg_confidence,
        latency_ms=latency_ms,
        models_used=["tesseract"],
        num_pages=len(images)
    )
    
    print(f"\n{result.summary()}\n")
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "data/samples/sample1.pdf"
    
    if Path(pdf_path).exists():
        result = process_pdf_with_tesseract(pdf_path)
        
        # Preview extracted text
        print("\n📝 Extracted Text Preview (first 500 chars):")
        print("=" * 50)
        print(result.full_text[:500])
        print("=" * 50)
        
        # Show sample blocks
        print(f"\n🔍 Sample Text Blocks (first 10):")
        for i, block in enumerate(result.text_blocks[:10], 1):
            print(f"{i:2}. [{block.confidence:.2%}] {block.text[:60]}")
    else:
        print(f"✗ PDF not found: {pdf_path}")
        print(f"   Please add a sample PDF to data/samples/")
