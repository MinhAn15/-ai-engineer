"""
Main extraction pipeline for Document Intelligence.
Supports both PDF and Image files.
"""

import sys
import time
from pathlib import Path
from typing import List, Union

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pdf_processor import (
    pdf_to_images,
    ocr_image_to_text,
    ocr_image_to_blocks
)
from src.models import TextBlock, DocumentExtractionResult
from config import (
    DEFAULT_LANG, 
    DEFAULT_DPI, 
    SUPPORTED_IMAGE_FORMATS, 
    SUPPORTED_PDF_FORMAT
)
from PIL import Image


def process_image_with_tesseract(
    image_path: str,
    lang: str = DEFAULT_LANG,
    preprocess: bool = False,
    preprocess_method: str = "adaptive"
) -> DocumentExtractionResult:
    '''
    Extract text from a single image using Tesseract OCR.
    
    Args:
        image_path: Path to image file
        lang: Language for OCR
    
    Returns:
        DocumentExtractionResult with extracted text and metadata
    '''
    print(f"📸 Processing image: {image_path}")
    print(f"   Language: {lang}, Preprocess: {preprocess}")
    
    start_time = time.time()
    
    # Load image
    try:
        img = Image.open(image_path)
        print(f"✓ Image loaded: {img.size[0]}x{img.size[1]} pixels")
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        return DocumentExtractionResult(
            full_text="",
            latency_ms=(time.time() - start_time) * 1000.0,
            models_used=["tesseract"]
        )
    
    all_blocks: List[TextBlock] = []
    
    # Extract plain text
    print(f"   Extracting text...", end=" ")
    full_text = ocr_image_to_text(img, lang=lang, preprocess=preprocess, preprocess_method=preprocess_method)
    
    # Extract blocks with bounding boxes
    blocks_raw = ocr_image_to_blocks(img, lang=lang, preprocess=preprocess, preprocess_method=preprocess_method)
    for block_dict in blocks_raw:
        text_block = TextBlock(
            text=block_dict["text"],
            confidence=block_dict["confidence"],
            bounding_box=block_dict["bbox"],
            source_model="tesseract",
            page_number=0  # Single image = page 0
        )
        all_blocks.append(text_block)
    
    print(f"✓ {len(blocks_raw)} blocks")
    
    # Calculate metrics
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
        num_pages=1
    )
    
    print(f"\n{result.summary()}\n")
    
    return result


def process_pdf_with_tesseract(
    pdf_path: str,
    lang: str = DEFAULT_LANG,
    dpi: int = DEFAULT_DPI,
    preprocess: bool = False,
    preprocess_method: str = "adaptive"
) -> DocumentExtractionResult:
    '''
    Extract text from PDF using Tesseract OCR.
    
    Args:
        pdf_path: Path to PDF file
        lang: Language for OCR
        dpi: Image resolution for conversion
    
    Returns:
        DocumentExtractionResult with extracted text and metadata
    '''
    print(f"📄 Processing PDF: {pdf_path}")
    print(f"   Language: {lang}, DPI: {dpi}, Preprocess: {preprocess}")
    
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
        
        page_text = ocr_image_to_text(img, lang=lang, preprocess=preprocess, preprocess_method=preprocess_method)
        full_text_parts.append(page_text)
        
        blocks_raw = ocr_image_to_blocks(img, lang=lang, preprocess=preprocess, preprocess_method=preprocess_method)
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


def process_document(
    file_path: str,
    lang: str = DEFAULT_LANG,
    dpi: int = DEFAULT_DPI,
    preprocess: bool = False,
    preprocess_method: str = "adaptive"
) -> DocumentExtractionResult:
    '''
    Unified document processing - auto-detect PDF or Image.
    
    Args:
        file_path: Path to PDF or image file
        lang: Language for OCR
        dpi: DPI for PDF conversion (ignored for images)
    
    Returns:
        DocumentExtractionResult
    '''
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"✗ File not found: {file_path}")
        return DocumentExtractionResult(
            full_text="",
            models_used=["tesseract"]
        )
    
    file_ext = file_path.suffix.lower()
    
    # Check file type
    if file_ext == SUPPORTED_PDF_FORMAT:
        return process_pdf_with_tesseract(str(file_path), lang, dpi, preprocess, preprocess_method)
    elif file_ext in SUPPORTED_IMAGE_FORMATS:
        return process_image_with_tesseract(str(file_path), lang, preprocess, preprocess_method)
    else:
        print(f"✗ Unsupported file type: {file_ext}")
        print(f"   Supported: PDF, {', '.join(SUPPORTED_IMAGE_FORMATS)}")
        return DocumentExtractionResult(
            full_text="",
            models_used=["tesseract"]
        )


def save_results(result: DocumentExtractionResult, file_path: str, outputs_dir: Path):
    """
    Save OCR results to text file and metadata to JSON.
    
    Args:
        result: DocumentExtractionResult from OCR
        file_path: Original input file path
        outputs_dir: Directory to save outputs
    
    Returns:
        Tuple of (text_path, json_path) or (None, None) if failed
    """
    import json
    from datetime import datetime
    
    # Create outputs directory if not exists
    outputs_dir.mkdir(exist_ok=True)
    
    # Get filename without extension
    input_path = Path(file_path)
    base_name = input_path.stem
    
    text_path = outputs_dir / f"{base_name}.txt"
    json_path = outputs_dir / f"{base_name}_metadata.json"
    
    try:
        # Save extracted text
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(result.full_text)
        
        # Create metadata
        metadata = {
            "source_file": input_path.name,
            "source_path": str(input_path.absolute()),
            "extracted_at": datetime.now().isoformat(),
            "num_pages": result.num_pages,
            "num_text_blocks": len(result.text_blocks),
            "avg_confidence": round(result.avg_confidence, 4),
            "ocr_score": round(result.ocr_score, 4),
            "processing_time_ms": round(result.latency_ms, 2),
            "models_used": result.models_used,
            "text_length": len(result.full_text),
            "output_files": {
                "text": str(text_path.absolute()),
                "metadata": str(json_path.absolute())
            }
        }
        
        # Save metadata JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return text_path, json_path
        
    except Exception as e:
        print(f"⚠ Error saving files: {e}")
        return None, None


def process_single_file(file_path: str, lang: str, outputs_dir: Path, show_preview: bool = True):
    """
    Process a single file and save results.
    
    Args:
        file_path: Path to file
        lang: Language for OCR
        outputs_dir: Directory to save outputs
        show_preview: Whether to show text preview
    
    Returns:
        DocumentExtractionResult or None
    """
    result = process_document(file_path, lang=lang)
    
    if not result.full_text:
        return None
    
    # Display preview
    if show_preview:
        print("\n📝 Extracted Text Preview (first 500 chars):")
        print("=" * 60)
        print(result.full_text[:500])
        print("=" * 60)
        
        print(f"\n🔍 Sample Text Blocks (first 10):")
        for i, block in enumerate(result.text_blocks[:10], 1):
            print(f"{i:2}. [{block.confidence:.2%}] {block.text[:60]}")
    
    # Save results
    text_path, json_path = save_results(result, file_path, outputs_dir)
    
    if text_path:
        print(f"\n💾 Saved to:")
        print(f"   Text: {text_path} ({text_path.stat().st_size:,} bytes)")
        print(f"   JSON: {json_path} ({json_path.stat().st_size:,} bytes)")
    
    return result


def batch_process(directory: str, lang: str, outputs_dir: Path, pattern: str = "*"):
    """
    Process all supported files in a directory.
    
    Args:
        directory: Path to directory containing files
        lang: Language for OCR
        outputs_dir: Directory to save outputs
        pattern: Glob pattern for filtering files (default: all)
    
    Returns:
        List of results with file info
    """
    import json
    from datetime import datetime
    
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"✗ Directory not found: {directory}")
        return []
    
    # Find all supported files
    all_supported = set(SUPPORTED_IMAGE_FORMATS) | {SUPPORTED_PDF_FORMAT}
    files = [f for f in dir_path.glob(pattern) if f.suffix.lower() in all_supported]
    
    if not files:
        print(f"✗ No supported files found in: {directory}")
        print(f"   Supported: {', '.join(all_supported)}")
        return []
    
    print(f"\n{'='*60}")
    print(f"📂 BATCH PROCESSING")
    print(f"{'='*60}")
    print(f"Directory: {dir_path}")
    print(f"Files found: {len(files)}")
    print(f"Language: {lang}")
    print(f"{'='*60}\n")
    
    results = []
    batch_start = time.time()
    
    for idx, file_path in enumerate(files, 1):
        print(f"\n[{idx}/{len(files)}] Processing: {file_path.name}")
        print("-" * 40)
        
        result = process_document(str(file_path), lang=lang)
        
        if result.full_text:
            text_path, json_path = save_results(result, str(file_path), outputs_dir)
            
            results.append({
                "file": file_path.name,
                "status": "success",
                "pages": result.num_pages,
                "confidence": result.avg_confidence,
                "processing_time_ms": result.latency_ms,
                "output_text": str(text_path) if text_path else None,
                "output_json": str(json_path) if json_path else None
            })
            
            if text_path:
                print(f"💾 Saved: {text_path.name}")
        else:
            results.append({
                "file": file_path.name,
                "status": "failed",
                "error": "No text extracted"
            })
            print(f"⚠ Failed to extract text")
    
    # Batch summary
    batch_time = time.time() - batch_start
    success_count = sum(1 for r in results if r["status"] == "success")
    
    print(f"\n{'='*60}")
    print(f"📊 BATCH SUMMARY")
    print(f"{'='*60}")
    print(f"Total files: {len(files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(files) - success_count}")
    print(f"Total time: {batch_time:.2f}s ({batch_time/len(files):.2f}s/file)")
    print(f"{'='*60}")
    
    # Save batch report
    report_path = outputs_dir / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "batch_info": {
                "directory": str(dir_path),
                "timestamp": datetime.now().isoformat(),
                "total_files": len(files),
                "successful": success_count,
                "failed": len(files) - success_count,
                "total_time_seconds": round(batch_time, 2),
                "language": lang
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 Batch report: {report_path}")
    
    return results


if __name__ == "__main__":
    import sys
    from config import OUTPUTS_DIR, IMAGE_DIR, SAMPLES_DIR
    
    # Parse arguments
    args = sys.argv[1:]
    
    # Check for preprocessing flag
    preprocess = "--preprocess" in args or "-p" in args
    if preprocess:
        args = [a for a in args if a not in ("--preprocess", "-p")]
    
    # Check for batch mode
    if "--batch" in args or "--all" in args:
        # Batch processing mode
        batch_idx = args.index("--batch") if "--batch" in args else args.index("--all")
        
        # Get directory (default: data/samples)
        if batch_idx + 1 < len(args) and not args[batch_idx + 1].startswith("--"):
            directory = args[batch_idx + 1]
        else:
            directory = str(SAMPLES_DIR)
        
        # Get language
        lang = DEFAULT_LANG
        if "--lang" in args:
            lang_idx = args.index("--lang")
            if lang_idx + 1 < len(args):
                lang = args[lang_idx + 1]
        
        batch_process(directory, lang, OUTPUTS_DIR)
    
    elif "--help" in args or "-h" in args:
        print("""
📄 Document OCR Extractor

Usage:
  python extractor.py <file_path> [language] [options]
  python extractor.py --batch [directory] [--lang <language>]

Arguments:
  file_path         Path to PDF or image file
  language          OCR language (default: eng)
                    Examples: eng, vie, eng+vie

Options:
  --preprocess, -p  Enable image preprocessing for better accuracy
  --extract "<schema>"  Extract structured data using LLM
                        Output in TOON format for graph databases
  --batch, --all    Process all files in directory
  --lang            Specify language for batch processing

Examples:
  python extractor.py document.pdf
  python extractor.py document.pdf vie
  python extractor.py image.png --preprocess
  python extractor.py invoice.jpg --extract "Extract: company, date, total, items"
  python extractor.py contract.pdf --extract "Extract: parties, effective date, terms"
  python extractor.py --batch data/samples --lang vie

Environment Variables:
  GEMINI_API_KEY    Required for --extract option (get from https://aistudio.google.com/)
        """)
    
    else:
        # Single file mode
        if args:
            file_path = args[0]
        else:
            # Default: find first file in image/ or data/samples/
            image_files = list(IMAGE_DIR.glob('*.*')) if IMAGE_DIR.exists() else []
            if image_files:
                file_path = str(image_files[0])
                print(f"Using first image in image/: {file_path}\n")
            else:
                file_path = str(SAMPLES_DIR / "sample1.pdf")
        
        # Get language (optional)
        lang = args[1] if len(args) > 1 and not args[1].startswith("--") else DEFAULT_LANG
        
        # Check for --extract option
        extract_schema = None
        if "--extract" in args:
            extract_idx = args.index("--extract")
            if extract_idx + 1 < len(args):
                extract_schema = args[extract_idx + 1]
        
        # Process single file with preprocessing option
        result = process_document(file_path, lang=lang, preprocess=preprocess)
        
        if result.full_text:
            print("\n📝 Extracted Text Preview (first 500 chars):")
            print("=" * 60)
            print(result.full_text[:500])
            print("=" * 60)
            
            print(f"\n🔍 Sample Text Blocks (first 10):")
            for i, block in enumerate(result.text_blocks[:10], 1):
                print(f"{i:2}. [{block.confidence:.2%}] {block.text[:60]}")
            
            # Save OCR results
            text_path, json_path = save_results(result, file_path, OUTPUTS_DIR)
            
            if text_path:
                print(f"\n💾 OCR Saved to:")
                print(f"   Text: {text_path} ({text_path.stat().st_size:,} bytes)")
                print(f"   JSON: {json_path} ({json_path.stat().st_size:,} bytes)")
            
            # LLM Schema Extraction (if requested)
            if extract_schema:
                print(f"\n🤖 LLM Schema Extraction...")
                print(f"   Schema: {extract_schema}")
                
                try:
                    from src.llm_extractor import extract_with_llm, parse_toon
                    from config import LLM_MODEL
                    
                    # Convert TextBlock objects to dicts for LLM filtering
                    ocr_blocks = [
                        {
                            'text': block.text,
                            'confidence': block.confidence
                        }
                        for block in result.text_blocks
                    ]
                    
                    toon_result = extract_with_llm(
                        result.full_text, 
                        extract_schema, 
                        model=LLM_MODEL,
                        ocr_blocks=ocr_blocks
                    )
                    
                    if toon_result.startswith("# Error"):
                        print(f"\n⚠ {toon_result}")
                    else:
                        print(f"\n📊 TOON Output:")
                        print("=" * 60)
                        print(toon_result)
                        print("=" * 60)
                        
                        # Parse and show stats
                        parsed = parse_toon(toon_result)
                        print(f"\n📈 Extracted: {len(parsed['entities'])} entities, {len(parsed['relations'])} relations")
                        
                        # Save TOON file
                        input_path = Path(file_path)
                        toon_path = OUTPUTS_DIR / f"{input_path.stem}.toon"
                        with open(toon_path, "w", encoding="utf-8") as f:
                            f.write(toon_result)
                        
                        print(f"\n💾 TOON Saved to: {toon_path}")
                        
                except ImportError as e:
                    print(f"\n⚠ LLM extraction not available: {e}")
                    print("   Run: pip install google-generativeai")

