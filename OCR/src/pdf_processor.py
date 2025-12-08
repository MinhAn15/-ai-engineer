"""
PDF processing and OCR module.
"""

import sys
import os
from pathlib import Path
from typing import List

# Add parent dir to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TESSERACT_PATH, POPPLER_PATH, DEFAULT_DPI

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import pypdf

# Set Tesseract command
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Try to import preprocessing module
try:
    from src.image_preprocessing import preprocess_for_ocr
    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False


def extract_text_from_pdf(pdf_path: str) -> str:
    '''Extract text from PDF using pypdf (for text-based PDFs).'''
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text
    except Exception as e:
        print(f"⚠ Error extracting text: {e}")
        return ""


def pdf_to_images(pdf_path: str, dpi: int = DEFAULT_DPI) -> List[Image.Image]:
    '''Convert PDF pages to images.'''
    try:
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            poppler_path=POPPLER_PATH  # Important for Windows
        )
        return images
    except Exception as e:
        print(f"⚠ Error converting PDF to images: {e}")
        print(f"   Check Poppler path: {POPPLER_PATH}")
        return []


# Try to import postprocessing module
try:
    from src.text_postprocessing import clean_ocr_text
    POSTPROCESSING_AVAILABLE = True
except ImportError:
    POSTPROCESSING_AVAILABLE = False


def ocr_image_to_text(
    image: Image.Image, 
    lang: str = "eng",
    preprocess: bool = False,
    preprocess_method: str = "adaptive",
    postprocess: bool = True
) -> str:
    '''
    OCR image to text using Tesseract.
    
    Args:
        image: PIL Image object
        lang: Tesseract language code
        preprocess: Enable preprocessing for better accuracy
        preprocess_method: 'simple', 'adaptive', or 'advanced'
        postprocess: Enable text cleanup after OCR
    '''
    try:
        # Apply preprocessing if enabled
        if preprocess and PREPROCESSING_AVAILABLE:
            image = preprocess_for_ocr(image, method=preprocess_method)
        
        text = pytesseract.image_to_string(image, lang=lang)
        
        # Apply postprocessing if enabled
        if postprocess and POSTPROCESSING_AVAILABLE:
            text = clean_ocr_text(text)
        
        return text
    except Exception as e:
        print(f"⚠ OCR error: {e}")
        return ""


def ocr_image_to_blocks(
    image: Image.Image, 
    lang: str = "eng",
    preprocess: bool = False,
    preprocess_method: str = "adaptive"
) -> List[dict]:
    '''
    OCR image and return text blocks with bounding boxes.
    
    Args:
        image: PIL Image object
        lang: Tesseract language code
        preprocess: Enable preprocessing for better accuracy
        preprocess_method: 'simple', 'adaptive', or 'advanced'
    '''
    try:
        # Apply preprocessing if enabled
        if preprocess and PREPROCESSING_AVAILABLE:
            image = preprocess_for_ocr(image, method=preprocess_method)
        
        data = pytesseract.image_to_data(
            image, lang=lang, output_type=pytesseract.Output.DICT
        )
        
        blocks = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            if not text:
                continue
            
            conf = float(data["conf"][i])
            if conf < 0:  # Tesseract uses -1 for noise
                continue
            
            blocks.append({
                "text": text,
                "confidence": conf / 100.0,
                "bbox": (
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i]
                )
            })
        
        return blocks
    except Exception as e:
        print(f"⚠ OCR blocks error: {e}")
        return []


if __name__ == "__main__":
    print("✓ PDF processor module loaded")
    print(f"✓ Tesseract: {TESSERACT_PATH}")
    print(f"✓ Poppler: {POPPLER_PATH}")
    print(f"✓ Preprocessing available: {PREPROCESSING_AVAILABLE}")

