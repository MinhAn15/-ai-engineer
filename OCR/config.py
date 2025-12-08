"""
Configuration for Document Intelligence OCR system.
"""
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars

# Project root
PROJECT_ROOT = Path(__file__).parent

# Tesseract path (Windows - Orient PC)
TESSERACT_PATH = r"C:\Users\an.ly\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Poppler path (Windows)
POPPLER_PATH = r"C:\Users\an.ly\poppler\Library\bin"

# Directories
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
IMAGE_DIR = PROJECT_ROOT / "image"  # <--- THÊM MỚI
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# OCR settings
DEFAULT_DPI = 150
DEFAULT_LANG = "eng"  # "vie" for Vietnamese, "eng+vie" for both

# Supported file types
SUPPORTED_IMAGE_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
SUPPORTED_PDF_FORMAT = '.pdf'

# LLM Settings (for schema extraction)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = "models/gemini-2.5-flash"  # Stable Gemini 2.5 Flash model

# Verify setup
if __name__ == "__main__":
    print(f"✓ Project root: {PROJECT_ROOT}")
    
    print(f"\n📍 Tesseract Configuration:")
    print(f"   Path: {TESSERACT_PATH}")
    print(f"   Exists: {os.path.exists(TESSERACT_PATH)}")
    
    if os.path.exists(TESSERACT_PATH):
        print(f"   ✓ Tesseract found!")
        import subprocess
        try:
            result = subprocess.run(
                [TESSERACT_PATH, "--version"],
                capture_output=True,
                text=True
            )
            version_line = result.stdout.split('\n')[0]
            print(f"   Version: {version_line}")
            
            lang_result = subprocess.run(
                [TESSERACT_PATH, "--list-langs"],
                capture_output=True,
                text=True
            )
            print(f"   Languages: {lang_result.stdout.strip()}")
            
        except Exception as e:
            print(f"   ⚠ Error: {e}")
    else:
        print(f"   ✗ Tesseract NOT found!")
    
    print(f"\n📍 Poppler Configuration:")
    print(f"   Path: {POPPLER_PATH}")
    print(f"   Exists: {os.path.exists(POPPLER_PATH)}")
    
    print(f"\n📁 Directories:")
    print(f"   Samples: {SAMPLES_DIR} (exists: {SAMPLES_DIR.exists()})")
    print(f"   Images: {IMAGE_DIR} (exists: {IMAGE_DIR.exists()})")
    print(f"   Outputs: {OUTPUTS_DIR} (exists: {OUTPUTS_DIR.exists()})")
