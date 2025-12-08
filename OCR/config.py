"""
Configuration for Document Intelligence OCR system.
"""
import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

# Tesseract path (Windows - Orient PC)
TESSERACT_PATH = r"C:\Users\an.ly\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Poppler path (Windows)
# Cập nhật path này sau khi giải nén Poppler
POPPLER_PATH = r"C:\Users\an.ly\poppler\poppler-24.08.0\Library\bin"

# Directories
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# OCR settings
DEFAULT_DPI = 150
DEFAULT_LANG = "eng"  # "vie" for Vietnamese, "eng+vie" for both

# Verify setup
if __name__ == "__main__":
    print(f"✓ Project root: {PROJECT_ROOT}")
    
    print(f"\n📍 Tesseract Configuration:")
    print(f"   Path: {TESSERACT_PATH}")
    print(f"   Exists: {os.path.exists(TESSERACT_PATH)}")
    
    if os.path.exists(TESSERACT_PATH):
        print(f"   ✓ Tesseract found!")
        # Test Tesseract version
        import subprocess
        try:
            result = subprocess.run(
                [TESSERACT_PATH, "--version"],
                capture_output=True,
                text=True
            )
            version_line = result.stdout.split('\n')[0]
            print(f"   Version: {version_line}")
            
            # Check languages
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
    print(f"   Samples: {SAMPLES_DIR}")
    print(f"   Outputs: {OUTPUTS_DIR}")
