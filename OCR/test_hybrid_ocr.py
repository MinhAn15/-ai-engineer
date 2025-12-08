"""
Validation script for Hybrid OCR feature.
Tests with image/invoice.jpg
"""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from src.hybrid_ocr import HybridOCR

def main():
    print("=" * 60)
    print("HYBRID OCR VALIDATION TEST")
    print("=" * 60)

    # Load test image
    img_path = "image/invoice.jpg"
    print(f"\n📄 Loading: {img_path}")

    # Test 1: Standard mode (no LLM, just Tesseract classification)
    print("\n--- TEST 1: Classification Only (no LLM) ---")
    ocr = HybridOCR(min_confidence=0.75, enable_vision_llm=False)
    result = ocr.process(img_path, lang="eng")

    print(f"\n📊 Results:")
    print(f"   Total blocks: {result.stats['total_blocks']}")
    print(f"   High confidence (>=75%): {result.stats['high_conf_blocks']}")
    print(f"   Low confidence (<75%): {result.stats['low_conf_blocks']}")
    print(f"   Tokens used: {result.stats['tokens_used']}")

    print(f"\n📝 Text preview (first 300 chars):")
    print(result.full_text[:300])

    # Test 2: Hybrid mode (with LLM for low-conf regions)
    print("\n\n--- TEST 2: Hybrid Mode (with Vision LLM) ---")
    ocr_hybrid = HybridOCR(min_confidence=0.75, enable_vision_llm=True)
    result_hybrid = ocr_hybrid.process(img_path, lang="eng")

    print(f"\n📊 Hybrid Results:")
    print(f"   Total blocks: {result_hybrid.stats['total_blocks']}")
    print(f"   High confidence: {result_hybrid.stats['high_conf_blocks']}")  
    print(f"   Low confidence: {result_hybrid.stats['low_conf_blocks']}")
    print(f"   LLM regions processed: {result_hybrid.stats['llm_regions']}")
    print(f"   Tokens used: {result_hybrid.stats['tokens_used']}")

    print(f"\n📝 Hybrid text preview (first 300 chars):")
    print(result_hybrid.full_text[:300])

    # Summary
    print("\n" + "=" * 60)
    print("📈 COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Standard OCR text length: {len(result.full_text)} chars")
    print(f"Hybrid OCR text length: {len(result_hybrid.full_text)} chars")
    print(f"Token cost: {result_hybrid.stats['tokens_used']} tokens")
    
    if result_hybrid.regions_processed:
        print(f"\nRegions that used Vision LLM:")
        for i, region in enumerate(result_hybrid.regions_processed):
            print(f"   Region {i+1}: {region.source} - {region.tokens_used} tokens")

    print("\n✅ VALIDATION COMPLETE")

if __name__ == "__main__":
    main()
