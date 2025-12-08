"""
Hybrid OCR Pipeline: Combines Tesseract OCR with Vision LLM for optimal accuracy.

Strategy:
- High confidence regions (≥75%): Trust Tesseract result (FREE)
- Low confidence regions (<75%): Crop and send to Vision LLM (PAID)

Region Grouping Algorithm:
- Uses DBSCAN-inspired spatial clustering to group nearby low-confidence blocks
- Merges overlapping/adjacent bounding boxes using Union-Find
- Based on document layout analysis research (Breuel, 2002; Meunier, 2005)

References:
- DBSCAN clustering: Ester et al., "A density-based algorithm for discovering clusters"
- Document layout: Breuel, "Two Geometric Algorithms for Layout Analysis"
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from PIL import Image
import io
import base64

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pdf_processor import ocr_image_to_blocks, ocr_image_to_text
from config import GEMINI_API_KEY, LLM_MODEL


@dataclass
class TextBlock:
    """A text block with position and confidence."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (left, top, width, height)
    source: str = "tesseract"  # "tesseract" or "vision_llm"
    
    @property
    def left(self) -> int:
        return self.bbox[0]
    
    @property
    def top(self) -> int:
        return self.bbox[1]
    
    @property
    def width(self) -> int:
        return self.bbox[2]
    
    @property
    def height(self) -> int:
        return self.bbox[3]
    
    @property
    def right(self) -> int:
        return self.left + self.width
    
    @property
    def bottom(self) -> int:
        return self.top + self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)
    
    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class Region:
    """A merged region of multiple low-confidence blocks."""
    blocks: List[TextBlock]
    bbox: Tuple[int, int, int, int]
    text: str = ""
    source: str = "pending"
    tokens_used: int = 0


@dataclass
class HybridOCRResult:
    """Result from hybrid OCR processing."""
    full_text: str
    blocks: List[TextBlock]
    stats: Dict
    regions_processed: List[Region] = field(default_factory=list)


class HybridOCR:
    """
    Hybrid OCR combining Tesseract with Vision LLM.
    
    High-confidence regions use Tesseract (free).
    Low-confidence regions use Vision LLM (paid but accurate).
    """
    
    def __init__(
        self,
        min_confidence: float = 0.75,
        region_gap_threshold: int = 50,  # pixels
        min_region_area: int = 100,  # minimum pixels to bother with LLM
        api_key: Optional[str] = None,
        model: str = None,
        max_tokens_per_doc: Optional[int] = None,  # Future: cost limiting
        enable_vision_llm: bool = True
    ):
        """
        Initialize Hybrid OCR.
        
        Args:
            min_confidence: Threshold for trusting Tesseract (0.0-1.0)
            region_gap_threshold: Max gap (px) to merge nearby low-conf blocks
            min_region_area: Minimum region area to send to LLM
            api_key: Gemini API key
            model: Vision model to use
            max_tokens_per_doc: Future cost limiting
            enable_vision_llm: If False, skip LLM and use Tesseract only
        """
        self.min_confidence = min_confidence
        self.region_gap_threshold = region_gap_threshold
        self.min_region_area = min_region_area
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model or LLM_MODEL
        self.max_tokens_per_doc = max_tokens_per_doc
        self.enable_vision_llm = enable_vision_llm
        
        # Track token usage for cost limiting
        self._tokens_used = 0
    
    def process(
        self,
        image: Union[str, Path, Image.Image],
        lang: str = "eng"
    ) -> HybridOCRResult:
        """
        Process image with hybrid OCR pipeline.
        
        Args:
            image: Path to image or PIL Image
            lang: Tesseract language code
            
        Returns:
            HybridOCRResult with merged text and statistics
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image)
        else:
            pil_image = image
        
        # Reset token counter for this document
        self._tokens_used = 0
        
        # ========== PHASE 1: Tesseract OCR ==========
        print("   📝 Phase 1: Running Tesseract OCR...")
        raw_blocks = ocr_image_to_blocks(pil_image, lang=lang)
        
        all_blocks = [
            TextBlock(
                text=b['text'],
                confidence=b['confidence'],
                bbox=b['bbox'],
                source="tesseract"
            )
            for b in raw_blocks
        ]
        
        # ========== PHASE 2: Classify by Confidence ==========
        print(f"   🔍 Phase 2: Classifying {len(all_blocks)} blocks by confidence (threshold: {self.min_confidence:.0%})...")
        
        high_conf_blocks = []
        low_conf_blocks = []
        
        for block in all_blocks:
            if block.confidence >= self.min_confidence:
                high_conf_blocks.append(block)
            else:
                low_conf_blocks.append(block)
        
        print(f"      ✓ High confidence (≥{self.min_confidence:.0%}): {len(high_conf_blocks)} blocks")
        print(f"      ✓ Low confidence (<{self.min_confidence:.0%}): {len(low_conf_blocks)} blocks")
        
        # ========== PHASE 3: Group Low-Conf Regions ==========
        regions = []
        if low_conf_blocks and self.enable_vision_llm:
            print(f"   🔗 Phase 3: Clustering low-confidence regions...")
            regions = self._cluster_regions(low_conf_blocks)
            print(f"      ✓ Merged into {len(regions)} regions")
        
        # ========== PHASE 4: Vision LLM for Low-Conf Regions ==========
        llm_blocks = []
        if regions and self.enable_vision_llm:
            print(f"   🤖 Phase 4: Processing {len(regions)} regions with Vision LLM...")
            for i, region in enumerate(regions):
                # Skip tiny regions
                if region.bbox[2] * region.bbox[3] < self.min_region_area:
                    print(f"      ⏭ Region {i+1}: Skipped (too small)")
                    # Fallback to Tesseract text
                    region.text = " ".join([b.text for b in region.blocks])
                    region.source = "tesseract_fallback"
                    continue
                
                try:
                    # Crop image region
                    cropped = self._crop_region(pil_image, region.bbox)
                    
                    # Call Vision LLM
                    llm_text, tokens = self._call_vision_llm(cropped)
                    
                    region.text = llm_text
                    region.source = "vision_llm"
                    region.tokens_used = tokens
                    self._tokens_used += tokens
                    
                    print(f"      ✓ Region {i+1}: {tokens} tokens used")
                    
                except Exception as e:
                    print(f"      ⚠ Region {i+1}: LLM failed - {e}")
                    # Fallback to Tesseract text
                    region.text = " ".join([b.text for b in region.blocks])
                    region.source = "tesseract_fallback"
                
                # Convert region to block for merging
                llm_blocks.append(TextBlock(
                    text=region.text,
                    confidence=0.95 if region.source == "vision_llm" else 0.5,
                    bbox=region.bbox,
                    source=region.source
                ))
        elif low_conf_blocks:
            # Vision LLM disabled - just use Tesseract results
            llm_blocks = low_conf_blocks
        
        # ========== PHASE 5: Merge All Blocks ==========
        print(f"   📋 Phase 5: Merging results...")
        final_text, merged_blocks = self._merge_blocks(high_conf_blocks, llm_blocks)
        
        # Build result
        stats = {
            "total_blocks": len(all_blocks),
            "high_conf_blocks": len(high_conf_blocks),
            "low_conf_blocks": len(low_conf_blocks),
            "llm_regions": len(regions),
            "tokens_used": self._tokens_used,
            "confidence_threshold": self.min_confidence,
            "tesseract_only": not self.enable_vision_llm
        }
        
        print(f"   ✅ Complete: {len(merged_blocks)} final blocks, {self._tokens_used} tokens used")
        
        return HybridOCRResult(
            full_text=final_text,
            blocks=merged_blocks,
            stats=stats,
            regions_processed=regions
        )
    
    def _cluster_regions(self, blocks: List[TextBlock]) -> List[Region]:
        """
        Cluster nearby low-confidence blocks into regions using spatial proximity.
        
        Algorithm: DBSCAN-inspired density-based clustering
        - Blocks within gap_threshold are considered neighbors
        - Connected components form regions
        
        Based on:
        - Breuel, T. "Two Geometric Algorithms for Layout Analysis"
        - Uses Union-Find for efficient merging
        """
        if not blocks:
            return []
        
        n = len(blocks)
        parent = list(range(n))  # Union-Find parent array
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Check proximity between all pairs
        threshold = self.region_gap_threshold
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._blocks_are_close(blocks[i], blocks[j], threshold):
                    union(i, j)
        
        # Group blocks by their root
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(blocks[i])
        
        # Create regions from groups
        regions = []
        for group_blocks in groups.values():
            # Calculate merged bounding box
            left = min(b.left for b in group_blocks)
            top = min(b.top for b in group_blocks)
            right = max(b.right for b in group_blocks)
            bottom = max(b.bottom for b in group_blocks)
            
            # Expand bbox slightly for context (5px padding)
            padding = 5
            left = max(0, left - padding)
            top = max(0, top - padding)
            right += padding
            bottom += padding
            
            regions.append(Region(
                blocks=group_blocks,
                bbox=(left, top, right - left, bottom - top)
            ))
        
        return regions
    
    def _blocks_are_close(self, b1: TextBlock, b2: TextBlock, threshold: int) -> bool:
        """
        Check if two blocks are within proximity threshold.
        Uses gap-based distance (minimum edge-to-edge distance).
        """
        # Horizontal gap
        h_gap = max(0, max(b1.left, b2.left) - min(b1.right, b2.right))
        
        # Vertical gap
        v_gap = max(0, max(b1.top, b2.top) - min(b1.bottom, b2.bottom))
        
        # Blocks are close if gaps are small
        return h_gap <= threshold and v_gap <= threshold
    
    def _crop_region(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
        """Crop image to region bbox with bounds checking."""
        left, top, width, height = bbox
        right = min(left + width, image.width)
        bottom = min(top + height, image.height)
        left = max(0, left)
        top = max(0, top)
        
        return image.crop((left, top, right, bottom))
    
    def _call_vision_llm(self, image: Image.Image) -> Tuple[str, int]:
        """
        Call Vision LLM to extract text from image crop.
        
        Returns:
            Tuple of (extracted_text, tokens_used)
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai not installed")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Use vision model
        model = genai.GenerativeModel(self.model)
        
        # Create prompt for OCR
        prompt = """Extract ALL text from this image exactly as shown.
Rules:
- Preserve original formatting and line breaks
- Include all visible text, even if partially visible
- Do not add any commentary or explanation
- Just output the raw text content"""
        
        # Send image to model
        response = model.generate_content([prompt, image])
        
        # Extract text
        text = response.text.strip() if response.text else ""
        
        # Estimate tokens (approximate)
        input_tokens = 256  # Image token approximation
        output_tokens = int(len(text.split()) * 1.3)
        total_tokens = input_tokens + output_tokens
        
        return text, total_tokens
    
    def _merge_blocks(
        self,
        high_conf: List[TextBlock],
        low_conf: List[TextBlock]
    ) -> Tuple[str, List[TextBlock]]:
        """
        Merge high and low confidence blocks in reading order.
        
        Uses document layout analysis principles:
        - Sort by vertical position (top to bottom)
        - Within similar Y, sort by horizontal (left to right)
        - Group into lines based on Y proximity
        """
        all_blocks = high_conf + low_conf
        
        if not all_blocks:
            return "", []
        
        # Sort by reading order: top-to-bottom, left-to-right
        # Use line grouping (blocks within 20px vertical are same line)
        line_threshold = 20
        
        def reading_order_key(block):
            return (block.top // line_threshold, block.left)
        
        all_blocks.sort(key=reading_order_key)
        
        # Remove duplicates based on overlapping bboxes
        merged = []
        for block in all_blocks:
            is_duplicate = False
            for existing in merged:
                if self._overlap_ratio(block.bbox, existing.bbox) > 0.5:
                    # Keep the one with higher confidence
                    if block.confidence > existing.confidence:
                        merged.remove(existing)
                    else:
                        is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(block)
        
        # Build text with line awareness
        lines = []
        current_line = []
        current_y = -1000
        
        for block in merged:
            if abs(block.top - current_y) > line_threshold:
                if current_line:
                    lines.append(" ".join([b.text for b in current_line]))
                current_line = [block]
                current_y = block.top
            else:
                current_line.append(block)
        
        if current_line:
            lines.append(" ".join([b.text for b in current_line]))
        
        full_text = "\n".join(lines)
        
        return full_text, merged
    
    def _overlap_ratio(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate overlap ratio between two bboxes."""
        l1, t1, w1, h1 = box1
        l2, t2, w2, h2 = box2
        
        # Intersection
        x_left = max(l1, l2)
        y_top = max(t1, t2)
        x_right = min(l1 + w1, l2 + w2)
        y_bottom = min(t1 + h1, t2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        
        if area1 == 0 or area2 == 0:
            return 0.0
        
        # Overlap relative to smaller box
        return intersection / min(area1, area2)


# Convenience function
def hybrid_ocr(
    image: Union[str, Path, Image.Image],
    lang: str = "eng",
    min_confidence: float = 0.75,
    enable_vision_llm: bool = True
) -> HybridOCRResult:
    """
    Convenience function for hybrid OCR.
    
    Args:
        image: Image path or PIL Image
        lang: OCR language
        min_confidence: Confidence threshold
        enable_vision_llm: Enable Vision LLM for low-conf regions
        
    Returns:
        HybridOCRResult
    """
    processor = HybridOCR(
        min_confidence=min_confidence,
        enable_vision_llm=enable_vision_llm
    )
    return processor.process(image, lang=lang)


if __name__ == "__main__":
    print("=" * 60)
    print("Hybrid OCR Module")
    print("=" * 60)
    print(f"✓ Min confidence threshold: 75%")
    print(f"✓ Region clustering: DBSCAN-inspired spatial clustering")
    print(f"✓ Vision LLM: Gemini")
    print(f"✓ Fallback: Tesseract on LLM failure")
    print()
    print("Usage:")
    print("  from src.hybrid_ocr import HybridOCR")
    print("  ocr = HybridOCR(min_confidence=0.75)")
    print("  result = ocr.process('image.png')")
    print("  print(result.full_text)")
