"""
Document Layout Analysis - Hybrid Approach
Tries LayoutLMv3 first, falls back to Gemini Vision if unavailable.
"""
from PIL import Image, ImageDraw
from typing import List, Dict, Optional, Tuple
import os

# Layout type colors
LAYOUT_COLORS = {
    "text": "#22c55e",        # Green
    "title": "#ef4444",       # Red
    "table": "#f97316",       # Orange
    "figure": "#3b82f6",      # Blue
    "list": "#eab308",        # Yellow
    "attestation": "#a855f7", # Purple
    "marginalia": "#ec4899",  # Pink
    "form": "#06b6d4",        # Cyan
    "checkbox": "#84cc16",    # Lime
    "signature": "#a855f7",   # Purple (same as attestation)
}

# Try to import LayoutLMv3
LAYOUTLM_AVAILABLE = False
try:
    from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
    import torch
    LAYOUTLM_AVAILABLE = True
    print("✓ LayoutLMv3 available")
except ImportError:
    print("⚠ LayoutLMv3 not available, will use Gemini Vision fallback")

# Try to import Gemini
GEMINI_AVAILABLE = False
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("⚠ Gemini not available")


# Try to import Layout Foundation Model
FOUNDATION_MODEL_AVAILABLE = False
try:
    from src.layout_foundation_model import LayoutFoundationModel, get_layout_foundation_model
    FOUNDATION_MODEL_AVAILABLE = True
except ImportError:
    pass


class LayoutAnalyzer:
    """
    Hybrid layout analyzer - uses best available method.
    
    Priority order:
    1. Layout Foundation Model (DocLayNet-style, like LandingAI)
    2. LayoutLMv3 (transformer-based)
    3. Gemini Vision (LLM-based)
    4. Heuristics (fallback)
    """
    
    # PubLayNet label mapping
    PUBLAYNET_LABELS = {
        0: "text",
        1: "title",
        2: "list",
        3: "table",
        4: "figure"
    }
    
    def __init__(self, prefer_foundation: bool = True, prefer_layoutlm: bool = True):
        """
        Initialize analyzer.
        
        Args:
            prefer_foundation: If True, try Layout Foundation Model first
            prefer_layoutlm: If True, try LayoutLMv3 second
        """
        self.use_foundation = prefer_foundation and FOUNDATION_MODEL_AVAILABLE
        self.use_layoutlm = prefer_layoutlm and LAYOUTLM_AVAILABLE
        self.use_gemini = GEMINI_AVAILABLE
        
        self.foundation_model = None
        self.model = None
        self.processor = None
        self.device = None
        self.gemini_model = None
        
        # Initialize best available method
        if self.use_foundation:
            self._init_foundation()
        
        if not self.foundation_model and self.use_layoutlm:
            self._init_layoutlm()
        elif not self.foundation_model and self.use_gemini:
            self._init_gemini()
    
    def _init_foundation(self):
        """Initialize Layout Foundation Model (DocLayNet-style)."""
        try:
            self.foundation_model = get_layout_foundation_model()
            if self.foundation_model.is_initialized:
                print("[+] Using Layout Foundation Model (DocLayNet-style)")
            else:
                self.foundation_model = None
        except Exception as e:
            print(f"[!] Foundation model init failed: {e}")
            self.foundation_model = None
    
    def _init_layoutlm(self):
        """Initialize LayoutLMv3."""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                "microsoft/layoutlmv3-base",
                num_labels=5  # PubLayNet labels
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ LayoutLMv3 initialized on {self.device}")
        except Exception as e:
            print(f"⚠ LayoutLMv3 init failed: {e}")
            self.use_layoutlm = False
            if self.use_gemini:
                self._init_gemini()
    
    def _init_gemini(self):
        """Initialize Gemini Vision."""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel("gemini-2.0-flash")
                print("✓ Gemini Vision initialized")
            else:
                print("⚠ GEMINI_API_KEY not set")
                self.use_gemini = False
        except Exception as e:
            print(f"⚠ Gemini init failed: {e}")
            self.use_gemini = False
    
    def analyze(
        self,
        image: Image.Image,
        segments: List[Dict]
    ) -> List[Dict]:
        """
        Analyze document layout and classify segments.
        
        Args:
            image: PIL Image
            segments: List of dicts with 'bbox', 'text', 'index', 'confidence'
        
        Returns:
            Segments with added 'type', 'type_confidence', 'color', 'reading_order', 'column'
        """
        # Step 1: Detect reading order (LandingAI-style)
        from src.reading_order_detector import get_reading_order_detector
        order_detector = get_reading_order_detector()
        segments = order_detector.detect_reading_order(segments)
        
        # Step 2: Classify layout types (Priority: Foundation > LayoutLMv3 > Gemini > Heuristic)
        if self.foundation_model and self.foundation_model.is_initialized:
            segments = self._analyze_foundation(image, segments)
        elif self.use_layoutlm and self.model:
            segments = self._analyze_layoutlm(image, segments)
        elif self.use_gemini and self.gemini_model:
            segments = self._analyze_gemini(image, segments)
        else:
            segments = self._analyze_heuristic(segments)
        
        # Step 3: Sort by reading order
        segments.sort(key=lambda s: s.get('reading_order', 0))
        
        return segments
    
    def _analyze_foundation(
        self,
        image: Image.Image,
        segments: List[Dict]
    ) -> List[Dict]:
        """
        Use Layout Foundation Model (DocLayNet-style) for classification.
        
        Detects layout regions first, then assigns types to segments based on overlap.
        """
        if not segments or not self.foundation_model:
            return segments
        
        # Get layout regions from foundation model
        layout_regions = self.foundation_model.analyze_layout(image)
        
        if not layout_regions:
            # Fallback to heuristic if no regions detected
            return self._analyze_heuristic(segments)
        
        # Assign types to segments based on overlap with detected regions
        for segment in segments:
            seg_bbox = segment.get('bbox', [0, 0, 0, 0])
            best_match = None
            best_overlap = 0.0
            
            for region in layout_regions:
                overlap = self._calculate_overlap(seg_bbox, region['bbox'])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = region
            
            if best_match and best_overlap > 0.3:
                segment['type'] = best_match['type']
                segment['type_confidence'] = best_match['confidence']
                segment['color'] = best_match['color']
                segment['detection_method'] = 'foundation_model'
            else:
                # No good match - use heuristic
                self._apply_heuristic_type(segment)
        
        return segments
    
    def _calculate_overlap(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate IoU overlap between two bboxes (x, y, w, h)."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        
        if area1 == 0:
            return 0.0
        
        # Return overlap relative to segment (not full IoU)
        return intersection / area1
    
    def _analyze_layoutlm(
        self,
        image: Image.Image,
        segments: List[Dict]
    ) -> List[Dict]:
        """Use LayoutLMv3 for classification."""
        if not segments:
            return segments
        
        width, height = image.size
        
        # Prepare inputs
        texts = [seg.get("text", "") for seg in segments]
        boxes = []
        for seg in segments:
            x, y, w, h = seg.get("bbox", [0, 0, 0, 0])
            # Normalize to 0-1000
            boxes.append([
                min(1000, int(1000 * x / width)),
                min(1000, int(1000 * y / height)),
                min(1000, int(1000 * (x + w) / width)),
                min(1000, int(1000 * (y + h) / height))
            ])
        
        try:
            # Encode
            encoding = self.processor(
                image,
                text=texts,
                boxes=boxes,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            
            # Forward
            with torch.no_grad():
                outputs = self.model(**encoding)
            
            # Get predictions
            preds = outputs.logits.argmax(-1).squeeze().tolist()
            confs = torch.softmax(outputs.logits, dim=-1).max(-1).values.squeeze().tolist()
            
            if isinstance(preds, int):
                preds = [preds]
            if isinstance(confs, float):
                confs = [confs]
            
            # Map to segments
            for i, seg in enumerate(segments):
                if i < len(preds):
                    label = self.PUBLAYNET_LABELS.get(preds[i], "text")
                    conf = confs[i] if i < len(confs) else 0.5
                else:
                    label = "text"
                    conf = 0.5
                
                seg["type"] = label
                seg["type_confidence"] = float(conf)
                seg["color"] = LAYOUT_COLORS.get(label, "#6b7280")
            
            return segments
            
        except Exception as e:
            print(f"LayoutLMv3 error: {e}")
            return self._analyze_heuristic(segments)
    
    def _analyze_gemini(
        self,
        image: Image.Image,
        segments: List[Dict]
    ) -> List[Dict]:
        """Use Gemini Vision for classification."""
        if not segments:
            return segments
        
        # Build classification prompt
        segment_texts = "\n".join([
            f"[{s['index']}] {s.get('text', '')[:100]}"
            for s in segments[:50]  # Limit to 50
        ])
        
        prompt = f"""Analyze this document image and classify each text segment.

Segments to classify:
{segment_texts}

For each segment, determine its type from these categories:
- text: Regular paragraph or sentence
- title: Headers, titles, section names
- table: Part of a table structure
- figure: Image, diagram, or chart caption
- form: Form fields, input areas
- list: List items, bullet points
- checkbox: Checkbox or radio button labels
- signature: Signature area
- attestation: Official stamps, seals

Respond with ONLY a JSON array like:
[{{"index": 1, "type": "title"}}, {{"index": 2, "type": "text"}}, ...]

Be accurate - look at the visual structure not just text content."""

        try:
            response = self.gemini_model.generate_content([prompt, image])
            result_text = response.text.strip()
            
            # Parse JSON
            import json
            if "```" in result_text:
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()
            
            classifications = json.loads(result_text)
            
            # Map to segments
            type_map = {c["index"]: c["type"] for c in classifications}
            
            for seg in segments:
                seg_type = type_map.get(seg["index"], "text")
                seg["type"] = seg_type
                seg["type_confidence"] = 0.8
                seg["color"] = LAYOUT_COLORS.get(seg_type, LAYOUT_COLORS["text"])
            
            return segments
            
        except Exception as e:
            print(f"Gemini error: {e}")
            return self._analyze_heuristic(segments)
    
    def _analyze_heuristic(self, segments: List[Dict]) -> List[Dict]:
        """Simple heuristic classification based on text patterns."""
        for seg in segments:
            text = seg.get("text", "").strip()
            seg_type = "text"
            
            # Simple heuristics
            if len(text) < 5 and text.isupper():
                seg_type = "title"
            elif text.startswith(("•", "-", "→", "○", "□", "☐")):
                seg_type = "list"
            elif any(x in text.lower() for x in ["signature", "sign", "chữ ký"]):
                seg_type = "attestation"
            elif "|" in text or "\t" in text:
                seg_type = "table"
            elif len(text.split()) <= 3 and text.isupper():
                seg_type = "title"
            
            seg["type"] = seg_type
            seg["type_confidence"] = 0.5
            seg["color"] = LAYOUT_COLORS.get(seg_type, "#6b7280")
        
        return segments
    
    def _apply_heuristic_type(self, segment: Dict) -> None:
        """Apply heuristic type classification to a single segment (mutates segment)."""
        text = segment.get("text", "").strip()
        seg_type = "text"
        
        # Simple heuristics
        if len(text) < 5 and text.isupper():
            seg_type = "title"
        elif text.startswith(("•", "-", "→", "○", "□", "☐")):
            seg_type = "list"
        elif any(x in text.lower() for x in ["signature", "sign", "chữ ký"]):
            seg_type = "attestation"
        elif "|" in text or "\t" in text:
            seg_type = "table"
        elif len(text.split()) <= 3 and text.isupper():
            seg_type = "title"
        
        segment["type"] = seg_type
        segment["type_confidence"] = 0.5
        segment["color"] = LAYOUT_COLORS.get(seg_type, "#6b7280")
        segment["detection_method"] = "heuristic"


# Singleton
_analyzer: Optional[LayoutAnalyzer] = None


def get_layout_analyzer() -> LayoutAnalyzer:
    """Get or create singleton analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = LayoutAnalyzer()
    return _analyzer


def classify_segments(
    image: Image.Image,
    segments: List[Dict]
) -> List[Dict]:
    """
    Classify segments using multi-stage approach:
    1. Visual table detection (OpenCV grid lines)
    2. Checkbox detection
    3. ML/LLM classification for remaining segments
    
    Args:
        image: PIL Image
        segments: List of segment dicts
    
    Returns:
        Segments with type, type_confidence, color added
    """
    # Import table detector
    try:
        from src.table_detector import get_table_detector
        table_detector = get_table_detector()
        
        # Stage 1: Detect tables visually
        tables = table_detector.detect_tables(image)
        
        # Stage 2: Detect tables by alignment
        tables_by_alignment = table_detector.detect_tables_by_alignment(segments)
        
        # Merge table regions
        all_tables = tables + [t for t in tables_by_alignment if not _overlaps_any(t['bbox'], [tb['bbox'] for tb in tables])]
        
        # Stage 3: Detect checkboxes
        checkboxes = table_detector.detect_checkboxes(image)
        
        # Mark segments that fall inside detected tables/checkboxes
        for seg in segments:
            seg_bbox = seg.get('bbox', [0, 0, 0, 0])
            seg_center = (seg_bbox[0] + seg_bbox[2]/2, seg_bbox[1] + seg_bbox[3]/2)
            
            # Check if inside any table
            for table in all_tables:
                tx, ty, tw, th = table['bbox']
                if tx <= seg_center[0] <= tx + tw and ty <= seg_center[1] <= ty + th:
                    seg['type'] = 'table'
                    seg['type_confidence'] = table.get('confidence', 0.85)
                    seg['color'] = LAYOUT_COLORS.get('table', '#f97316')
                    seg['_classified'] = True
                    break
            
            # Check if near any checkbox
            if not seg.get('_classified'):
                for cb in checkboxes:
                    if _is_adjacent(seg_bbox, cb['bbox'], threshold=30):
                        seg['type'] = 'checkbox'
                        seg['type_confidence'] = cb.get('confidence', 0.7)
                        seg['color'] = LAYOUT_COLORS.get('checkbox', '#84cc16')
                        seg['_classified'] = True
                        break
        
    except ImportError:
        print("⚠ Table detector not available")
        all_tables = []
    except Exception as e:
        print(f"⚠ Table detection error: {e}")
        all_tables = []
    
    # Stage 4: Classify remaining segments with ML/LLM
    unclassified = [s for s in segments if not s.get('_classified')]
    
    if unclassified:
        analyzer = get_layout_analyzer()
        classified = analyzer.analyze(image, unclassified)
        
        # Merge back
        classified_map = {s['index']: s for s in classified}
        for seg in segments:
            if not seg.get('_classified') and seg['index'] in classified_map:
                c = classified_map[seg['index']]
                seg['type'] = c.get('type', 'text')
                seg['type_confidence'] = c.get('type_confidence', 0.5)
                seg['color'] = c.get('color', LAYOUT_COLORS['text'])
    
    # Clean up temp flag
    for seg in segments:
        seg.pop('_classified', None)
    
    return segments


def _overlaps_any(bbox: List[int], others: List[List[int]], threshold: float = 0.5) -> bool:
    """Check if bbox overlaps significantly with any other bbox."""
    x1, y1, w1, h1 = bbox
    area1 = w1 * h1
    
    for other in others:
        x2, y2, w2, h2 = other
        
        # Calculate intersection
        ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        intersection = ix * iy
        
        if area1 > 0 and intersection / area1 > threshold:
            return True
    
    return False


def _is_adjacent(bbox1: List[int], bbox2: List[int], threshold: int = 50) -> bool:
    """Check if two bboxes are adjacent (within threshold pixels)."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Check horizontal adjacency
    horizontal_gap = max(x1 - (x2 + w2), x2 - (x1 + w1))
    
    # Check vertical overlap
    y_overlap = not (y1 + h1 < y2 or y2 + h2 < y1)
    
    return horizontal_gap < threshold and y_overlap


def get_layout_summary(segments: List[Dict]) -> Dict[str, int]:
    """Get count of each layout type."""
    summary = {}
    for seg in segments:
        seg_type = seg.get("type", "text")
        summary[seg_type] = summary.get(seg_type, 0) + 1
    return summary


def visualize_layout(image: Image.Image, segments: List[Dict]) -> Image.Image:
    """Draw classification boxes on image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    for seg in segments:
        bbox = seg.get("bbox", [0, 0, 0, 0])
        x, y, w, h = bbox
        color = seg.get("color", "#22c55e")
        label = seg.get("type", "text")
        conf = seg.get("type_confidence", 0)
        
        # Rectangle
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        
        # Label
        label_text = f"{label} ({conf:.0%})"
        draw.text((x, max(0, y - 12)), label_text, fill=color)
    
    return img
