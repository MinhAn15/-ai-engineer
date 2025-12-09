"""
Document Layout Classifier using Gemini Vision.
Classifies document regions into: text, figure, table, attestation, marginalia.
"""

import os
import io
import base64
from typing import List, Dict, Optional, Tuple
from PIL import Image

# Try to import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠ google-generativeai not installed. Run: pip install google-generativeai")

# Layout types with descriptions for the prompt
LAYOUT_TYPES = {
    "text": "Regular text paragraphs, sentences, or words",
    "figure": "Images, diagrams, charts, drawings, or illustrations",
    "table": "Tabular data with rows and columns",
    "attestation": "Signatures, stamps, seals, or official marks",
    "marginalia": "Handwritten notes, annotations, or marks in margins"
}

# Color mapping for frontend
LAYOUT_COLORS = {
    "text": "#22c55e",        # Green
    "figure": "#3b82f6",      # Blue
    "table": "#f97316",       # Orange
    "attestation": "#a855f7", # Purple
    "marginalia": "#ec4899"   # Pink
}

# Estimated tokens per classification call
TOKENS_PER_CLASSIFICATION = 100


def estimate_classification_tokens(num_segments: int) -> int:
    """Estimate total tokens needed to classify all segments."""
    return num_segments * TOKENS_PER_CLASSIFICATION


def init_gemini() -> bool:
    """Initialize Gemini API if not already done."""
    if not GEMINI_AVAILABLE:
        return False
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠ GEMINI_API_KEY not set")
        return False
    
    genai.configure(api_key=api_key)
    return True


def crop_image_region(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    """Crop a region from an image using bounding box (x, y, width, height)."""
    x, y, w, h = bbox
    # Ensure we don't go out of bounds
    x = max(0, x)
    y = max(0, y)
    right = min(image.width, x + w)
    bottom = min(image.height, y + h)
    
    return image.crop((x, y, right, bottom))


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def classify_single_segment(
    image_crop: Image.Image,
    segment_text: str,
    model: Optional[genai.GenerativeModel] = None
) -> Tuple[str, float]:
    """
    Classify a single segment using Gemini Vision.
    
    Args:
        image_crop: Cropped image of the segment region
        segment_text: OCR text from the segment
        model: Optional pre-initialized Gemini model
    
    Returns:
        Tuple of (layout_type, confidence)
    """
    if not GEMINI_AVAILABLE:
        return "text", 0.5
    
    if model is None:
        if not init_gemini():
            return "text", 0.5
        model = genai.GenerativeModel("gemini-2.0-flash")
    
    # Build classification prompt
    types_desc = "\n".join([f"- {t}: {d}" for t, d in LAYOUT_TYPES.items()])
    
    prompt = f"""Analyze this document region and classify it into ONE of these categories:

{types_desc}

The OCR text from this region is:
"{segment_text[:500] if segment_text else '(no text)'}"

Respond with ONLY a JSON object in this exact format:
{{"type": "<category>", "confidence": <0.0-1.0>}}

Choose the most appropriate category based on the visual appearance and content."""

    try:
        response = model.generate_content([prompt, image_crop])
        result_text = response.text.strip()
        
        # Parse JSON response
        import json
        # Clean up response (remove markdown code blocks if present)
        if "```" in result_text:
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()
        
        result = json.loads(result_text)
        layout_type = result.get("type", "text").lower()
        confidence = float(result.get("confidence", 0.7))
        
        # Validate type
        if layout_type not in LAYOUT_TYPES:
            layout_type = "text"
        
        return layout_type, confidence
        
    except Exception as e:
        print(f"Classification error: {e}")
        return "text", 0.5


def classify_segments_batch(
    image: Image.Image,
    segments: List[Dict],
    progress_callback: Optional[callable] = None
) -> List[Dict]:
    """
    Classify all segments in a document image.
    
    Args:
        image: Full document image
        segments: List of segment dicts with 'bbox' and 'text' keys
        progress_callback: Optional callback(current, total) for progress updates
    
    Returns:
        Segments with added 'type' and 'type_confidence' fields
    """
    if not GEMINI_AVAILABLE or not init_gemini():
        # Return segments with default type
        for seg in segments:
            seg["type"] = "text"
            seg["type_confidence"] = 0.5
            seg["color"] = LAYOUT_COLORS["text"]
        return segments
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    total = len(segments)
    
    for i, segment in enumerate(segments):
        if progress_callback:
            progress_callback(i + 1, total)
        
        bbox = segment.get("bbox")
        text = segment.get("text", "")
        
        if bbox:
            try:
                # Crop the segment region
                crop = crop_image_region(image, tuple(bbox))
                
                # Classify
                layout_type, confidence = classify_single_segment(crop, text, model)
                
                segment["type"] = layout_type
                segment["type_confidence"] = confidence
                segment["color"] = LAYOUT_COLORS.get(layout_type, LAYOUT_COLORS["text"])
                
            except Exception as e:
                print(f"Error classifying segment {segment.get('index')}: {e}")
                segment["type"] = "text"
                segment["type_confidence"] = 0.5
                segment["color"] = LAYOUT_COLORS["text"]
        else:
            segment["type"] = "text"
            segment["type_confidence"] = 0.5
            segment["color"] = LAYOUT_COLORS["text"]
    
    return segments


def get_layout_summary(segments: List[Dict]) -> Dict[str, int]:
    """Get count of each layout type in segments."""
    summary = {t: 0 for t in LAYOUT_TYPES}
    for seg in segments:
        seg_type = seg.get("type", "text")
        if seg_type in summary:
            summary[seg_type] += 1
    return {k: v for k, v in summary.items() if v > 0}
