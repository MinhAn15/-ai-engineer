"""
Layout Foundation Model - DocLayNet-style layout detection
Similar to LandingAI's approach using pre-trained models for document layout analysis.

Supports:
- microsoft/table-transformer-detection (table-specific)
- facebook/detr-resnet-50 (general object detection)
"""
from typing import List, Dict, Optional, Tuple
from PIL import Image
import os
import sys

# Add parent to path
sys.path.insert(0, str(os.path.dirname(os.path.dirname(__file__))))


class LayoutFoundationModel:
    """
    DocLayNet-based layout detection (similar to LandingAI)
    
    Detects:
    - Text blocks (green)
    - Tables (blue)
    - Figures (purple)  
    - Headers/Titles
    - Lists
    - Formulas
    """
    
    # Model options
    MODELS = {
        'table-transformer': 'microsoft/table-transformer-detection',
        'table-structure': 'microsoft/table-transformer-structure-recognition',
        'detr': 'facebook/detr-resnet-50',
    }
    
    # Color mapping (LandingAI style)
    COLOR_MAP = {
        'text': '#22c55e',      # Green
        'title': '#ef4444',     # Red  
        'table': '#3b82f6',     # Blue
        'figure': '#a855f7',    # Purple
        'list': '#eab308',      # Yellow
        'formula': '#f97316',   # Orange
        'caption': '#ec4899',   # Pink
        'header': '#ef4444',    # Red
        'footer': '#6b7280',    # Gray
    }
    
    def __init__(self, model_name: str = "table-transformer", threshold: float = 0.7):
        """
        Initialize foundation model
        
        Args:
            model_name: One of 'table-transformer', 'table-structure', 'detr'
            threshold: Confidence threshold for detections
        """
        self.model = None
        self.processor = None
        self.device = None
        self.threshold = threshold
        self.model_name = model_name
        self.is_initialized = False
        
        # Lazy initialization
        self._init_model(model_name)
    
    def _init_model(self, model_name: str):
        """Initialize the model (lazy loading)"""
        try:
            import torch
            from transformers import AutoModelForObjectDetection, AutoImageProcessor
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Get model path
            model_path = self.MODELS.get(model_name, model_name)
            
            print(f"[*] Loading layout model: {model_path}")
            
            # Load model and processor
            self.model = AutoModelForObjectDetection.from_pretrained(model_path)
            self.processor = AutoImageProcessor.from_pretrained(model_path)
            
            self.model.to(self.device)
            self.model.eval()
            
            self.is_initialized = True
            print(f"[+] Layout Foundation Model initialized on {self.device}")
            
        except ImportError as e:
            print(f"[!] transformers/torch not available: {e}")
            self.is_initialized = False
        except Exception as e:
            print(f"[!] Failed to load layout model: {e}")
            self.is_initialized = False
    
    def analyze_layout(self, image: Image.Image) -> List[Dict]:
        """
        Analyze document layout with foundation model
        
        Args:
            image: PIL Image
        
        Returns:
            List of layout chunks with type, bbox, confidence, color
        """
        if not self.is_initialized:
            return []
        
        import torch
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Prepare inputs
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=self.threshold,
            target_sizes=target_sizes
        )[0]
        
        # Extract chunks
        chunks = []
        for score, label, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"]
        ):
            label_name = self.model.config.id2label[label.item()]
            normalized_type = self._normalize_type(label_name)
            
            x1, y1, x2, y2 = box.tolist()
            
            chunks.append({
                'type': normalized_type,
                'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                'confidence': score.item(),
                'color': self.COLOR_MAP.get(normalized_type, '#6b7280'),
                'raw_label': label_name,
                'detection_method': 'foundation_model'
            })
        
        # Sort by reading order (top to bottom, left to right)
        chunks.sort(key=lambda c: (c['bbox'][1], c['bbox'][0]))
        
        return chunks
    
    def detect_tables(self, image: Image.Image) -> List[Dict]:
        """
        Detect only tables in document
        
        Returns:
            List of table regions
        """
        all_chunks = self.analyze_layout(image)
        return [c for c in all_chunks if c['type'] == 'table']
    
    def _normalize_type(self, label: str) -> str:
        """Normalize label names to standard types"""
        label_lower = label.lower()
        
        if 'table' in label_lower:
            return 'table'
        elif 'figure' in label_lower or 'picture' in label_lower or 'image' in label_lower:
            return 'figure'
        elif 'title' in label_lower or 'header' in label_lower:
            return 'title'
        elif 'list' in label_lower:
            return 'list'
        elif 'formula' in label_lower or 'equation' in label_lower:
            return 'formula'
        elif 'caption' in label_lower:
            return 'caption'
        elif 'footer' in label_lower:
            return 'footer'
        else:
            return 'text'
    
    def visualize(self, image: Image.Image, chunks: List[Dict], output_path: str):
        """
        Draw detection boxes on image
        
        Args:
            image: Original image
            chunks: Detection results
            output_path: Path to save visualization
        """
        from PIL import ImageDraw, ImageFont
        
        img = image.copy()
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        for chunk in chunks:
            x, y, w, h = chunk['bbox']
            color = chunk['color']
            label = f"{chunk['type']} ({chunk['confidence']:.0%})"
            
            # Draw box
            draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
            
            # Draw label background
            text_bbox = draw.textbbox((x, y-20), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x, y-20), label, fill='white', font=font)
        
        img.save(output_path)
        print(f"[+] Saved visualization: {output_path}")


# Singleton
_model: Optional[LayoutFoundationModel] = None


def get_layout_foundation_model(model_name: str = "table-transformer") -> LayoutFoundationModel:
    """Get singleton LayoutFoundationModel instance"""
    global _model
    if _model is None or _model.model_name != model_name:
        _model = LayoutFoundationModel(model_name)
    return _model


if __name__ == '__main__':
    print("=" * 50)
    print("Layout Foundation Model Test")
    print("=" * 50)
    
    # Test initialization
    model = LayoutFoundationModel()
    
    if model.is_initialized:
        print("\n[+] Model loaded successfully!")
        print(f"    Device: {model.device}")
        print(f"    Labels: {model.model.config.id2label}")
    else:
        print("\n[!] Model failed to initialize")
        print("    Install: pip install transformers torch")
