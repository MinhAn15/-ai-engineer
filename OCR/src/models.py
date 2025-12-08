"""
Data models for Document Intelligence system.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TextBlock:
    '''Single text block extracted from document.'''
    text: str
    confidence: float  # 0.0 to 1.0
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    source_model: str  # "tesseract", "huggingface", "hybrid"
    page_number: int


@dataclass
class DocumentExtractionResult:
    '''Result of document extraction processing.'''
    full_text: str
    text_blocks: List[TextBlock] = field(default_factory=list)
    ocr_score: float = 0.0
    avg_confidence: float = 0.0
    latency_ms: float = 0.0
    models_used: List[str] = field(default_factory=list)
    num_pages: int = 0
    
    def summary(self) -> str:
        '''Return summary string.'''
        return (
            f"📄 Document Processing Summary\n"
            f"{'='*50}\n"
            f"Pages: {self.num_pages}\n"
            f"Text blocks: {len(self.text_blocks)}\n"
            f"Avg confidence: {self.avg_confidence:.2%}\n"
            f"Processing time: {self.latency_ms:.2f}ms\n"
            f"Models used: {', '.join(self.models_used)}\n"
            f"{'='*50}"
        )
