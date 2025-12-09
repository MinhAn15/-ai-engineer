"""
Detect hierarchical relationships between layout elements
Based on LandingAI's approach for understanding document structure.

Relationships detected:
- Caption → Figure (text below/above image)
- Header → Table (title above table)  
- Row → Table (text inside table bbox)
- Label → Form field
"""
from typing import List, Dict, Tuple, Optional


class RelationshipDetector:
    """
    Detect relationships: caption→figure, row→table, etc.
    LandingAI understands "this caption belongs to that image".
    """
    
    def __init__(self, 
                 caption_distance: int = 30,
                 header_distance: int = 50,
                 overlap_threshold: float = 0.3):
        """
        Args:
            caption_distance: Max pixels between figure and caption
            header_distance: Max pixels between table and header
            overlap_threshold: Min horizontal overlap ratio
        """
        self.caption_distance = caption_distance
        self.header_distance = header_distance
        self.overlap_threshold = overlap_threshold
    
    def detect_relationships(self, chunks: List[Dict]) -> List[Dict]:
        """
        Add 'parent', 'children', 'relationship' fields to chunks.
        
        Returns:
            Chunks with relationship metadata
        """
        # Initialize relationship fields
        for chunk in chunks:
            if 'parent' not in chunk:
                chunk['parent'] = None
            if 'children' not in chunk:
                chunk['children'] = []
            if 'relationship' not in chunk:
                chunk['relationship'] = None
        
        # Detect each relationship type
        self._detect_caption_figure(chunks)
        self._detect_header_table(chunks)
        self._detect_table_contents(chunks)
        self._detect_form_labels(chunks)
        
        return chunks
    
    def _detect_caption_figure(self, chunks: List[Dict]):
        """
        Detect captions for figures.
        Caption is typically text immediately below/above a figure.
        """
        figures = [c for c in chunks if c.get('type') == 'figure']
        texts = [c for c in chunks if c.get('type') == 'text' and not c.get('parent')]
        
        for fig in figures:
            fig_bbox = fig.get('bbox', (0, 0, 0, 0))
            fig_bottom = fig_bbox[1] + fig_bbox[3]
            fig_top = fig_bbox[1]
            
            best_caption = None
            best_distance = float('inf')
            
            for text in texts:
                if text.get('parent'):  # Already assigned
                    continue
                    
                text_bbox = text.get('bbox', (0, 0, 0, 0))
                text_top = text_bbox[1]
                text_bottom = text_bbox[1] + text_bbox[3]
                
                # Check below figure
                dist_below = abs(text_top - fig_bottom)
                # Check above figure  
                dist_above = abs(fig_top - text_bottom)
                
                distance = min(dist_below, dist_above)
                
                if (distance < self.caption_distance and
                    distance < best_distance and
                    self._horizontal_overlap(fig_bbox, text_bbox) > self.overlap_threshold):
                    
                    best_caption = text
                    best_distance = distance
            
            if best_caption:
                best_caption['parent'] = fig.get('reading_order', id(fig))
                best_caption['relationship'] = 'caption'
                best_caption['type'] = 'caption'  # Reclassify
                fig['children'].append(best_caption.get('reading_order', id(best_caption)))
    
    def _detect_header_table(self, chunks: List[Dict]):
        """
        Detect headers/titles for tables.
        Header is typically title immediately above table.
        """
        tables = [c for c in chunks if c.get('type') == 'table']
        titles = [c for c in chunks if c.get('type') in ('title', 'text') and not c.get('parent')]
        
        for table in tables:
            table_bbox = table.get('bbox', (0, 0, 0, 0))
            table_top = table_bbox[1]
            
            best_header = None
            best_distance = float('inf')
            
            for title in titles:
                if title.get('parent'):
                    continue
                    
                title_bbox = title.get('bbox', (0, 0, 0, 0))
                title_bottom = title_bbox[1] + title_bbox[3]
                
                distance = abs(table_top - title_bottom)
                
                if (distance < self.header_distance and
                    distance < best_distance and
                    title_bottom < table_top and  # Must be above
                    self._horizontal_overlap(table_bbox, title_bbox) > self.overlap_threshold):
                    
                    best_header = title
                    best_distance = distance
            
            if best_header:
                best_header['parent'] = table.get('reading_order', id(table))
                best_header['relationship'] = 'table_header'
                table['children'].append(best_header.get('reading_order', id(best_header)))
                table['has_header'] = True
    
    def _detect_table_contents(self, chunks: List[Dict]):
        """
        Detect text blocks inside tables (table cells).
        """
        tables = [c for c in chunks if c.get('type') == 'table']
        texts = [c for c in chunks if c.get('type') == 'text' and not c.get('parent')]
        
        for table in tables:
            table_bbox = table.get('bbox', (0, 0, 0, 0))
            
            for text in texts:
                if text.get('parent'):
                    continue
                
                text_bbox = text.get('bbox', (0, 0, 0, 0))
                
                # Check if text is inside or mostly inside table
                if self._is_inside(text_bbox, table_bbox) or \
                   self._overlap_ratio(text_bbox, table_bbox) > 0.7:
                    
                    text['parent'] = table.get('reading_order', id(table))
                    text['relationship'] = 'table_cell'
                    table['children'].append(text.get('reading_order', id(text)))
    
    def _detect_form_labels(self, chunks: List[Dict]):
        """
        Detect form field labels (label: value pattern).
        """
        texts = [c for c in chunks if c.get('type') == 'text' and not c.get('parent')]
        
        # Sort by reading order
        texts.sort(key=lambda x: x.get('reading_order', 0))
        
        for i, text in enumerate(texts):
            if text.get('parent'):
                continue
            
            content = text.get('text', '')
            
            # Check if this looks like a label (ends with : or is short)
            if content.endswith(':') or (len(content.split()) <= 3 and i + 1 < len(texts)):
                # Check if next text is on same line (potential value)
                if i + 1 < len(texts):
                    next_text = texts[i + 1]
                    
                    if self._are_on_same_line(text.get('bbox'), next_text.get('bbox')):
                        text['relationship'] = 'form_label'
                        next_text['relationship'] = 'form_value'
                        next_text['parent'] = text.get('reading_order', id(text))
                        text['children'].append(next_text.get('reading_order', id(next_text)))
    
    def _horizontal_overlap(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate horizontal overlap ratio."""
        x1_start, _, w1, _ = bbox1
        x2_start, _, w2, _ = bbox2
        
        x1_end = x1_start + w1
        x2_end = x2_start + w2
        
        overlap_start = max(x1_start, x2_start)
        overlap_end = min(x1_end, x2_end)
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap = overlap_end - overlap_start
        min_width = min(w1, w2)
        
        return overlap / min_width if min_width > 0 else 0.0
    
    def _is_inside(self, small_bbox: Tuple, large_bbox: Tuple) -> bool:
        """Check if small bbox is inside large bbox."""
        sx, sy, sw, sh = small_bbox
        lx, ly, lw, lh = large_bbox
        
        return (sx >= lx and sy >= ly and
                sx + sw <= lx + lw and sy + sh <= ly + lh)
    
    def _overlap_ratio(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate area overlap ratio relative to bbox1."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Intersection
        ix = max(x1, x2)
        iy = max(y1, y2)
        ix2 = min(x1 + w1, x2 + w2)
        iy2 = min(y1 + h1, y2 + h2)
        
        if ix2 <= ix or iy2 <= iy:
            return 0.0
        
        intersection = (ix2 - ix) * (iy2 - iy)
        area1 = w1 * h1
        
        return intersection / area1 if area1 > 0 else 0.0
    
    def _are_on_same_line(self, bbox1: Tuple, bbox2: Tuple, tolerance: int = 15) -> bool:
        """Check if two bboxes are on the same line."""
        y1 = bbox1[1]
        y2 = bbox2[1]
        return abs(y1 - y2) <= tolerance


# Singleton
_detector: Optional[RelationshipDetector] = None


def get_relationship_detector() -> RelationshipDetector:
    """Get singleton RelationshipDetector instance."""
    global _detector
    if _detector is None:
        _detector = RelationshipDetector()
    return _detector


if __name__ == '__main__':
    # Test with sample chunks
    chunks = [
        {'type': 'figure', 'bbox': (100, 100, 200, 150), 'reading_order': 1},
        {'type': 'text', 'bbox': (100, 260, 200, 20), 'text': 'Figure 1: Sample image', 'reading_order': 2},
        {'type': 'title', 'bbox': (100, 300, 200, 30), 'text': 'Sales Data', 'reading_order': 3},
        {'type': 'table', 'bbox': (100, 340, 300, 200), 'reading_order': 4},
        {'type': 'text', 'bbox': (110, 350, 80, 20), 'text': 'Product', 'reading_order': 5},
        {'type': 'text', 'bbox': (200, 350, 80, 20), 'text': 'Amount', 'reading_order': 6},
    ]
    
    detector = RelationshipDetector()
    result = detector.detect_relationships(chunks)
    
    print("Relationship Detection Test")
    print("=" * 50)
    for chunk in result:
        rel = chunk.get('relationship', '-')
        parent = chunk.get('parent', '-')
        children = chunk.get('children', [])
        print(f"  {chunk['reading_order']}. {chunk['type']} -> rel={rel}, parent={parent}, children={children}")
