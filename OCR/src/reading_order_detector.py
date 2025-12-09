"""
Reading order detection - LandingAI style
Based on: https://github.com/DS4SD/docling

Detects reading order of document segments following human reading patterns.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image


class ReadingOrderDetector:
    """
    Detect reading order of document segments
    Following human reading patterns (top→bottom, left→right in columns)
    """
    
    def __init__(self, column_threshold: float = 0.3, line_threshold: int = 15):
        """
        Args:
            column_threshold: Overlap threshold for same column (0.0-1.0)
            line_threshold: Max vertical gap (px) to consider same line
        """
        self.column_threshold = column_threshold
        self.line_threshold = line_threshold
    
    def detect_reading_order(self, segments: List[Dict]) -> List[Dict]:
        """
        Assign reading order numbers to segments
        
        Args:
            segments: List of {text, bbox, ...}
        
        Returns:
            Segments with added 'reading_order' and 'column' fields
        """
        if not segments:
            return []
        
        # Step 1: Detect columns
        columns = self._detect_columns(segments)
        
        # Step 2: Sort segments within each column (top to bottom)
        ordered_segments = []
        order_num = 1
        
        # Sort columns left to right
        columns.sort(key=lambda col: min(s['bbox'][0] for s in col) if col else 0)
        
        for col_idx, column in enumerate(columns):
            # Sort by Y coordinate (top to bottom)
            column_sorted = sorted(column, key=lambda s: s['bbox'][1])
            
            for segment in column_sorted:
                segment['reading_order'] = order_num
                segment['column'] = col_idx + 1
                ordered_segments.append(segment)
                order_num += 1
        
        return ordered_segments
    
    def _detect_columns(self, segments: List[Dict]) -> List[List[Dict]]:
        """
        Detect column structure in document using spatial clustering
        
        Returns:
            List of columns, each column is a list of segments
        """
        if not segments:
            return []
        
        # Sort segments by X coordinate (left edge)
        sorted_segs = sorted(segments, key=lambda s: s['bbox'][0])
        
        # Group into columns based on X overlap
        columns = []
        current_column = [sorted_segs[0]]
        current_x_range = self._get_x_range(sorted_segs[0]['bbox'])
        
        for segment in sorted_segs[1:]:
            seg_x_range = self._get_x_range(segment['bbox'])
            
            # Check overlap with current column
            overlap = self._calculate_overlap(current_x_range, seg_x_range)
            
            if overlap > self.column_threshold:
                # Same column
                current_column.append(segment)
                # Update column X range
                current_x_range = self._merge_ranges(current_x_range, seg_x_range)
            else:
                # New column
                columns.append(current_column)
                current_column = [segment]
                current_x_range = seg_x_range
        
        # Add last column
        if current_column:
            columns.append(current_column)
        
        return columns
    
    def _get_x_range(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get X range (left, right) from bbox"""
        x, y, w, h = bbox
        return (x, x + w)
    
    def _calculate_overlap(self, range1: Tuple[int, int], range2: Tuple[int, int]) -> float:
        """Calculate overlap ratio between two X ranges"""
        x1_start, x1_end = range1
        x2_start, x2_end = range2
        
        # Calculate intersection
        intersect_start = max(x1_start, x2_start)
        intersect_end = min(x1_end, x2_end)
        
        if intersect_end <= intersect_start:
            return 0.0
        
        intersection = intersect_end - intersect_start
        
        # Calculate union
        union = (x1_end - x1_start) + (x2_end - x2_start) - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_ranges(self, range1: Tuple[int, int], range2: Tuple[int, int]) -> Tuple[int, int]:
        """Merge two X ranges"""
        return (min(range1[0], range2[0]), max(range1[1], range2[1]))
    
    def detect_lines(self, segments: List[Dict]) -> List[List[Dict]]:
        """
        Group segments into lines (for tables/aligned text)
        
        Returns:
            List of lines, each line is a list of segments
        """
        if not segments:
            return []
        
        # Sort by Y coordinate
        sorted_segs = sorted(segments, key=lambda s: s['bbox'][1])
        
        lines = []
        current_line = [sorted_segs[0]]
        current_y = sorted_segs[0]['bbox'][1]
        
        for segment in sorted_segs[1:]:
            seg_y = segment['bbox'][1]
            
            if abs(seg_y - current_y) <= self.line_threshold:
                # Same line
                current_line.append(segment)
            else:
                # New line
                lines.append(sorted(current_line, key=lambda s: s['bbox'][0]))
                current_line = [segment]
                current_y = seg_y
        
        # Add last line
        if current_line:
            lines.append(sorted(current_line, key=lambda s: s['bbox'][0]))
        
        return lines
    
    def get_context_for_segment(self, segment: Dict, all_segments: List[Dict]) -> Dict:
        """
        Get context information for a segment (previous/next segments)
        
        Returns:
            Dict with 'prev_segment', 'next_segment', 'above', 'below'
        """
        order = segment.get('reading_order', 0)
        
        context = {
            'prev_segment': None,
            'next_segment': None,
            'above': [],
            'below': [],
            'same_line': []
        }
        
        seg_y = segment['bbox'][1]
        seg_h = segment['bbox'][3]
        
        for s in all_segments:
            s_order = s.get('reading_order', 0)
            s_y = s['bbox'][1]
            
            if s_order == order - 1:
                context['prev_segment'] = s
            elif s_order == order + 1:
                context['next_segment'] = s
            
            # Check vertical relationship
            if s is not segment:
                if s_y < seg_y - self.line_threshold:
                    context['above'].append(s)
                elif s_y > seg_y + seg_h + self.line_threshold:
                    context['below'].append(s)
                elif abs(s_y - seg_y) <= self.line_threshold:
                    context['same_line'].append(s)
        
        return context


# Singleton instance
_detector = None

def get_reading_order_detector() -> ReadingOrderDetector:
    """Get singleton reading order detector"""
    global _detector
    if _detector is None:
        _detector = ReadingOrderDetector()
    return _detector


if __name__ == '__main__':
    # Test with sample segments
    segments = [
        {'text': 'Title', 'bbox': (50, 50, 200, 30)},
        {'text': 'Column 1 para 1', 'bbox': (50, 100, 150, 50)},
        {'text': 'Column 2 para 1', 'bbox': (220, 100, 150, 50)},
        {'text': 'Column 1 para 2', 'bbox': (50, 170, 150, 50)},
        {'text': 'Column 2 para 2', 'bbox': (220, 170, 150, 50)},
    ]
    
    detector = ReadingOrderDetector()
    ordered = detector.detect_reading_order(segments)
    
    print("Reading Order Detection Test")
    print("=" * 40)
    for seg in ordered:
        print(f"  {seg['reading_order']}. [Col {seg['column']}] {seg['text']}")
