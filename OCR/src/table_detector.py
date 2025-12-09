"""
Visual table detection using OpenCV
Detects tables by grid lines and text alignment patterns.
"""
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Union
import os


class TableDetector:
    """
    Detect tables using visual features (grid lines, alignment)
    """
    
    def __init__(self):
        self.min_table_area = 10000  # Minimum pixels for table
        self.line_threshold = 50      # Minimum line length
    
    def detect_tables(self, image: Union[str, Image.Image, np.ndarray]) -> List[Dict]:
        """
        Detect tables using grid line detection
        
        Args:
            image: Path to image, PIL Image, or numpy array
        
        Returns:
            List of table regions with bbox (x, y, w, h)
        """
        # Load image
        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image
        
        if img is None:
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        # Detect lines
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines
        table_mask = cv2.add(horizontal_lines, vertical_lines)
        
        # Dilate to connect nearby lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        table_mask = cv2.dilate(table_mask, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(
            table_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and sort contours
        tables = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter by size
            if area < self.min_table_area:
                continue
            
            # Filter by aspect ratio (tables usually have reasonable proportions)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.3 or aspect_ratio > 10:  # Too extreme
                continue
            
            tables.append({
                'bbox': [x, y, w, h],
                'type': 'table',
                'confidence': 0.9,
                'detection_method': 'grid_lines',
                'color': '#f97316'  # Orange
            })
        
        # Sort by area (largest first)
        tables.sort(key=lambda t: t['bbox'][2] * t['bbox'][3], reverse=True)
        
        return tables
    
    def detect_tables_by_alignment(self, ocr_blocks: List[Dict]) -> List[Dict]:
        """
        Detect tables by text alignment patterns (no grid lines needed)
        Works for invoice-style label:value tables (2 columns)
        
        Args:
            ocr_blocks: List of OCR blocks with bbox [x, y, w, h]
        
        Returns:
            List of table regions with row/column info
        """
        if not ocr_blocks or len(ocr_blocks) < 4:
            return []
        
        # Group blocks by vertical alignment (same Y coordinate)
        rows = self._group_into_rows(ocr_blocks)
        
        if len(rows) < 2:
            return []
        
        # Find rows with multiple aligned columns (minimum 2 for invoice-style)
        table_rows = []
        for row in rows:
            if len(row) >= 2:  # At least 2 columns (label:value pattern)
                # Check if evenly spaced or has consistent alignment
                if len(row) >= 3:
                    x_coords = [b['bbox'][0] for b in row]
                    if self._is_evenly_spaced(x_coords):
                        table_rows.append(row)
                elif len(row) == 2:
                    # 2-column table (invoice style) - check left/right pattern
                    if self._is_label_value_pattern(row):
                        table_rows.append(row)
        
        # Merge consecutive table rows into table regions
        tables = self._merge_table_rows(table_rows, min_rows=2)
        
        # Add header detection
        for table in tables:
            table['has_header'] = self._detect_header_row(table, rows)
        
        return tables
    
    def _is_label_value_pattern(self, row: List[Dict]) -> bool:
        """
        Check if a 2-element row looks like label:value pattern
        (left side shorter text, right side longer or equal)
        """
        if len(row) != 2:
            return False
        
        left = row[0]
        right = row[1]
        
        # Check if left side is shorter (label) and right side is value
        left_width = left['bbox'][2]
        right_width = right['bbox'][2]
        
        # Left should be reasonably shorter or similar
        # Also check if there's a clear gap between them
        left_x_end = left['bbox'][0] + left['bbox'][2]
        right_x_start = right['bbox'][0]
        gap = right_x_start - left_x_end
        
        # Valid if: clear gap exists AND not overlapping
        return gap > 10 and left_width > 20 and right_width > 20
    
    def _detect_header_row(self, table: Dict, all_rows: List[List[Dict]]) -> bool:
        """
        Detect if first row of table is a header
        Heuristics: 
        - All text is shorter/capitalized
        - Different font weight (via text patterns)
        """
        # Find first row in table bbox
        table_y = table['bbox'][1]
        table_h = table['bbox'][3]
        
        for row in all_rows:
            if not row:
                continue
            row_y = row[0]['bbox'][1]
            
            # Check if this row is near top of table
            if abs(row_y - table_y) < 30:
                # Simple heuristic: check if text looks like header
                # (often shorter words, possibly uppercase)
                texts = [b.get('text', '') for b in row]
                avg_len = sum(len(t) for t in texts) / len(texts) if texts else 0
                
                # Headers typically have shorter text
                return avg_len < 20
        
        return False
    
    def _group_into_rows(self, blocks: List[Dict], y_tolerance: int = 15) -> List[List[Dict]]:
        """
        Group blocks into rows by Y coordinate
        """
        if not blocks:
            return []
        
        # Sort by Y coordinate
        sorted_blocks = sorted(blocks, key=lambda b: b['bbox'][1])
        
        rows = []
        current_row = [sorted_blocks[0]]
        current_y = sorted_blocks[0]['bbox'][1]
        
        for block in sorted_blocks[1:]:
            block_y = block['bbox'][1]
            
            if abs(block_y - current_y) <= y_tolerance:
                # Same row
                current_row.append(block)
            else:
                # New row
                rows.append(sorted(current_row, key=lambda b: b['bbox'][0]))  # Sort by X
                current_row = [block]
                current_y = block_y
        
        # Add last row
        if current_row:
            rows.append(sorted(current_row, key=lambda b: b['bbox'][0]))
        
        return rows
    
    def _is_evenly_spaced(self, x_coords: List[int], tolerance: float = 0.4) -> bool:
        """
        Check if X coordinates are evenly spaced (indicating columns)
        """
        if len(x_coords) < 3:
            return False
        
        x_sorted = sorted(x_coords)
        
        # Calculate gaps
        gaps = [x_sorted[i+1] - x_sorted[i] for i in range(len(x_sorted)-1)]
        
        if not gaps:
            return False
        
        # Check variance
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        
        if mean_gap == 0:
            return False
        
        return (std_gap / mean_gap) < tolerance
    
    def _merge_table_rows(self, table_rows: List[List[Dict]], min_rows: int = 2) -> List[Dict]:
        """
        Merge consecutive table rows into table regions
        
        Args:
            table_rows: List of rows (each row is list of blocks)
            min_rows: Minimum rows required to form a table
        """
        if not table_rows:
            return []
        
        tables = []
        current_table_rows = [table_rows[0]]
        
        for i in range(1, len(table_rows)):
            prev_row = table_rows[i-1]
            curr_row = table_rows[i]
            
            # Check if rows are consecutive (similar Y distance)
            prev_y = max(b['bbox'][1] + b['bbox'][3] for b in prev_row)
            curr_y = min(b['bbox'][1] for b in curr_row)
            
            if curr_y - prev_y < 50:  # Within 50 pixels
                current_table_rows.append(curr_row)
            else:
                # Save current table, start new one
                if len(current_table_rows) >= 2:  # At least 2 rows
                    tables.append(self._create_table_region(current_table_rows))
                current_table_rows = [curr_row]
        
        # Add last table
        if len(current_table_rows) >= 2:
            tables.append(self._create_table_region(current_table_rows))
        
        return tables
    
    def _create_table_region(self, rows: List[List[Dict]]) -> Dict:
        """
        Create table region from multiple rows
        """
        all_blocks = [block for row in rows for block in row]
        
        # Calculate bounding box
        x_min = min(b['bbox'][0] for b in all_blocks)
        y_min = min(b['bbox'][1] for b in all_blocks)
        x_max = max(b['bbox'][0] + b['bbox'][2] for b in all_blocks)
        y_max = max(b['bbox'][1] + b['bbox'][3] for b in all_blocks)
        
        return {
            'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
            'type': 'table',
            'num_rows': len(rows),
            'num_cols': max(len(row) for row in rows),
            'confidence': 0.85,
            'detection_method': 'alignment',
            'color': '#f97316'
        }
    
    def detect_tables_by_data_pattern(self, ocr_blocks: List[Dict]) -> List[Dict]:
        """
        Detect tables by analyzing data patterns (LandingAI style).
        Works for tables WITHOUT gridlines.
        
        Patterns:
        - Multiple rows with same column structure
        - Numeric alignment (amounts, quantities)
        - Repeated keywords (headers)
        
        Args:
            ocr_blocks: List of OCR blocks with bbox and text
        
        Returns:
            List of detected table regions
        """
        if not ocr_blocks or len(ocr_blocks) < 4:
            return []
        
        rows = self._group_into_rows(ocr_blocks)
        
        if len(rows) < 2:
            return []
        
        # Analyze row patterns to find table candidates
        table_candidates = []
        i = 0
        
        while i < len(rows) - 1:
            # Try to find table starting at row i
            table_rows = [rows[i]]
            
            for j in range(i + 1, len(rows)):
                # Check if this row has consistent structure with previous
                if self._has_consistent_structure([table_rows[-1], rows[j]]):
                    table_rows.append(rows[j])
                else:
                    break
            
            # Need at least 2 rows with consistent structure
            if len(table_rows) >= 2:
                # Bonus confidence if has numeric columns
                has_numeric = self._has_numeric_columns(table_rows)
                
                table_candidates.append({
                    'start_idx': i,
                    'end_idx': i + len(table_rows) - 1,
                    'rows': table_rows,
                    'has_numeric': has_numeric,
                    'confidence': 0.9 if has_numeric else 0.75
                })
                i += len(table_rows)
            else:
                i += 1
        
        # Convert candidates to table regions
        tables = []
        for candidate in table_candidates:
            table = self._create_table_region(candidate['rows'])
            table['confidence'] = candidate['confidence']
            table['detection_method'] = 'data_pattern'
            table['has_numeric_columns'] = candidate['has_numeric']
            tables.append(table)
        
        return tables
    
    def _has_consistent_structure(self, rows: List[List[Dict]]) -> bool:
        """
        Check if rows have consistent column structure.
        Used for data pattern table detection.
        """
        if len(rows) < 2:
            return True
        
        # Count columns in each row
        col_counts = [len(row) for row in rows]
        
        # Must have similar column count (within 1)
        if max(col_counts) - min(col_counts) > 1:
            return False
        
        min_cols = min(col_counts)
        if min_cols < 2:  # Need at least 2 columns
            return False
        
        # Check X alignment of columns
        for col_idx in range(min_cols):
            x_coords = []
            for row in rows:
                if len(row) > col_idx:
                    x_coords.append(row[col_idx]['bbox'][0])
            
            if x_coords:
                # X coordinates should be similar (within 30px)
                if max(x_coords) - min(x_coords) > 30:
                    return False
        
        return True
    
    def _has_numeric_columns(self, rows: List[List[Dict]]) -> bool:
        """
        Check if any columns contain primarily numeric data.
        Tables often have numeric columns (prices, quantities, etc.).
        """
        if not rows:
            return False
        
        num_cols = max(len(row) for row in rows)
        
        for col_idx in range(num_cols):
            numeric_count = 0
            total_count = 0
            
            for row in rows:
                if len(row) > col_idx:
                    text = row[col_idx].get('text', '')
                    total_count += 1
                    
                    if self._is_numeric(text):
                        numeric_count += 1
            
            # If >40% of column is numeric → likely table column
            if total_count > 0 and numeric_count / total_count > 0.4:
                return True
        
        return False
    
    def _is_numeric(self, text: str) -> bool:
        """Check if text is numeric (flexible for currency, percentages, etc.)."""
        if not text:
            return False
        
        # Remove common numeric formatting
        clean = text.strip()
        clean = clean.replace(',', '').replace('.', '')
        clean = clean.replace('$', '').replace('€', '').replace('£', '')
        clean = clean.replace('%', '').replace('VND', '').replace('đ', '')
        clean = clean.replace(' ', '').replace('-', '')
        
        # Check if mostly digits
        if not clean:
            return False
        
        digit_count = sum(1 for c in clean if c.isdigit())
        return digit_count / len(clean) > 0.5
    
    def detect_all_tables(self, image, ocr_blocks: List[Dict]) -> List[Dict]:
        """
        Combined table detection using all methods.
        
        1. Grid line detection (visual)
        2. Alignment-based detection
        3. Data pattern detection
        
        Returns deduplicated table regions.
        """
        tables = []
        
        # Method 1: Grid lines
        grid_tables = self.detect_tables(image)
        for t in grid_tables:
            t['detection_method'] = 'grid_lines'
        tables.extend(grid_tables)
        
        # Method 2: Alignment-based
        alignment_tables = self.detect_tables_by_alignment(ocr_blocks)
        tables.extend(alignment_tables)
        
        # Method 3: Data pattern
        pattern_tables = self.detect_tables_by_data_pattern(ocr_blocks)
        tables.extend(pattern_tables)
        
        # Deduplicate overlapping tables (keep highest confidence)
        tables = self._deduplicate_tables(tables)
        
        return tables
    
    def _deduplicate_tables(self, tables: List[Dict]) -> List[Dict]:
        """Remove overlapping table detections, keep highest confidence."""
        if len(tables) <= 1:
            return tables
        
        # Sort by confidence (descending)
        tables.sort(key=lambda t: t.get('confidence', 0), reverse=True)
        
        result = []
        for table in tables:
            # Check if overlaps with any existing table
            overlaps = False
            for existing in result:
                if self._tables_overlap(table['bbox'], existing['bbox']):
                    overlaps = True
                    break
            
            if not overlaps:
                result.append(table)
        
        return result
    
    def _tables_overlap(self, bbox1: List, bbox2: List, threshold: float = 0.5) -> bool:
        """Check if two table bboxes overlap significantly."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        ix = max(x1, x2)
        iy = max(y1, y2)
        ix2 = min(x1 + w1, x2 + w2)
        iy2 = min(y1 + h1, y2 + h2)
        
        if ix2 <= ix or iy2 <= iy:
            return False
        
        intersection = (ix2 - ix) * (iy2 - iy)
        area1 = w1 * h1
        area2 = w2 * h2
        smaller_area = min(area1, area2)
        
        return intersection / smaller_area > threshold if smaller_area > 0 else False
    
    def detect_checkboxes(self, image: Union[str, Image.Image, np.ndarray]) -> List[Dict]:
        """
        Detect checkboxes/form elements using contour detection
        
        Returns:
            List of checkbox regions
        """
        # Load image
        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image
        
        if img is None:
            return []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        checkboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Checkbox characteristics: small, roughly square
            if 10 < w < 50 and 10 < h < 50:
                aspect_ratio = w / h
                if 0.7 < aspect_ratio < 1.3:  # Square-ish
                    checkboxes.append({
                        'bbox': [x, y, w, h],
                        'type': 'checkbox',
                        'confidence': 0.7,
                        'color': '#84cc16'  # Lime
                    })
        
        return checkboxes


# Singleton
_detector = None


def get_table_detector() -> TableDetector:
    """Get singleton TableDetector instance."""
    global _detector
    if _detector is None:
        _detector = TableDetector()
    return _detector


def detect_tables_in_image(image: Union[str, Image.Image]) -> List[Dict]:
    """
    Convenience function to detect tables in an image.
    
    Args:
        image: Path or PIL Image
    
    Returns:
        List of table regions
    """
    detector = get_table_detector()
    return detector.detect_tables(image)


if __name__ == '__main__':
    # Test
    print("Testing TableDetector...")
    
    detector = TableDetector()
    
    # Create test image with grid lines
    test_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Draw table grid
    cv2.rectangle(test_img, (50, 50), (550, 350), (0, 0, 0), 2)
    cv2.line(test_img, (50, 150), (550, 150), (0, 0, 0), 1)
    cv2.line(test_img, (50, 250), (550, 250), (0, 0, 0), 1)
    cv2.line(test_img, (200, 50), (200, 350), (0, 0, 0), 1)
    cv2.line(test_img, (400, 50), (400, 350), (0, 0, 0), 1)
    
    tables = detector.detect_tables(test_img)
    print(f"✓ Detected {len(tables)} tables in test image")
    
    for t in tables:
        print(f"  - {t['detection_method']}: bbox={t['bbox']}, conf={t['confidence']:.2f}")
