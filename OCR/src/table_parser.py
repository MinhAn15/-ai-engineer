"""
Table structure recognition using Table Transformer (Microsoft)
Parses table structure (rows, columns, cells) from detected table regions.
"""
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
import os

# Try to import Table Transformer
TATR_AVAILABLE = False
try:
    from transformers import AutoModelForObjectDetection, DetrImageProcessor
    import torch
    TATR_AVAILABLE = True
except ImportError:
    print("⚠ Table Transformer not available (transformers not installed)")


class TableParser:
    """
    Parse table structure (rows, columns, cells) using Microsoft Table Transformer
    """
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        
        if TATR_AVAILABLE:
            self._init_model()
    
    def _init_model(self):
        """Initialize Table Transformer model."""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Use Table Transformer for structure recognition
            model_name = "microsoft/table-transformer-structure-recognition"
            
            self.processor = DetrImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForObjectDetection.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ Table Transformer initialized on {self.device}")
        except Exception as e:
            print(f"⚠ Failed to init Table Transformer: {e}")
            self.model = None
    
    def parse_table(self, image: Image.Image, table_bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Parse table structure from image region.
        
        Args:
            image: PIL Image
            table_bbox: (x, y, w, h) of table region
        
        Returns:
            Table structure with cells, rows, columns
        """
        if not self.model:
            return self._fallback_parse(image, table_bbox)
        
        try:
            # Crop table region
            x, y, w, h = table_bbox
            table_img = image.crop((x, y, x + w, y + h))
            
            # Prepare inputs
            inputs = self.processor(images=table_img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process predictions
            target_sizes = torch.tensor([table_img.size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes,
                threshold=0.5
            )[0]
            
            # Extract cells
            cells = []
            rows_detected = []
            cols_detected = []
            
            for score, label, box in zip(
                results["scores"], 
                results["labels"], 
                results["boxes"]
            ):
                label_name = self.model.config.id2label[label.item()]
                cell_x1, cell_y1, cell_x2, cell_y2 = box.tolist()
                
                element = {
                    'bbox': [
                        int(cell_x1) + x,  # Absolute coordinates
                        int(cell_y1) + y,
                        int(cell_x2 - cell_x1),
                        int(cell_y2 - cell_y1)
                    ],
                    'type': label_name,
                    'confidence': score.item()
                }
                
                if label_name == "table row":
                    rows_detected.append(element)
                elif label_name == "table column":
                    cols_detected.append(element)
                elif label_name in ["table cell", "table spanning cell"]:
                    cells.append(element)
            
            # Organize cells into structure
            return {
                'bbox': list(table_bbox),
                'cells': cells,
                'rows': sorted(rows_detected, key=lambda r: r['bbox'][1]),
                'columns': sorted(cols_detected, key=lambda c: c['bbox'][0]),
                'num_rows': len(rows_detected) or self._estimate_rows(cells),
                'num_columns': len(cols_detected) or self._estimate_cols(cells),
                'method': 'table_transformer'
            }
            
        except Exception as e:
            print(f"Table parsing error: {e}")
            return self._fallback_parse(image, table_bbox)
    
    def _fallback_parse(self, image: Image.Image, table_bbox: Tuple[int, int, int, int]) -> Dict:
        """Fallback parsing without model."""
        return {
            'bbox': list(table_bbox),
            'cells': [],
            'rows': [],
            'columns': [],
            'num_rows': 0,
            'num_columns': 0,
            'method': 'fallback'
        }
    
    def _estimate_rows(self, cells: List[Dict]) -> int:
        """Estimate number of rows from cells."""
        if not cells:
            return 0
        y_coords = sorted(set(c['bbox'][1] for c in cells))
        return len(y_coords)
    
    def _estimate_cols(self, cells: List[Dict]) -> int:
        """Estimate number of columns from cells."""
        if not cells:
            return 0
        x_coords = sorted(set(c['bbox'][0] for c in cells))
        return len(x_coords)


def extract_table_content(
    table_structure: Dict, 
    ocr_blocks: List[Dict]
) -> List[List[str]]:
    """
    Extract text content for each cell using OCR blocks.
    
    Args:
        table_structure: Output from TableParser
        ocr_blocks: OCR blocks with 'bbox' and 'text'
    
    Returns:
        2D array of cell contents (rows x cols)
    """
    cells = table_structure.get('cells', [])
    
    if not cells:
        # No cells detected - try to infer from OCR blocks within table
        table_bbox = table_structure.get('bbox', [0, 0, 0, 0])
        blocks_in_table = [
            b for b in ocr_blocks 
            if _is_inside_bbox(b['bbox'], table_bbox)
        ]
        
        if not blocks_in_table:
            return []
        
        # Group into rows
        return _group_blocks_into_rows(blocks_in_table)
    
    # Sort cells into grid
    cells_with_text = []
    for cell in cells:
        cell_text = _extract_text_in_bbox(ocr_blocks, tuple(cell['bbox']))
        cells_with_text.append({
            **cell,
            'text': cell_text
        })
    
    # Organize into 2D array
    return _organize_cells_into_grid(cells_with_text)


def _is_inside_bbox(small_bbox: List[int], large_bbox: List[int]) -> bool:
    """Check if small bbox is inside large bbox."""
    sx, sy, sw, sh = small_bbox
    lx, ly, lw, lh = large_bbox
    
    # Center of small bbox should be inside large bbox
    cx = sx + sw / 2
    cy = sy + sh / 2
    
    return (lx <= cx <= lx + lw and ly <= cy <= ly + lh)


def _extract_text_in_bbox(ocr_blocks: List[Dict], bbox: Tuple[int, int, int, int]) -> str:
    """Extract all text from OCR blocks within bbox."""
    x, y, w, h = bbox
    texts = []
    
    for block in ocr_blocks:
        bx, by, bw, bh = block['bbox']
        
        # Check if block center is inside cell
        bcx = bx + bw / 2
        bcy = by + bh / 2
        
        if x <= bcx <= x + w and y <= bcy <= y + h:
            texts.append(block.get('text', ''))
    
    return ' '.join(texts).strip()


def _group_blocks_into_rows(blocks: List[Dict], y_tolerance: int = 10) -> List[List[str]]:
    """Group OCR blocks into rows by Y coordinate."""
    if not blocks:
        return []
    
    sorted_blocks = sorted(blocks, key=lambda b: b['bbox'][1])
    
    rows = []
    current_row = [sorted_blocks[0]]
    current_y = sorted_blocks[0]['bbox'][1]
    
    for block in sorted_blocks[1:]:
        if abs(block['bbox'][1] - current_y) <= y_tolerance:
            current_row.append(block)
        else:
            # Sort row by X and extract text
            row_sorted = sorted(current_row, key=lambda b: b['bbox'][0])
            rows.append([b.get('text', '') for b in row_sorted])
            current_row = [block]
            current_y = block['bbox'][1]
    
    if current_row:
        row_sorted = sorted(current_row, key=lambda b: b['bbox'][0])
        rows.append([b.get('text', '') for b in row_sorted])
    
    return rows


def _organize_cells_into_grid(cells: List[Dict]) -> List[List[str]]:
    """Organize cells into 2D grid."""
    if not cells:
        return []
    
    # Sort by Y then X
    cells_sorted = sorted(cells, key=lambda c: (c['bbox'][1], c['bbox'][0]))
    
    # Group into rows
    rows = []
    current_row = [cells_sorted[0]]
    current_y = cells_sorted[0]['bbox'][1]
    
    for cell in cells_sorted[1:]:
        if abs(cell['bbox'][1] - current_y) < 20:
            current_row.append(cell)
        else:
            row_sorted = sorted(current_row, key=lambda c: c['bbox'][0])
            rows.append([c.get('text', '') for c in row_sorted])
            current_row = [cell]
            current_y = cell['bbox'][1]
    
    if current_row:
        row_sorted = sorted(current_row, key=lambda c: c['bbox'][0])
        rows.append([c.get('text', '') for c in row_sorted])
    
    return rows


def format_table_as_markdown(table_data: List[List[str]]) -> str:
    """
    Convert 2D array to markdown table.
    
    Args:
        table_data: 2D array of cell contents
    
    Returns:
        Markdown formatted table string
    """
    if not table_data:
        return ""
    
    lines = []
    
    # Header row
    header = table_data[0]
    lines.append('| ' + ' | '.join(str(c) for c in header) + ' |')
    lines.append('|' + '---|' * len(header))
    
    # Data rows
    for row in table_data[1:]:
        # Pad row to match header length
        padded_row = row + [''] * (len(header) - len(row))
        lines.append('| ' + ' | '.join(str(c) for c in padded_row) + ' |')
    
    return '\n'.join(lines)


# Singleton
_parser: Optional[TableParser] = None


def get_table_parser() -> TableParser:
    """Get singleton TableParser instance."""
    global _parser
    if _parser is None:
        _parser = TableParser()
    return _parser


if __name__ == '__main__':
    print("Testing TableParser...")
    
    parser = TableParser()
    
    # Create test image
    test_img = Image.new('RGB', (400, 200), 'white')
    
    # Test parse
    result = parser.parse_table(test_img, (50, 50, 300, 100))
    print(f"✓ Parsed table: {result['num_rows']} rows x {result['num_columns']} cols")
    print(f"  Method: {result['method']}")
