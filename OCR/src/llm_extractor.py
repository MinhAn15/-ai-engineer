"""
LLM-based schema extraction from OCR text.
Uses Google Gemini API to extract structured data based on natural language schema prompts.
Output in TOON (Token-Optimized Object Notation) format for graph database compatibility.
"""

import os
import re
import json
from typing import Optional, Dict, List, Any

# Try to import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠ google-generativeai not installed. Run: pip install google-generativeai")

# Try to import usage tracker
try:
    from src.llm_usage_tracker import UsageTracker
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False


# TOON Format Specification
TOON_SPEC = """
TOON (Token-Optimized Object Notation) Format:

Entities:
@entity:<EntityType>
<property>:<value>
<property>:<value>

Relations:
@rel:<RelationType>
from:<source_id>
to:<target_id>
<property>:<value>

Rules:
- One property per line
- Use : as key-value separator
- No quotes needed for values
- Use | for multi-value properties
- Empty lines separate blocks
"""


def get_extraction_prompt(ocr_text: str, schema_prompt: str) -> str:
    """
    Generate LLM prompt for schema extraction.
    
    Args:
        ocr_text: Raw OCR text
        schema_prompt: User's natural language schema description
    
    Returns:
        Complete prompt for LLM
    """
    return f"""You are a data extraction expert. Extract structured information from the OCR text below.

USER SCHEMA REQUEST:
{schema_prompt}

OCR TEXT:
---
{ocr_text}
---

INSTRUCTIONS:
1. Analyze the user's schema request to understand what entities and properties to extract
2. Extract relevant information from the OCR text
3. Output in TOON format (Token-Optimized Object Notation)

TOON FORMAT RULES:
- @entity:<Type> starts an entity block
- @rel:<Type> starts a relationship block
- property:value on each line
- Use | to separate multiple values
- Empty line between blocks
- Use _id property for unique identifiers
- Keep property names short but clear

EXAMPLE OUTPUT:
@entity:Company
_id:company_1
name:Acme Corp
address:123 Main St

@entity:Invoice
_id:inv_001
number:INV-2024-001
date:2024-01-15
total:1500.00
currency:USD

@rel:ISSUED_BY
from:inv_001
to:company_1

OUTPUT (TOON format only, no explanations):"""


def extract_with_llm(
    ocr_text: str,
    schema_prompt: str,
    api_key: Optional[str] = None,
    model: str = "gemini-2.0-flash-exp",
    min_confidence: float = 0.75,
    ocr_blocks: Optional[List[Dict]] = None
) -> str:
    """
    Extract structured data from OCR text using Gemini LLM.
    
    Args:
        ocr_text: Raw OCR text from document
        schema_prompt: Natural language description of what to extract
        api_key: Gemini API key (uses env var if not provided)
        model: Gemini model to use
        min_confidence: Minimum confidence threshold for OCR blocks (0.0-1.0)
        ocr_blocks: Optional list of OCR text blocks with confidence scores
    
    Returns:
        TOON formatted string with extracted data
    """
    if not GEMINI_AVAILABLE:
        return "# Error: google-generativeai not installed"
    
    # Filter low-confidence blocks if provided
    if ocr_blocks:
        high_conf_blocks = [
            b for b in ocr_blocks 
            if b.get('confidence', 1.0) >= min_confidence
        ]
        
        if high_conf_blocks:
            # Rebuild text from high-confidence blocks only
            filtered_text = ' '.join([b['text'] for b in high_conf_blocks])
            
            filtered_pct = len(high_conf_blocks) / len(ocr_blocks) * 100
            print(f"   🔍 Confidence filter: {len(high_conf_blocks)}/{len(ocr_blocks)} blocks ({filtered_pct:.1f}%) above {min_confidence:.0%}")
            
            # Use filtered text for LLM
            ocr_text = filtered_text
        else:
            print(f"   ⚠ Warning: No blocks above {min_confidence:.0%} confidence threshold")
    
    # Get API key
    api_key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return "# Error: GEMINI_API_KEY not set. Set environment variable or pass api_key parameter."
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Create model
        llm = genai.GenerativeModel(model)
        
        # Generate prompt
        prompt = get_extraction_prompt(ocr_text, schema_prompt)
        
        # Call LLM
        response = llm.generate_content(prompt)
        
        # Track usage
        if TRACKER_AVAILABLE:
            try:
                tracker = UsageTracker()
                # Approximate token count (more accurate than word count)
                input_tokens = len(prompt.split()) * 1.3  # ~1.3 tokens per word
                output_tokens = len(response.text.split()) * 1.3
                tracker.log_request(model, input_tokens, output_tokens, schema_prompt)
            except Exception as e:
                print(f"   Warning: Usage tracking failed: {e}")
        
        # Extract and clean response
        toon_output = response.text.strip()
        
        # Remove markdown code blocks if present
        toon_output = re.sub(r'^```\w*\n?', '', toon_output)
        toon_output = re.sub(r'\n?```$', '', toon_output)
        
        return toon_output.strip()
        
    except Exception as e:
        return f"# Error: LLM extraction failed - {str(e)}"


def parse_toon(toon_text: str) -> Dict[str, Any]:
    """
    Parse TOON format into structured dict with validation.
    Useful for verification and conversion to other formats.
    
    Args:
        toon_text: TOON formatted string
    
    Returns:
        Dict with 'entities', 'relations', and optional 'error' key
    """
    result = {
        "entities": [],
        "relations": []
    }
    
    try:
        current_block = None
        current_type = None
        current_data = {}
        
        for line in toon_text.split('\n'):
            line = line.strip()
            
            if not line or line.startswith('#'):
                # Empty line or comment - save current block if exists
                if current_block and current_data:
                    if current_block == 'entity':
                        result['entities'].append({
                            'type': current_type,
                            'properties': current_data.copy()
                        })
                    elif current_block == 'rel':
                        result['relations'].append({
                            'type': current_type,
                            'properties': current_data.copy()
                        })
                    current_data = {}
                continue
            
            if line.startswith('@entity:'):
                # Save previous block
                if current_block and current_data:
                    if current_block == 'entity':
                        result['entities'].append({
                            'type': current_type,
                            'properties': current_data.copy()
                        })
                    elif current_block == 'rel':
                        result['relations'].append({
                            'type': current_type,
                            'properties': current_data.copy()
                        })
                
                current_block = 'entity'
                current_type = line.split(':', 1)[1]
                current_data = {}
                
            elif line.startswith('@rel:'):
                # Save previous block
                if current_block and current_data:
                    if current_block == 'entity':
                        result['entities'].append({
                            'type': current_type,
                            'properties': current_data.copy()
                        })
                    elif current_block == 'rel':
                        result['relations'].append({
                            'type': current_type,
                            'properties': current_data.copy()
                        })
                
                current_block = 'rel'
                current_type = line.split(':', 1)[1]
                current_data = {}
                
            elif ':' in line:
                # Property line
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Handle multi-value
                if '|' in value:
                    value = [v.strip() for v in value.split('|')]
                
                current_data[key] = value
        
        # Save last block
        if current_block and current_data:
            if current_block == 'entity':
                result['entities'].append({
                    'type': current_type,
                    'properties': current_data.copy()
                })
            elif current_block == 'rel':
                result['relations'].append({
                    'type': current_type,
                    'properties': current_data.copy()
                })
        
        # Validation checks
        if not result['entities']:
            print("   ⚠ Warning: No entities extracted from TOON")
        
        if len(result['entities']) > 20:
            print(f"   ⚠ Warning: Too many entities ({len(result['entities'])}) - possible hallucination")
        
        # Validate relationships have valid from/to references
        if result['relations']:
            entity_ids = set()
            for entity in result['entities']:
                if '_id' in entity['properties']:
                    entity_ids.add(entity['properties']['_id'])
            
            for rel in result['relations']:
                from_id = rel['properties'].get('from')
                to_id = rel['properties'].get('to')
                
                if from_id and from_id not in entity_ids:
                    print(f"   ⚠ Warning: Relation references unknown entity: {from_id}")
                if to_id and to_id not in entity_ids:
                    print(f"   ⚠ Warning: Relation references unknown entity: {to_id}")
        
        return result
        
    except Exception as e:
        print(f"   ✗ TOON parsing error: {e}")
        return {
            'entities': [],
            'relations': [],
            'error': str(e)
        }


def toon_to_json(toon_text: str) -> str:
    """Convert TOON to JSON format."""
    parsed = parse_toon(toon_text)
    return json.dumps(parsed, indent=2, ensure_ascii=False)


def toon_to_cypher(toon_text: str) -> str:
    """
    Convert TOON to Cypher query format for Neo4j.
    
    Args:
        toon_text: TOON formatted string
    
    Returns:
        Cypher CREATE statements
    """
    parsed = parse_toon(toon_text)
    cypher_lines = []
    
    # Create nodes
    for entity in parsed['entities']:
        entity_type = entity['type']
        props = entity['properties']
        
        # Build properties string
        prop_parts = []
        for k, v in props.items():
            if isinstance(v, list):
                v = v[0]  # Take first value for Cypher
            if isinstance(v, str):
                prop_parts.append(f'{k}: "{v}"')
            else:
                prop_parts.append(f'{k}: {v}')
        
        props_str = ', '.join(prop_parts)
        node_id = props.get('_id', entity_type.lower())
        
        cypher_lines.append(f"CREATE ({node_id}:{entity_type} {{{props_str}}})")
    
    # Create relationships
    for rel in parsed['relations']:
        rel_type = rel['type']
        props = rel['properties']
        from_id = props.get('from', '')
        to_id = props.get('to', '')
        
        cypher_lines.append(f"CREATE ({from_id})-[:{rel_type}]->({to_id})")
    
    return '\n'.join(cypher_lines)


if __name__ == '__main__':
    # Test with sample text
    sample_ocr = """
    INSURANCE COMPANY
    Home Office: Boston
    
    PHILIP MORRIS INCORPORATED Account #00 40 66
    100 PARK AVENUE
    NEW YORK, NY 10017
    
    Invoice #0125
    Date: 01-01-77
    Total: $3,612,852
    """
    
    sample_schema = "Extract: company name, address, invoice number, date, total amount"
    
    print("Testing LLM Extraction...")
    print(f"API Key set: {bool(os.getenv('GEMINI_API_KEY'))}")
    
    if os.getenv('GEMINI_API_KEY'):
        result = extract_with_llm(sample_ocr, sample_schema)
        print("\nTOON Output:")
        print(result)
        
        print("\nParsed:")
        parsed = parse_toon(result)
        print(json.dumps(parsed, indent=2))
    else:
        print("\n⚠ Set GEMINI_API_KEY to test LLM extraction")
        print("Example: $env:GEMINI_API_KEY='your-api-key'")
