"""
Post-processing to clean OCR output
"""
import re


def clean_ocr_text(text: str) -> str:
    '''
    Clean common OCR errors
    '''
    # Remove weird characters
    text = re.sub(r'[#~@\$%\^&\*]', '', text)
    
    # Fix multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # Fix weird quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Remove isolated special chars at line start
    text = re.sub(r'^[^\w\s]+', '', text, flags=re.MULTILINE)
    
    # Fix common OCR substitutions
    substitutions = {
        '0': 'O',  # Zero to O (context-dependent)
        'l': 'I',  # lowercase L to I
        '|': 'I',
        '1': 'l',  # Depends on context
    }
    
    # Fix multiple newlines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Strip whitespace
    text = text.strip()
    
    return text


def format_structured_text(text: str) -> str:
    '''
    Format text with better structure
    '''
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Capitalize sentences
        if line and not line[0].isupper():
            line = line.capitalize()
        
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)


if __name__ == '__main__':
    # Test
    sample = '''#HHH PROCESS.
 "Lesson': 
 'Data Analysis'

includes 'mean', 'median'). Example: If the document states'''
    
    cleaned = clean_ocr_text(sample)
    print('Cleaned:')
    print(cleaned)
