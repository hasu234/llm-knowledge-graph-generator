"""
RDF triple parsing utilities
"""

from typing import List, Tuple

def parse_triples_from_text(text: str) -> List[Tuple[str, str, str]]:
    """
    Parse RDF triples from LLM response text
    
    Args:
        text: LLM response text containing RDF triples
        
    Returns:
        List of parsed triples as (subject, predicate, object) tuples
    """
    triples = []
    
    # Split text into lines and process each line
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or not line.startswith('<'):
            continue
            
        try:
            # Remove the outer < and > brackets
            if line.startswith('<') and line.endswith('>'):
                content = line[1:-1]
            else:
                continue
            
            # Use a more sophisticated approach to split the triple
            # Find the commas that separate the three parts
            comma_positions = []
            in_uri = False
            in_quotes = False
            
            for i, char in enumerate(content):
                if char == '"' and not in_uri:
                    in_quotes = not in_quotes
                elif char == '<' and not in_quotes:
                    in_uri = True
                elif char == '>' and not in_quotes:
                    in_uri = False
                elif char == ',' and not in_uri and not in_quotes:
                    comma_positions.append(i)
            
            # We need exactly 2 commas to split into 3 parts
            if len(comma_positions) >= 2:
                # Use the first two commas to split into subject, predicate, object
                first_comma = comma_positions[0]
                second_comma = comma_positions[1]
                
                subject = content[:first_comma].strip()
                predicate = content[first_comma + 1:second_comma].strip()
                obj = content[second_comma + 1:].strip()
                
                # Clean object - remove quotes if it's a literal
                if obj.startswith('"') and (obj.endswith('"') or '@' in obj):
                    # Handle cases like "Thousands"@en
                    if '@' in obj:
                        obj = obj.split('@')[0][1:-1]  # Remove quotes and language tag
                    else:
                        obj = obj[1:-1]  # Just remove quotes
                
                triples.append((subject, predicate, obj))
            
        except Exception as e:
            print(f"Error parsing line: {line} - {e}")
            continue
    
    return triples

def format_triples_for_display(triples_list: List[Tuple[str, str, str]]) -> List[str]:
    """
    Format triples for display in API responses
    
    Args:
        triples_list: List of (subject, predicate, object) tuples
        
    Returns:
        List of formatted triple strings
    """
    return [f"<{s}, {p}, {o}>" for s, p, o in triples_list]
