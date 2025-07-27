"""
Text preprocessing utilities for handling various encoding issues
"""

import re
import json
import html
from typing import Optional

from ..core.config import settings

def preprocess_news_text(text: str) -> str:
    """
    Preprocess news text to handle JSON encoding issues and special characters
    
    Args:
        text: Input news text to preprocess
        
    Returns:
        Cleaned and normalized text
        
    Raises:
        ValueError: If text becomes too short after preprocessing
    """
    # Handle None or empty input
    if not text or not isinstance(text, str):
        return ""
    
    # HTML decode first to handle entities like &quot;, &amp;, etc.
    text = html.unescape(text)
    
    # Remove or replace problematic control characters (but keep newlines for now)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)
    
    # Handle various unicode characters that might cause issues
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # Smart quotes
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # Smart apostrophes
    text = text.replace('\u2013', '-').replace('\u2014', '-')  # Em/en dashes
    text = text.replace('\u2026', '...')  # Ellipsis
    text = text.replace('\u00a0', ' ')  # Non-breaking space
    text = text.replace('\u00ad', '')  # Soft hyphen
    
    # Handle percentage symbols and other special characters that might cause issues
    text = text.replace('%', ' percent ')
    text = text.replace('&', ' and ')
    text = text.replace('#', ' number ')
    text = text.replace('@', ' at ')
    
    # Normalize quotes
    text = re.sub(r'["""]', '"', text)  # Normalize all quote types to standard quotes
    text = re.sub(r"[''']", "'", text)  # Normalize all apostrophe types
    
    # Handle escape sequences and backslashes
    text = text.replace('\\n', '\n')  # Convert literal \n to actual newline
    text = text.replace('\\r', '\r')  # Convert literal \r to actual carriage return
    text = text.replace('\\t', '\t')  # Convert literal \t to actual tab
    text = text.replace('\\\\', '\\')  # Convert double backslash to single
    
    # Now normalize all whitespace including newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove any remaining problematic characters
    text = re.sub(r'[^\x20-\x7E\x80-\xFF]', ' ', text)
    
    # Length check
    if len(text) > settings.MAX_TEXT_LENGTH:
        text = text[:settings.MAX_TEXT_LENGTH] + "..."
    
    # Final validation - ensure it's valid JSON-serializable
    try:
        json.dumps(text)
        cleaned_text = text
    except (UnicodeDecodeError, TypeError, UnicodeEncodeError):
        # If still problematic, encode and decode to ensure clean UTF-8
        try:
            cleaned_text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
        except:
            # Last resort: return a safe message
            raise ValueError("Error processing text input")
    
    if not cleaned_text or len(cleaned_text.strip()) < settings.MIN_TEXT_LENGTH:
        raise ValueError("Text became too short after preprocessing")
    
    return cleaned_text

def validate_text_input(text: str) -> str:
    """
    Validate and preprocess text input
    
    Args:
        text: Input text to validate
        
    Returns:
        Validated and preprocessed text
        
    Raises:
        ValueError: If text is invalid or too short
    """
    if not text or len(text.strip()) < settings.MIN_TEXT_LENGTH:
        raise ValueError(f"News text must be at least {settings.MIN_TEXT_LENGTH} characters long")
    
    try:
        cleaned_text = preprocess_news_text(text)
    except Exception as e:
        raise ValueError(f"Error preprocessing text: {str(e)}")
    
    if not cleaned_text or len(cleaned_text.strip()) < settings.MIN_TEXT_LENGTH:
        raise ValueError("Text became too short after preprocessing. Please provide longer, more meaningful content.")
    
    return cleaned_text
