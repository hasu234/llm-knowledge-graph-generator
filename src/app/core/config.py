"""
Core configuration and settings module
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings and configuration"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Model configurations
    OPENAI_MODEL = "gpt-3.5-turbo"
    MISTRAL_MODEL = "mistral-medium-latest"
    GEMINI_MODEL = "gemini-2.5-flash"
    
    # API settings
    API_TITLE = "News Analysis API"
    API_DESCRIPTION = "API for calculating ICR and IPR scores from news text"
    API_VERSION = "1.0.0"
    
    # LLM settings
    TEMPERATURE = 0.3
    MAX_TOKENS = 1000
    
    # Text processing settings
    MAX_TEXT_LENGTH = 10000
    MIN_TEXT_LENGTH = 10
    
    def __init__(self, validate_keys: bool = True):
        """Initialize settings and optionally validate required API keys"""
        if validate_keys:
            self._validate_api_keys()
    
    def _validate_api_keys(self):
        """Validate that all required API keys are set"""
        if not self.OPENAI_API_KEY:
            raise ValueError("Please set your OPENAI_API_KEY environment variable")
        
        if not self.MISTRAL_API_KEY:
            raise ValueError("Please set your MISTRAL_API_KEY environment variable")
        
        if not self.GEMINI_API_KEY:
            raise ValueError("Please set your GEMINI_API_KEY environment variable")

# Global settings instance - validate keys by default, but allow override via environment
_SKIP_VALIDATION = os.getenv("SKIP_API_KEY_VALIDATION", "false").lower() == "true"
settings = Settings(validate_keys=not _SKIP_VALIDATION)
