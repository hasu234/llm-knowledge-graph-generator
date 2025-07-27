"""
Test script for Google Gemini API
"""

import os
import asyncio
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

async def test_gemini_api():
    """Test Google Gemini API connection and basic functionality"""
    try:
        # Initialize client
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Test basic generation
        print("Testing Gemini API...")
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents="Explain how AI works in a few words"
        )
        
        print(f"Response: {response.text}")
        print("✅ Gemini API test successful!")
        
    except Exception as e:
        print(f"❌ Gemini API test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_gemini_api())
