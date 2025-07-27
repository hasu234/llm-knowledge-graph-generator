"""
LLM client integrations for OpenAI, Mistral, and Gemini
"""

import asyncio
from typing import Optional

import openai
from mistralai import Mistral
from google import genai

from ..core.config import settings
from ..core.prompts import SYSTEM_PROMPT

class LLMClients:
    """LLM client manager for all supported language models"""
    
    def __init__(self):
        """Initialize all LLM clients"""
        self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.mistral_client = Mistral(api_key=settings.MISTRAL_API_KEY)
        self.gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
    
    async def generate_triples_openai(self, news_text: str) -> str:
        """Generate RDF triples from news text using OpenAI API"""
        try:
            response = await asyncio.create_task(
                asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model=settings.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Extract RDF triples from this news text:\n\n{news_text}"}
                    ],
                    temperature=settings.TEMPERATURE,
                    max_tokens=settings.MAX_TOKENS
                )
            )
            
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("OpenAI returned empty response")
            
            return content.strip()
        
        except Exception as e:
            raise RuntimeError(f"Error generating triples with OpenAI: {str(e)}")
    
    async def generate_triples_mistral(self, news_text: str) -> str:
        """Generate RDF triples from news text using Mistral API"""
        try:
            response = await asyncio.create_task(
                asyncio.to_thread(
                    self.mistral_client.chat.complete,
                    model=settings.MISTRAL_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Extract RDF triples from this news text:\n\n{news_text}"}
                    ],
                    temperature=settings.TEMPERATURE,
                    max_tokens=settings.MAX_TOKENS
                )
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Mistral returned empty response")
            
            return content.strip()

        except Exception as e:
            raise RuntimeError(f"Error generating triples with Mistral: {str(e)}")
    
    async def generate_triples_gemini(self, news_text: str) -> str:
        """Generate RDF triples from news text using Google Gemini API"""
        try:
            response = await asyncio.create_task(
                asyncio.to_thread(
                    self.gemini_client.models.generate_content,
                    model=settings.GEMINI_MODEL,
                    contents=f"{SYSTEM_PROMPT}. Now extract RDF triples from this news text:\n\n{news_text}"
                )
            )

            content = response.text
            if content is None:
                raise ValueError("Gemini returned empty response")
            
            return content.strip()

        except Exception as e:
            raise RuntimeError(f"Error generating triples with Gemini: {str(e)}")

# Global LLM clients instance
llm_clients = LLMClients()
