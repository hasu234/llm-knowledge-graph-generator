"""
Pydantic models for API responses
"""

from typing import List, Set
from pydantic import BaseModel, Field

class AnalysisResult(BaseModel):
    """Analysis result model for the main analysis endpoint"""
    openai_model: str = Field(default="gpt-3.5-turbo", description="OpenAI model used for analysis")
    icr_score_openai: float = Field(..., description="ICR (Instantiated Class Ratio) score")
    ipr_score_openai: float = Field(..., description="IPR (Instantiated Property Ratio) score")
    latency_openai: float = Field(..., description="Latency for OpenAI model in seconds")
    mistral_model: str = Field(default="mistral-medium-latest", description="Mistral model used for analysis")
    icr_score_mistral: float = Field(..., description="ICR score from Mistral model")
    ipr_score_mistral: float = Field(..., description="IPR score from Mistral model")
    latency_mistral: float = Field(..., description="Latency for Mistral model in seconds")
    gemini_model: str = Field(default="gemini-2.5-flash", description="Google Gemini model used for analysis")
    icr_score_gemini: float = Field(..., description="ICR score from Gemini model")
    ipr_score_gemini: float = Field(..., description="IPR score from Gemini model")
    latency_gemini: float = Field(..., description="Latency for Gemini model in seconds")

class DetailedAnalysisResult(BaseModel):
    """Detailed analysis result model for the debug endpoint"""
    icr_score_openai: float = Field(..., description="ICR (Instantiated Class Ratio) score")
    ipr_score_openai: float = Field(..., description="IPR (Instantiated Property Ratio) score")
    total_classes: int = Field(..., description="Total number of unique classes")
    total_classes_name: Set[str] = Field(..., description="Name of the total classes")
    instantiated_classes: int = Field(..., description="Number of instantiated classes")
    instantiated_classes_name: Set[str] = Field(..., description="Name of the instantiated classes")
    total_properties: int = Field(..., description="Total number of unique properties")
    rdf_syntax_properties: int = Field(..., description="Number of RDF syntax namespace properties")
    all_properties: List[str] = Field(..., description="List of all properties found")
    rdf_properties: List[str] = Field(..., description="List of RDF syntax namespace properties")
    generated_triples: List[str] = Field(..., description="Generated RDF triples")
