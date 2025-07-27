#!/usr/bin/env python3
"""
FastAPI application for calculating ICR (Instantiated Class Ratio) from news text.
This API takes news text as input and returns ICR metrics based on generated RDF triples.
"""

import os
import re
import time
import asyncio
from typing import List, Tuple, Set
from mistralai import Mistral
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, status, Form
from fastapi.middleware.cors import CORSMiddleware
import openai
from google import genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="News Analysis API",
    description="API for calculating ICR and IPR scores from news text",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY environment variable")

# Check if MISTRAL_API_KEY is set
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("Please set your MISTRAL_API_KEY environment variable")

# Check if GEMINI_API_KEY is set
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set your GEMINI_API_KEY environment variable")

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize Mistral client
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# Initialize Google GenAI client
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# System prompt for RDF triple generation
SYSTEM_PROMPT = """
You are a Knowledge Graph Generation Agent trained to extract and convert natural language news articles into structured DBpedia-style RDF triplets. 
Your task is to deeply understand the context of the news, map entities and relations to the DBpedia ontology, and generate 
well-formed RDF triples in the format: <subject, predicate, object>. 

Your Objectives: 
1. Contextual Understanding: Read and understand the news article, including implicit and explicit relations. 
2. Entity Linking: Map all identified entities to their DBpedia resources, classes, or subclasses. Use proper URI structure as 
per DBpedia: Classes: http://dbpedia.org/ontology/Person, http://dbpedia.org/ontology/Company, etc. 
Resources: http://dbpedia.org/resource/Elon_Musk, http://dbpedia.org/resource/Tesla_Inc
3. Relation Mapping: Use predicates from DBpedia ontology (e.g., dbo:founder, dbo:location, dbo:date, dbo:spouse, etc.). 
4. Triplet Format: Output each triple in this exact format: <http://dbpedia.org/resource/Subject, dbo:property, http://dbpedia.org/resource/Object> or 
for literals: <http://dbpedia.org/resource/Subject, dbo:property, \"Literal_Value\"@en>. 
5. Typing Triplets (Class Assignment): For each main entity, include an RDF 
type triplet: <http://dbpedia.org/resource/Elon_Musk, rdf:type, http://dbpedia.org/ontology/Person>. 

Important Rules: 
- Only use properties and classes from the official DBpedia ontology (http://dbpedia.org/ontology/). 
- Disambiguate ambiguous entities where possible. 
- Prefer dbo: over dbp:, unless a dbo: mapping doesn't exist. 
- Avoid duplicate triplets.
- NEVER use commas in URI resource names. Use underscores instead (e.g., Tesla_Inc not Tesla,_Inc.).
- Each triple must be on its own line. 

Example Output: 
News Input: \"Elon Musk announced that Tesla would build a new factory in Berlin.\" 
Output Triplets:
<http://dbpedia.org/resource/Elon_Musk, rdf:type, http://dbpedia.org/ontology/Person>
<http://dbpedia.org/resource/Elon_Musk, foaf:name, "Elon Musk"@en>
<http://dbpedia.org/resource/Elon_Musk, dbo:birthPlace, http://dbpedia.org/resource/Pretoria>
<http://dbpedia.org/resource/Elon_Musk, dbo:occupation, http://dbpedia.org/resource/Entrepreneur>
<http://dbpedia.org/resource/Tesla_Inc, rdf:type, http://dbpedia.org/ontology/Company>
<http://dbpedia.org/resource/Tesla_Inc, foaf:name, "Tesla, Inc."@en>
<http://dbpedia.org/resource/Tesla_Inc, dbo:industry, http://dbpedia.org/resource/Automotive_industry>
<http://dbpedia.org/resource/Tesla_Inc, dbo:keyPerson, http://dbpedia.org/resource/Elon_Musk>
<http://dbpedia.org/resource/Tesla_Inc, dbo:foundationPlace, http://dbpedia.org/resource/San_Carlos,_California>
<http://dbpedia.org/resource/Tesla_Inc, dbo:foundationYear, "2003"^^xsd:gYear>
<http://dbpedia.org/resource/Berlin, rdf:type, http://dbpedia.org/ontology/Place>
<http://dbpedia.org/resource/Berlin, dbo:country, http://dbpedia.org/resource/Germany>
<http://dbpedia.org/resource/Berlin, foaf:name, "Berlin"@en>
<http://dbpedia.org/resource/Berlin, dbo:populationTotal, "3769000"^^xsd:integer>
<http://dbpedia.org/resource/Elon_Musk, dbo:founder, http://dbpedia.org/resource/Tesla_Inc>
<http://dbpedia.org/resource/Tesla_Inc, dbo:locationCity, http://dbpedia.org/resource/Berlin>
<http://dbpedia.org/resource/Tesla_Inc, dbo:project, http://dbpedia.org/resource/Gigafactory_Berlin>
<http://dbpedia.org/resource/Gigafactory_Berlin, rdf:type, http://dbpedia.org/ontology/Factory>
<http://dbpedia.org/resource/Gigafactory_Berlin, dbo:location, http://dbpedia.org/resource/Berlin>
<http://dbpedia.org/resource/Gigafactory_Berlin, dbo:owner, http://dbpedia.org/resource/Tesla_Inc>
<http://dbpedia.org/resource/Gigafactory_Berlin, dbo:buildingStartDate, "2020-06"^^xsd:gYearMonth>

Later the ICR, IPR, CI, CI 2, SPA, IMI and other metrics will be calculated from the generated RDF triples.
"""

# Pydantic models for response

class AnalysisResult(BaseModel):
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

# Text preprocessing helper
def preprocess_news_text(text: str) -> str:
    """Preprocess news text to handle JSON encoding issues and special characters"""
    import json
    import html
    
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
    if len(text) > 10000:  # Reasonable limit for news articles
        text = text[:10000] + "..."
    
    # Final validation - ensure it's valid JSON-serializable
    try:
        json.dumps(text)
        return text
    except (UnicodeDecodeError, TypeError, UnicodeEncodeError):
        # If still problematic, encode and decode to ensure clean UTF-8
        try:
            return text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
        except:
            # Last resort: return a safe message
            return "Error processing text input"

# Core functions from the notebook
async def generate_triples_from_news_openai(news_text: str) -> str:
    """Generate RDF triples from news text using OpenAI API"""
    try:
        response = await asyncio.create_task(
            asyncio.to_thread(
                openai_client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extract RDF triples from this news text:\n\n{news_text}"}
                ],
                temperature=0.3,
                max_tokens=1000
            )
        )
        
        content = response.choices[0].message.content
        if content is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OpenAI returned empty response"
            )
        triples_text = content.strip()
        return triples_text
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating triples openai: {str(e)}"
        )

async def generate_triples_from_news_mistral(news_text: str) -> str:
    """Generate RDF triples from news text using Mistral API"""
    try:
        response = await asyncio.create_task(
            asyncio.to_thread(
                mistral_client.chat.complete,
                model="mistral-medium-latest",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extract RDF triples from this news text:\n\n{news_text}"}
                ],
                temperature=0.3,
                max_tokens=1000
            )
        )

        content = response.choices[0].message.content
        if content is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Mistral returned empty response"
            )
        triples_text = content.strip()
        return triples_text

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating triples mistral: {str(e)}"
        )
    
async def generate_triples_from_news_gemini(news_text: str) -> str:
    """Generate RDF triples from news text using Google Gemini API"""
    try:
        response = await asyncio.create_task(
            asyncio.to_thread(
                gemini_client.models.generate_content,
                model="gemini-2.5-flash",
                contents=f"{SYSTEM_PROMPT}. Now extract RDF triples from this news text:\n\n{news_text}"
            )
        )

        content = response.text
        if content is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Gemini returned empty response"
            )
        triples_text = content.strip()
        return triples_text

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating triples gemini: {str(e)}"
        )

def parse_triples_from_text(text: str) -> List[Tuple[str, str, str]]:
    """Parse RDF triples from LLM response text"""
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

def extract_unique_classes_from_triples(triples_list: List[Tuple[str, str, str]]) -> Set[str]:
    """Extract unique classes from the generated triples based on subjects and objects"""
    unique_classes = set()
    
    for subject, predicate, obj in triples_list:
        # Add subjects that are DBpedia ontology classes/resources
        if subject.startswith("http://dbpedia.org/"):
            unique_classes.add(subject)
        
        # Add objects that are DBpedia ontology classes/resources (not literals)
        # Exclude literal values (those with quotes, datatype markers, or language tags)
        if (obj.startswith("http://dbpedia.org/") and 
            not obj.startswith('"') and 
            '^^' not in obj and 
            '@' not in obj):
            unique_classes.add(obj)
    
    return unique_classes

def find_instantiated_classes(triples_list: List[Tuple[str, str, str]], unique_classes: Set[str]) -> Set[str]:
    """Find classes that have instances (appear as objects in rdf:type relations)"""
    instantiated_classes = set()
    
    for subject, predicate, obj in triples_list:
        # Look for rdf:type predicates or predicates containing 'type'
        if 'type' in predicate.lower() or 'rdf:type' in predicate:
            # Object is the class being instantiated
            if obj in unique_classes:
                instantiated_classes.add(obj)
        
        # Also look for RDFS predicates that indicate class relationships
        if 'rdfs:' in predicate.lower() or 'rdfs' in predicate.lower():
            if obj in unique_classes:
                instantiated_classes.add(obj)
    
    return instantiated_classes

def calculate_icr_metric(triples_list: List[Tuple[str, str, str]]) -> Tuple[float, Set[str], Set[str]]:
    """
    Calculate ICR (Instantiated Class Ratio) metric
    ICR = n(IC) / n(C)
    where:
    - n(C) = total number of unique classes in the ontology/triples
    - n(IC) = number of classes that have instances
    """
    # Step 1: Extract all unique classes from triples
    unique_classes = extract_unique_classes_from_triples(triples_list)
    
    # Step 2: Find classes that have instances
    instantiated_classes = find_instantiated_classes(triples_list, unique_classes)
    
    # Step 3: Calculate ICR
    n_C = len(unique_classes)  # Total number of classes
    n_IC = len(instantiated_classes)  # Number of instantiated classes
    
    if n_C == 0:
        icr = 0.0
    else:
        icr = n_IC / n_C
    
    return icr, unique_classes, instantiated_classes

def extract_unique_properties_from_triples(triples_list: List[Tuple[str, str, str]]) -> Set[str]:
    """Extract unique properties (predicates) from the generated triples"""
    unique_properties = set()
    
    for subject, predicate, obj in triples_list:
        # Add the predicate to unique properties
        unique_properties.add(predicate)
    
    return unique_properties

def calculate_ipr_metric(triples_list: List[Tuple[str, str, str]]) -> Tuple[float, Set[str], Set[str]]:
    """
    Calculate IPR (Instantiated Property Ratio) metric
    IPR = n(IP) / n(P)
    where:
    - n(P) = total number of unique properties found in the triples
    - n(IP) = number of properties that are from RDF/RDFS/OWL standard namespaces
    
    This measures how many standard semantic web properties are used
    compared to all properties in the knowledge graph.
    """
    # Step 1: Extract all unique properties from triples
    unique_properties = extract_unique_properties_from_triples(triples_list)
    
    # Step 2: Count instantiated properties (properties from standard semantic web namespaces)
    instantiated_properties = set()
    
    # Standard semantic web namespaces
    standard_namespaces = [
        'http://www.w3.org/1999/02/22-rdf-syntax-ns#',  # RDF
        'http://www.w3.org/2000/01/rdf-schema#',        # RDFS
        'http://www.w3.org/2002/07/owl#',               # OWL
        'http://xmlns.com/foaf/0.1/',                   # FOAF
        'http://purl.org/dc/elements/1.1/',             # Dublin Core
        'http://purl.org/dc/terms/',                    # Dublin Core Terms
    ]
    
    # Short form prefixes for standard namespaces
    standard_prefixes = ['rdf:', 'rdfs:', 'owl:', 'foaf:', 'dc:', 'dct:']
    
    for _, predicate, _ in triples_list:
        # Check for full URI form
        for namespace in standard_namespaces:
            if namespace in predicate:
                instantiated_properties.add(predicate)
                break
        else:
            # Check for short form (rdf:type, foaf:name, etc.)
            for prefix in standard_prefixes:
                if predicate.startswith(prefix):
                    instantiated_properties.add(predicate)
                    break
    
    # Step 3: Calculate IPR
    n_P = len(unique_properties)  # Total number of unique properties
    n_IP = len(instantiated_properties)  # Number of standard namespace properties
    
    if n_P == 0:
        ipr = 0.0
    else:
        ipr = n_IP / n_P
    
    return ipr, unique_properties, instantiated_properties

@app.post("/analyze", response_model=AnalysisResult, tags=["Analysis"])
async def analyze_news(news_text: str = Form(..., description="The news text to analyze")):
    """
    Analyze news text and return ICR and IPR scores.
    
    This endpoint accepts form data, making it easier to send
    text with special characters, quotes, newlines, etc.
    """
    try:
        # Validate input length
        if not news_text or len(news_text.strip()) < 10:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="News text must be at least 10 characters long"
            )
        
        # Step 0: Preprocess the text to handle any encoding issues
        try:
            cleaned_text = preprocess_news_text(news_text)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Error preprocessing text: {str(e)}"
            )
        
        if not cleaned_text or len(cleaned_text.strip()) < 10:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Text became too short after preprocessing. Please provide longer, more meaningful content."
            )
        try:
            # Measure start time for latency calculation
            openai_start_time = time.time()
            
            # Step 1: Generate RDF triples from news text
            generated_triples_text_openai = await generate_triples_from_news_openai(cleaned_text)
            if not generated_triples_text_openai:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Failed to generate triples from the provided news text"
                )
            
            # Step 2: Parse the generated triples
            triples_list_openai = parse_triples_from_text(generated_triples_text_openai)
            if not triples_list_openai:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="No valid triples could be parsed from the generated text"
                )
            
            # Step 3: Calculate both ICR and IPR metrics
            icr_score_openai, _, _ = calculate_icr_metric(triples_list_openai)
            ipr_score_openai, _, _ = calculate_ipr_metric(triples_list_openai)

            # Measure latency for OpenAI model
            latency_openai = time.time() - openai_start_time
        
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing OpenAI response: {str(e)}"
            )

        try:
            # Step 2: Measure start time for Mistral model
            mistral_start_time = time.time()

            generated_triples_text_mistral = await generate_triples_from_news_mistral(cleaned_text)
            if not generated_triples_text_mistral:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Failed to generate triples from the provided news text using Mistral"
                )
            
            triples_list_mistral = parse_triples_from_text(generated_triples_text_mistral)
            if not triples_list_mistral:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="No valid triples could be parsed from the generated text using Mistral"
                )
            
            icr_score_mistral, _, _ = calculate_icr_metric(triples_list_mistral)
            ipr_score_mistral, _, _ = calculate_ipr_metric(triples_list_mistral)

            # Measure latency for Mistral model
            latency_mistral = time.time() - mistral_start_time

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing Mistral response: {str(e)}"
            )
        
        try:
            # Step 3: Measure start time for Gemini model
            gemini_start_time = time.time()

            generated_triples_text_gemini = await generate_triples_from_news_gemini(cleaned_text)
            if not generated_triples_text_gemini:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Failed to generate triples from the provided news text using Gemini"
                )
            
            
            triples_list_gemini = parse_triples_from_text(generated_triples_text_gemini)
            if not triples_list_gemini:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="No valid triples could be parsed from the generated text using Gemini"
                )

            icr_score_gemini, _, _ = calculate_icr_metric(triples_list_gemini)
            ipr_score_gemini, _, _ = calculate_ipr_metric(triples_list_gemini)

            # Measure latency for Gemini model
            latency_gemini = time.time() - gemini_start_time

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing Gemini response: {str(e)}"
            )

        # Step 4: Prepare response
        result = AnalysisResult(
            openai_model="gpt-3.5-turbo",
            icr_score_openai=icr_score_openai,
            ipr_score_openai=ipr_score_openai,
            latency_openai=latency_openai,
            mistral_model="mistral-medium-latest",
            icr_score_mistral=icr_score_mistral,
            ipr_score_mistral=ipr_score_mistral,
            latency_mistral=latency_mistral,
            gemini_model="gemini-1.5",
            icr_score_gemini=icr_score_gemini,
            ipr_score_gemini=ipr_score_gemini,
            latency_gemini=latency_gemini
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@app.post("/analyze-debug", response_model=DetailedAnalysisResult, tags=["Debug"])
async def analyze_news_debug(news_text: str = Form(..., description="The news text to analyze")):
    """
    Analyze news text and return detailed ICR and IPR scores with debug information.
    
    This endpoint provides detailed breakdown of the calculation for debugging purposes.
    Accepts form data, making it easier to send text with special characters.
    """
    try:
        # Validate input length
        if not news_text or len(news_text.strip()) < 10:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="News text must be at least 10 characters long"
            )
        
        # Step 0: Preprocess the text to handle any encoding issues
        try:
            cleaned_text = preprocess_news_text(news_text)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Error preprocessing text: {str(e)}"
            )
        
        if not cleaned_text or len(cleaned_text.strip()) < 10:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Text became too short after preprocessing. Please provide longer, more meaningful content."
            )
        
        # Step 1: Generate RDF triples from news text
        generated_triples_text = await generate_triples_from_news_openai(cleaned_text)
        
        if not generated_triples_text:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to generate triples from the provided news text"
            )
        
        # Step 2: Parse the generated triples
        triples_list = parse_triples_from_text(generated_triples_text)
        
        if not triples_list:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No valid triples could be parsed from the generated text"
            )
        
        # Step 3: Calculate both ICR and IPR metrics with detailed info
        icr_score, unique_classes, instantiated_classes = calculate_icr_metric(triples_list)
        ipr_score, unique_properties, rdf_syntax_properties = calculate_ipr_metric(triples_list)
        
        # Step 4: Format triples for display
        formatted_triples = [f"<{s}, {p}, {o}>" for s, p, o in triples_list]
        
        # Step 5: Prepare detailed response
        result = DetailedAnalysisResult(
            icr_score_openai=icr_score,
            ipr_score_openai=ipr_score,
            total_classes=len(unique_classes),
            total_classes_name=unique_classes,
            instantiated_classes=len(instantiated_classes),
            instantiated_classes_name=instantiated_classes,
            total_properties=len(unique_properties),
            rdf_syntax_properties=len(rdf_syntax_properties),
            all_properties=list(unique_properties),
            rdf_properties=list(rdf_syntax_properties),
            generated_triples=formatted_triples
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    print("Starting News Analysis API...")
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)

