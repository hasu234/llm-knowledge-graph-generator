"""
Analysis API endpoints
"""

import time
from fastapi import HTTPException, status, Form

from ...api.models.response import AnalysisResult, DetailedAnalysisResult
from ...services.llm_clients import llm_clients
from ...services.text_processing import validate_text_input
from ...services.triple_parser import parse_triples_from_text, format_triples_for_display
from ...services.metrics import calculate_icr_metric, calculate_ipr_metric
from ...core.config import settings

async def analyze_news_endpoint(news_text: str = Form(..., description="The news text to analyze")) -> AnalysisResult:
    """
    Analyze news text and return ICR and IPR scores from all three LLM providers.
    
    This endpoint accepts form data, making it easier to send
    text with special characters, quotes, newlines, etc.
    
    Args:
        news_text: The news text to analyze
        
    Returns:
        AnalysisResult with ICR/IPR scores and latencies for all models
        
    Raises:
        HTTPException: For various validation and processing errors
    """
    try:
        # Step 0: Validate and preprocess the text
        cleaned_text = validate_text_input(news_text)
        
        # Process with OpenAI
        try:
            openai_start_time = time.time()
            generated_triples_text_openai = await llm_clients.generate_triples_openai(cleaned_text)
            
            if not generated_triples_text_openai:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Failed to generate triples from the provided news text"
                )
            
            triples_list_openai = parse_triples_from_text(generated_triples_text_openai)
            if not triples_list_openai:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="No valid triples could be parsed from the generated text"
                )
            
            icr_score_openai, _, _ = calculate_icr_metric(triples_list_openai)
            ipr_score_openai, _, _ = calculate_ipr_metric(triples_list_openai)
            latency_openai = time.time() - openai_start_time
        
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing OpenAI response: {str(e)}"
            )

        # Process with Mistral
        try:
            mistral_start_time = time.time()
            generated_triples_text_mistral = await llm_clients.generate_triples_mistral(cleaned_text)
            
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
            latency_mistral = time.time() - mistral_start_time

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing Mistral response: {str(e)}"
            )
        
        # Process with Gemini
        try:
            gemini_start_time = time.time()
            generated_triples_text_gemini = await llm_clients.generate_triples_gemini(cleaned_text)
            
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
            latency_gemini = time.time() - gemini_start_time

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing Gemini response: {str(e)}"
            )

        # Prepare response
        result = AnalysisResult(
            openai_model=settings.OPENAI_MODEL,
            icr_score_openai=icr_score_openai,
            ipr_score_openai=ipr_score_openai,
            latency_openai=latency_openai,
            mistral_model=settings.MISTRAL_MODEL,
            icr_score_mistral=icr_score_mistral,
            ipr_score_mistral=ipr_score_mistral,
            latency_mistral=latency_mistral,
            gemini_model=settings.GEMINI_MODEL,
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

async def analyze_news_debug_endpoint(news_text: str = Form(..., description="The news text to analyze")) -> DetailedAnalysisResult:
    """
    Analyze news text and return detailed ICR and IPR scores with debug information.
    
    This endpoint provides detailed breakdown of the calculation for debugging purposes.
    Accepts form data, making it easier to send text with special characters.
    
    Args:
        news_text: The news text to analyze
        
    Returns:
        DetailedAnalysisResult with comprehensive debug information
        
    Raises:
        HTTPException: For various validation and processing errors
    """
    try:
        # Step 0: Validate and preprocess the text
        cleaned_text = validate_text_input(news_text)
        
        # Step 1: Generate RDF triples from news text (using OpenAI for debug)
        generated_triples_text = await llm_clients.generate_triples_openai(cleaned_text)
        
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
        formatted_triples = format_triples_for_display(triples_list)
        
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
