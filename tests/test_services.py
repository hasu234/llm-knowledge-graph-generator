"""
Tests for services modules
"""

import pytest
from src.app.services.text_processing import preprocess_news_text, validate_text_input
from src.app.services.triple_parser import parse_triples_from_text, format_triples_for_display
from src.app.services.metrics import calculate_icr_metric, calculate_ipr_metric

class TestTextProcessing:
    """Tests for text processing functions"""
    
    def test_preprocess_news_text_basic(self):
        """Test basic text preprocessing"""
        text = "This is a basic news text."
        result = preprocess_news_text(text)
        assert result == text
    
    def test_preprocess_news_text_special_chars(self):
        """Test preprocessing with special characters"""
        text = "Elon Musk's Tesla & Co. announced 50% growth."
        result = preprocess_news_text(text)
        assert " and " in result
        assert " percent " in result
    
    def test_validate_text_input_valid(self, sample_news_text):
        """Test text validation with valid input"""
        result = validate_text_input(sample_news_text)
        assert len(result) >= 10
    
    def test_validate_text_input_too_short(self):
        """Test text validation with too short input"""
        with pytest.raises(ValueError, match="must be at least"):
            validate_text_input("short")

class TestTripleParser:
    """Tests for triple parsing functions"""
    
    def test_parse_triples_from_text(self):
        """Test parsing triples from text"""
        text = """
        <http://dbpedia.org/resource/Elon_Musk, rdf:type, http://dbpedia.org/ontology/Person>
        <http://dbpedia.org/resource/Tesla_Inc, rdf:type, http://dbpedia.org/ontology/Company>
        """
        result = parse_triples_from_text(text)
        assert len(result) == 2
        assert result[0][0] == "http://dbpedia.org/resource/Elon_Musk"
        assert result[0][1] == "rdf:type"
        assert result[0][2] == "http://dbpedia.org/ontology/Person"
    
    def test_format_triples_for_display(self, sample_triples):
        """Test formatting triples for display"""
        result = format_triples_for_display(sample_triples)
        assert len(result) == len(sample_triples)
        assert all(triple.startswith("<") and triple.endswith(">") for triple in result)

class TestMetrics:
    """Tests for metrics calculation functions"""
    
    def test_calculate_icr_metric(self, sample_triples):
        """Test ICR metric calculation"""
        icr_score, unique_classes, instantiated_classes = calculate_icr_metric(sample_triples)
        assert 0 <= icr_score <= 1
        assert len(unique_classes) > 0
        assert len(instantiated_classes) <= len(unique_classes)
    
    def test_calculate_ipr_metric(self, sample_triples):
        """Test IPR metric calculation"""
        ipr_score, unique_properties, rdf_properties = calculate_ipr_metric(sample_triples)
        assert 0 <= ipr_score <= 1
        assert len(unique_properties) > 0
        assert len(rdf_properties) <= len(unique_properties)
    
    def test_metrics_with_empty_triples(self):
        """Test metrics calculation with empty triples"""
        empty_triples = []
        icr_score, _, _ = calculate_icr_metric(empty_triples)
        ipr_score, _, _ = calculate_ipr_metric(empty_triples)
        assert icr_score == 0.0
        assert ipr_score == 0.0
