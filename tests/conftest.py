"""
Test configuration and fixtures
"""

import pytest
import asyncio
from typing import Generator

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def sample_news_text():
    """Sample news text for testing"""
    return "Elon Musk announced that Tesla will build a new factory in Berlin, Germany."

@pytest.fixture
def sample_triples():
    """Sample RDF triples for testing metrics"""
    return [
        ("http://dbpedia.org/resource/Elon_Musk", "rdf:type", "http://dbpedia.org/ontology/Person"),
        ("http://dbpedia.org/resource/Tesla_Inc", "rdf:type", "http://dbpedia.org/ontology/Company"),
        ("http://dbpedia.org/resource/Elon_Musk", "dbo:founder", "http://dbpedia.org/resource/Tesla_Inc"),
        ("http://dbpedia.org/resource/Berlin", "rdf:type", "http://dbpedia.org/ontology/Place"),
    ]
