# LLM Knowledge Graph Generation

A comprehensive toolkit for analyzing news text and generating knowledge graphs using Large Language Models (OpenAI GPT, Mistral, and Google Gemini).

## Features

- **Multi-LLM Support**: Integrates with OpenAI GPT, Mistral AI, and Google Gemini
- **Knowledge Graph Metrics**: Calculates ICR (Instantiated Class Ratio) and IPR (Instantiated Property Ratio)
- **RDF Triple Generation**: Converts news text into structured DBpedia-style RDF triples
- **RESTful API**: FastAPI-based web service with automatic documentation
- **Modular Architecture**: Well-organized codebase with proper separation of concerns
- **Comprehensive Testing**: Unit tests and integration tests included

## Project Structure

```
llm-knowledge-graph-generation/
├── src/
│   ├── app/
│   │   ├── main.py              # FastAPI application entry point
│   │   ├── api/
│   │   │   ├── endpoints/       # API endpoint definitions
│   │   │   └── models/          # Pydantic models for requests/responses
│   │   ├── core/
│   │   │   ├── config.py        # Configuration and settings
│   │   │   └── prompts.py       # LLM system prompts
│   │   └── services/
│   │       ├── llm_clients.py   # LLM client integrations
│   │       ├── text_processing.py # Text preprocessing utilities
│   │       ├── triple_parser.py  # RDF triple parsing
│   │       └── metrics.py       # ICR/IPR calculation
├── tests/                       # Test suite
├── scripts/                     # Utility scripts
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
└── pyproject.toml             # Project configuration
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hasmatali/llm-knowledge-graph-generation
   cd llm-knowledge-graph-generation
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys
   ```

   Add your API keys to the `.env` file:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   MISTRAL_API_KEY=your-mistral-api-key-here
   GEMINI_API_KEY=your-gemini-api-key-here
   ```

5. **Run the application:**
   ```bash
   python app.py
   ```

The API will be available at: `http://localhost:8000`

### Interactive Documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### POST `/analyze`
Analyze news text and return ICR and IPR scores from all three LLM providers.

**Request:** Form data
```bash
curl -X 'POST' \
  'http://localhost:8000/analyze' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'news_text=Elon Musk announced that Tesla will build a new factory in Berlin.'
```

**Response:**
```json
{
  "openai_model": "gpt-3.5-turbo",
  "icr_score_openai": 0.75,
  "ipr_score_openai": 0.33,
  "latency_openai": 2.5,
  "mistral_model": "mistral-medium-latest",
  "icr_score_mistral": 0.70,
  "ipr_score_mistral": 0.30,
  "latency_mistral": 1.8,
  "gemini_model": "gemini-2.5-flash",
  "icr_score_gemini": 0.72,
  "ipr_score_gemini": 0.35,
  "latency_gemini": 1.2
}
```

### POST `/analyze-debug`
Get detailed analysis results with debug information (uses OpenAI only).

**Response includes:**
- Generated RDF triples
- Detailed class and property breakdowns
- ICR/IPR calculation details

## Development

### Install Development Dependencies
```bash
pip install -r requirements-dev.txt
```

### Run Tests
```bash
pytest
```

### Code Formatting
```bash
black src/ tests/
isort src/ tests/
```

### Type Checking
```bash
mypy src/
```

## Metrics Explanation

### ICR (Instantiated Class Ratio)
Measures the ratio of classes that have instances to the total number of unique classes in the knowledge graph.

**Formula:** `ICR = n(IC) / n(C)`
- `n(C)`: Total number of unique classes
- `n(IC)`: Number of classes that have instances

### IPR (Instantiated Property Ratio)
Measures the ratio of standard semantic web properties to all properties used in the knowledge graph.

**Formula:** `IPR = n(IP) / n(P)`
- `n(P)`: Total number of unique properties
- `n(IP)`: Number of properties from standard namespaces (RDF, RDFS, OWL, FOAF, etc.)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Author

**Hasmot Ali** - [GitHub](https://github.com/hasmatali)
