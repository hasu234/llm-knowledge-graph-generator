"""
FastAPI main application entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .api.endpoints.analysis import analyze_news_endpoint, analyze_news_debug_endpoint
from .api.models.response import AnalysisResult, DetailedAnalysisResult

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register endpoints
app.post("/analyze", response_model=AnalysisResult, tags=["Analysis"])(analyze_news_endpoint)
app.post("/analyze-debug", response_model=DetailedAnalysisResult, tags=["Debug"])(analyze_news_debug_endpoint)

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint for health check"""
    return {
        "message": "News Analysis API is running",
        "version": settings.API_VERSION,
        "status": "healthy"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": settings.API_TITLE,
        "version": settings.API_VERSION
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting News Analysis API...")
    uvicorn.run("src.app.main:app", host="0.0.0.0", port=8000, reload=True)
