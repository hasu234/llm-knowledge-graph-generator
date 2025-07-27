#!/usr/bin/env python3
"""
Entry point script for the LLM Knowledge Graph Generation API
"""

import uvicorn
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    print("🚀 Starting LLM Knowledge Graph Generation API...")
    print("📊 API Documentation available at: http://localhost:8000/docs")
    print("🔍 Health check available at: http://localhost:8000/health")
    print("=" * 60)
    
    try:
        uvicorn.run(
            "src.app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error starting the server: {e}")
        sys.exit(1)
