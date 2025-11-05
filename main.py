#!/usr/bin/env python3
"""
Syllabus Checker Application - Main Entry Point

A FastAPI-based application for processing question banks.
Usage: python main.py
"""

import os
import sys
import logging
import uvicorn
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_environment():
    current_dir = Path(__file__).parent.absolute()
    sys.path.insert(0, str(current_dir))
    
    data_dir = current_dir / "data_process"
    data_dir.mkdir(exist_ok=True)
    
    if not os.getenv("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY not set")

def check_dependencies():
    try:
        import fastapi
        import pandas
        import sentence_transformers
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

def main():
    print(" Starting Syllabus Checker Application...")
    
    setup_environment()
    
    if not check_dependencies():
        sys.exit(1)
    
    try:
        from main_api import app
        logger.info("FastAPI application loaded")
    except ImportError as e:
        logger.error(f"Failed to import app: {e}")
        sys.exit(1)
    
    print(" Server: http://localhost:8000")
    print(" API Docs: http://localhost:8000/docs")
    print(" Web UI: http://localhost:8000/question_matcher.html")
    print("Press Ctrl+C to stop")
    
    try:
        uvicorn.run(
            "main_api:app",
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n Server stopped")

if __name__ == "__main__":
    main()
