"""
Direct server runner for the question matching API
"""

import uvicorn
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the app
from main import app

if __name__ == "__main__":
    print("🚀 Starting Question Matcher API Server...")
    print("📍 Server will be available at: http://127.0.0.1:8000")
    print("📋 API Documentation at: http://127.0.0.1:8000/docs")
    print("🎯 Question Matcher UI at: http://127.0.0.1:8000/question_matcher.html")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        access_log=True
    )