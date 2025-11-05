"""
Simple Question Matcher API using Open Source Embeddings
Similar to the Streamlit app but as a REST API endpoint
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import EmbeddingGenerator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Simple Question Matcher API",
    description="Match user questions with master questions using open-source embeddings",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global embedding generator
embedding_generator = None

def initialize_embedding_generator():
    """Initialize the embedding generator on startup."""
    global embedding_generator
    try:
        embedding_generator = EmbeddingGenerator(model_key="bge-large-en")
        logger.info("Embedding generator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize embedding generator: {str(e)}")
        raise

# Request/Response models
class QuestionMatchSimpleRequest(BaseModel):
    master_questions: List[str]
    user_questions: List[str]
    similarity_threshold: float = 0.75
    model_key: Optional[str] = "bge-large-en"

class MatchedQuestion(BaseModel):
    user_question: str
    matched_master_question: str
    similarity_score: float

class QuestionMatchSimpleResponse(BaseModel):
    matches: List[MatchedQuestion]
    total_user_questions: int
    total_master_questions: int
    total_matches: int
    match_percentage: float
    similarity_threshold: float

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    initialize_embedding_generator()

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Simple Question Matcher API",
        "version": "1.0.0",
        "description": "Match user questions with master questions using open-source embeddings",
        "endpoints": [
            "/match-questions",
            "/health"
        ],
        "models_available": [
            "bge-large-en (default)",
            "gte-large", 
            "e5-large-v2",
            "bge-m3"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global embedding_generator
    
    try:
        if embedding_generator is None:
            return {"status": "error", "message": "Embedding generator not initialized"}
        
        # Test with a simple embedding
        test_embedding = embedding_generator.generate_embedding("test")
        
        return {
            "status": "healthy",
            "embedding_generator": "ready",
            "model_info": embedding_generator.get_model_info(),
            "test_embedding_shape": test_embedding.shape
        }
    except Exception as e:
        return {"status": "error", "message": f"Health check failed: {str(e)}"}

@app.post("/match-questions", response_model=QuestionMatchSimpleResponse)
async def match_questions(request: QuestionMatchSimpleRequest):
    """
    Match user questions with master questions using semantic similarity.
    
    Args:
        request: Contains master_questions, user_questions, similarity_threshold, and model_key
        
    Returns:
        List of matches with similarity scores and statistics
    """
    global embedding_generator
    
    try:
        logger.info("Starting simple question matching...")
        logger.info(f"Master questions: {len(request.master_questions)}")
        logger.info(f"User questions: {len(request.user_questions)}")
        logger.info(f"Similarity threshold: {request.similarity_threshold}")
        
        # Validate inputs
        if not request.master_questions:
            raise HTTPException(
                status_code=400, detail="Master questions list cannot be empty"
            )
        
        if not request.user_questions:
            raise HTTPException(
                status_code=400, detail="User questions list cannot be empty"
            )

        if not 0 <= request.similarity_threshold <= 1:
            raise HTTPException(
                status_code=400, detail="Similarity threshold must be between 0 and 1"
            )

        # Initialize embedding generator if needed or switch model
        if embedding_generator is None or embedding_generator.model_key != request.model_key:
            if request.model_key:
                embedding_generator = EmbeddingGenerator(model_key=request.model_key)
            else:
                if embedding_generator is None:
                    embedding_generator = EmbeddingGenerator()

        # Generate embeddings for master questions
        logger.info("Generating embeddings for master questions...")
        master_embeddings = embedding_generator.generate_embeddings_batch(
            request.master_questions, 
            show_progress=False
        )
        
        # Generate embeddings for user questions
        logger.info("Generating embeddings for user questions...")
        user_embeddings = embedding_generator.generate_embeddings_batch(
            request.user_questions, 
            show_progress=False
        )

        # Calculate similarity matrix between user and master questions
        logger.info("Calculating similarity matrix...")
        similarity_matrix = cosine_similarity(user_embeddings, master_embeddings)

        # Find matches
        matches = []
        for user_idx, user_question in enumerate(request.user_questions):
            # Find the best matching master question
            similarities = similarity_matrix[user_idx]
            best_master_idx = np.argmax(similarities)
            best_similarity = similarities[best_master_idx]
            
            # Check if similarity meets threshold
            if best_similarity >= request.similarity_threshold:
                matches.append(MatchedQuestion(
                    user_question=user_question,
                    matched_master_question=request.master_questions[best_master_idx],
                    similarity_score=float(best_similarity)
                ))

        # Calculate statistics
        total_matches = len(matches)
        match_percentage = (total_matches / len(request.user_questions)) * 100 if request.user_questions else 0

        logger.info(f"Found {total_matches} matches out of {len(request.user_questions)} user questions")

        return QuestionMatchSimpleResponse(
            matches=matches,
            total_user_questions=len(request.user_questions),
            total_master_questions=len(request.master_questions),
            total_matches=total_matches,
            match_percentage=match_percentage,
            similarity_threshold=request.similarity_threshold
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Question matching failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Question matching failed: {str(e)}"
        )

# Utility function for batch processing
@app.post("/analyze-similar-questions")
async def analyze_similar_questions(
    questions: List[str],
    similarity_threshold: float = 0.75,
    model_key: Optional[str] = "bge-large-en"
):
    """
    Analyze a list of questions to find similar pairs (like the Streamlit app).
    
    Args:
        questions: List of questions to analyze
        similarity_threshold: Minimum similarity score for a match
        model_key: Embedding model to use
        
    Returns:
        List of similar question pairs with similarity scores
    """
    global embedding_generator
    
    try:
        logger.info(f"Analyzing {len(questions)} questions for similarity...")
        
        if len(questions) < 2:
            raise HTTPException(
                status_code=400, detail="Need at least 2 questions to analyze"
            )

        # Initialize embedding generator if needed
        if embedding_generator is None or embedding_generator.model_key != model_key:
            if model_key:
                embedding_generator = EmbeddingGenerator(model_key=model_key)
            else:
                if embedding_generator is None:
                    embedding_generator = EmbeddingGenerator()

        # Generate embeddings
        embeddings = embedding_generator.generate_embeddings_batch(
            questions, 
            show_progress=False
        )

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Find similar pairs
        similar_pairs = []
        for i in range(len(questions)):
            for j in range(i + 1, len(questions)):
                similarity_score = similarity_matrix[i][j]
                if similarity_score >= similarity_threshold:
                    similar_pairs.append({
                        "question_1": questions[i],
                        "question_2": questions[j],
                        "similarity_score": float(similarity_score)
                    })

        logger.info(f"Found {len(similar_pairs)} similar question pairs")

        return {
            "similar_pairs": similar_pairs,
            "total_questions": len(questions),
            "total_pairs_found": len(similar_pairs),
            "similarity_threshold": similarity_threshold,
            "model_used": embedding_generator.model_key if embedding_generator else model_key
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similarity analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Similarity analysis failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Simple Question Matcher API...")
    print("📍 Server will be available at: http://127.0.0.1:8001")
    print("📋 API Documentation at: http://127.0.0.1:8001/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info"
    )