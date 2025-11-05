"""
FastAPI Application for Syllabus Checker

Provides REST API endpoints for:
1. Health check for LLM and embeddings
2. Similarity checker (Excel + Master questions)
3. Syllabus checker with text syllabus
4. Syllabus checker with .txt file syllabus

All processed files are saved to data_process folder and cleaned up after completion.
"""

import os
import shutil
import uuid
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd

# Import our custom modules
from syllabus_check import create_syllabus_checker, SyllabusChecker
from similarity import create_similarity_checker, QuestionSimilarityChecker
from embeddings import EmbeddingGenerator, find_question_matches

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Syllabus Checker API",
    description="API for processing question banks with similarity checking and syllabus filtering",
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

# Create data_process directory
DATA_PROCESS_DIR = "data_process"
os.makedirs(DATA_PROCESS_DIR, exist_ok=True)

# Global instances for health checking
global_similarity_checker = None
global_syllabus_checker = None
global_embedding_generator = None


def initialize_global_instances():
    """Initialize global instances for health checking."""
    global global_similarity_checker, global_syllabus_checker, global_embedding_generator
    try:
        global_similarity_checker = create_similarity_checker()
        global_syllabus_checker = create_syllabus_checker()
        global_embedding_generator = EmbeddingGenerator()
        logger.info("Global instances initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize global instances: {str(e)}")


def generate_unique_filename(original_filename: str, suffix: str = "") -> str:
    """
    Generate unique filename with 10-digit unique identifier.

    Args:
        original_filename (str): Original uploaded filename
        suffix (str): Optional suffix to add

    Returns:
        str: Unique filename
    """
    # Generate 10-digit unique ID
    unique_id = str(uuid.uuid4().hex)[:10]

    # Extract name and extension
    base_name = Path(original_filename).stem
    extension = Path(original_filename).suffix

    # Create unique filename
    if suffix:
        unique_filename = f"{base_name}_{suffix}_{unique_id}{extension}"
    else:
        unique_filename = f"{base_name}_{unique_id}{extension}"

    return unique_filename


def save_uploaded_file(upload_file: UploadFile, directory: str) -> str:
    """
    Save uploaded file to specified directory.

    Args:
        upload_file (UploadFile): Uploaded file
        directory (str): Directory to save file

    Returns:
        str: Path to saved file
    """
    try:
        # Create unique filename
        unique_filename = generate_unique_filename(upload_file.filename)
        file_path = os.path.join(directory, unique_filename)

        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

        logger.info(f"File saved: {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to save uploaded file: {str(e)}"
        )


def cleanup_files(*file_paths):
    """
    Clean up specified files.

    Args:
        *file_paths: Variable number of file paths to delete
    """
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup file {file_path}: {str(e)}")


def cleanup_folder(folder_path: str):
    """
    Clean up all files in the specified folder.

    Args:
        folder_path (str): Path to folder to clean
    """
    try:
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info(f"Cleaned up folder: {folder_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup folder {folder_path}: {str(e)}")


# Pydantic models for request/response
class HealthCheckResponse(BaseModel):
    status: str
    llm_health: Dict[str, Any]
    embeddings_health: Dict[str, Any]
    timestamp: str


# API Endpoints


@app.on_event("startup")
async def startup_event():
    """Initialize global instances on startup and ensure models are available."""
    logger.info("🚀 Starting Syllabus Checker API...")
    
    # Ensure default embedding model is available
    try:
        from embeddings import ensure_model_downloaded
        logger.info("📦 Checking embedding model availability...")
        model_ready = ensure_model_downloaded()
        if model_ready:
            logger.info("✅ Default embedding model is ready")
        else:
            logger.warning("⚠️ Default embedding model download failed, but continuing startup")
    except Exception as e:
        logger.warning(f"⚠️ Model check failed: {str(e)}, but continuing startup")
    
    # Initialize global instances
    initialize_global_instances()
    logger.info("🎯 Syllabus Checker API startup completed")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Syllabus Checker API",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/similarity-check",
            "/question-match",
            "/syllabus-check-text",
            "/syllabus-check-file",
            "/question-matcher.html"
        ],
        "note": "All processing endpoints return Excel files directly for download",
    }


@app.get("/question-matcher.html", response_class=HTMLResponse)
async def question_matcher_ui():
    """Serve the Question Matcher HTML interface."""
    try:
        html_file = Path("question_matcher.html")
        if html_file.exists():
            return html_file.read_text(encoding='utf-8')
        else:
            raise HTTPException(status_code=404, detail="Question matcher UI not found")
    except Exception as e:
        logger.error(f"Error serving question matcher UI: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading question matcher UI")


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint for LLM and embeddings.

    Returns:
        HealthCheckResponse: Health status of all components
    """
    try:
        logger.info("Performing health check...")

        # Check embeddings health
        embeddings_health = {"status": "unknown", "error": None}
        try:
            if global_embedding_generator is None:
                initialize_global_instances()

            # Test embedding generation
            test_embeddings = global_embedding_generator.generate_embeddings_batch(
                ["Test question for health check"], batch_size=1, show_progress=False
            )

            if test_embeddings is not None and len(test_embeddings) > 0:
                embeddings_health = {
                    "status": "healthy",
                    "model": global_embedding_generator.model_key,
                    "embedding_dim": (
                        test_embeddings.shape[1]
                        if len(test_embeddings.shape) > 1
                        else len(test_embeddings[0])
                    ),
                }
            else:
                embeddings_health = {
                    "status": "unhealthy",
                    "error": "Failed to generate embeddings",
                }

        except Exception as e:
            embeddings_health = {"status": "unhealthy", "error": str(e)}

        # Check similarity checker health
        llm_health = {"status": "unknown", "error": None}
        try:
            if global_similarity_checker is None:
                initialize_global_instances()

            # Test similarity checking
            test_questions = ["What is AI?", "What is artificial intelligence?"]
            similarity_scores = global_similarity_checker.find_similar_questions(
                test_questions[:1], test_questions[1:]
            )

            llm_health = {
                "status": "healthy",
                "similarity_threshold": global_similarity_checker.similarity_threshold,
                "model": global_similarity_checker.embedding_generator.model_key,
            }

        except Exception as e:
            llm_health = {"status": "unhealthy", "error": str(e)}

        # Overall status
        overall_status = (
            "healthy"
            if (
                embeddings_health["status"] == "healthy"
                and llm_health["status"] == "healthy"
            )
            else "unhealthy"
        )

        return HealthCheckResponse(
            status=overall_status,
            llm_health=llm_health,
            embeddings_health=embeddings_health,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/similarity-check")
async def similarity_check(
    background_tasks: BackgroundTasks,
    excel_file: UploadFile = File(..., description="Excel file with questions"),
    master_questions: List[str] = Form(..., description="List of master questions"),
    question_column: str = Form(
        "Question", description="Name of question column in Excel"
    ),
    similarity_threshold: float = Form(0.8, description="Similarity threshold (0-1)"),
):
    """
    Similarity checker endpoint - removes questions similar to master questions.

    Args:
        excel_file: Excel file with questions
        master_questions: List of master questions to compare against
        question_column: Name of column containing questions
        similarity_threshold: Threshold for similarity detection

    Returns:
        FileResponse: Excel file with cleaned questions ready for download
    """
    start_time = datetime.now()
    saved_excel_path = None
    output_file_path = None

    try:
        logger.info("Starting similarity check processing...")

        # Validate file type
        if not excel_file.filename.endswith((".xlsx", ".xls")):
            raise HTTPException(
                status_code=400, detail="File must be Excel format (.xlsx or .xls)"
            )

        # Save uploaded Excel file
        saved_excel_path = save_uploaded_file(excel_file, DATA_PROCESS_DIR)

        # Create similarity checker
        checker = create_similarity_checker(similarity_threshold=similarity_threshold)

        # Process the file
        results = checker.process_excel_file(
            excel_path=saved_excel_path,
            master_questions=master_questions,
            question_column=question_column,
        )

        if not results["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Processing failed: {results.get('error', 'Unknown error')}",
            )

        # Move output file to data_process directory with unique name
        original_output = results["output_file"]
        unique_output_filename = generate_unique_filename(
            excel_file.filename, "similarity_cleaned"
        )
        output_file_path = os.path.join(DATA_PROCESS_DIR, unique_output_filename)

        # Move the file
        shutil.move(original_output, output_file_path)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        # Schedule cleanup of input files and output file after download
        background_tasks.add_task(cleanup_files, saved_excel_path, output_file_path)

        # Return the Excel file directly for download
        return FileResponse(
            path=output_file_path,
            filename=unique_output_filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "X-Processing-Time": str(processing_time),
                "X-Original-Count": str(results["statistics"]["original_count"]),
                "X-Removed-Count": str(results["statistics"]["removed_count"]),
                "X-Remaining-Count": str(results["statistics"]["remaining_count"]),
            },
        )

    except HTTPException:
        # Cleanup on error
        cleanup_files(saved_excel_path, output_file_path)
        raise
    except Exception as e:
        # Cleanup on error
        cleanup_files(saved_excel_path, output_file_path)
        logger.error(f"Similarity check failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Similarity check failed: {str(e)}"
        )


# Request/Response models for question matching
class QuestionMatchRequest(BaseModel):
    master_questions: List[str]
    user_questions: List[str]
    similarity_threshold: float = 0.8
    model_key: str = "bge-large-en"


class QuestionMatchResponse(BaseModel):
    matches: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    parsed_master_questions: List[Dict[str, str]]


@app.post("/question-match", response_model=QuestionMatchResponse)
async def question_match(request: QuestionMatchRequest):
    """
    Find matches between master questions and user questions.
    
    Master questions should be in one of these formats:
    - Format 1: "Why is my animation lagging?<br>SUB_TOPIC_JS_PERFORMANCE" 
    - Format 2: "How do I center a div?\tSUB_TOPIC_CSS_LAYOUT"
    
    Args:
        request: JSON request containing master_questions, user_questions, similarity_threshold, model_key
        
    Returns:
        JSON response with matches, statistics, and parsed master questions
    """
    try:
        logger.info("Starting question matching...")
        logger.info(f"Master questions: {len(request.master_questions)}")
        logger.info(f"User questions: {len(request.user_questions)}")
        logger.info(f"Similarity threshold: {request.similarity_threshold}")
        logger.info(f"Model: {request.model_key}")

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

        # Perform question matching
        results = find_question_matches(
            master_questions=request.master_questions,
            user_questions=request.user_questions,
            similarity_threshold=request.similarity_threshold,
            model_key=request.model_key
        )

        logger.info(f"Question matching completed. Found {results['statistics']['total_matches']} matches.")

        return QuestionMatchResponse(**results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Question matching failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Question matching failed: {str(e)}"
        )


@app.post("/syllabus-check-text")
async def syllabus_check_text(
    background_tasks: BackgroundTasks,
    excel_file: UploadFile = File(..., description="Excel file with questions"),
    master_questions: List[str] = Form(..., description="List of master questions"),
    syllabus_content: str = Form(..., description="Syllabus content as text"),
    question_column: str = Form(
        "Question", description="Name of question column in Excel"
    ),
    similarity_threshold: float = Form(
        0.8, description="Similarity threshold for master questions"
    ),
    relevance_threshold: float = Form(
        0.6, description="Relevance threshold for syllabus filtering"
    ),
):
    """
    Syllabus checker with text syllabus content.

    Args:
        excel_file: Excel file with questions
        master_questions: List of master questions
        syllabus_content: Syllabus content as text string
        question_column: Name of question column
        similarity_threshold: Threshold for master question similarity
        relevance_threshold: Threshold for syllabus relevance

    Returns:
        FileResponse: Excel file with cleaned questions ready for download
    """
    start_time = datetime.now()
    saved_excel_path = None
    output_file_path = None

    try:
        logger.info("Starting syllabus check with text content...")

        # Validate file type
        if not excel_file.filename.endswith((".xlsx", ".xls")):
            raise HTTPException(
                status_code=400, detail="File must be Excel format (.xlsx or .xls)"
            )

        # Save uploaded Excel file
        saved_excel_path = save_uploaded_file(excel_file, DATA_PROCESS_DIR)

        # Create syllabus checker
        checker = create_syllabus_checker(
            similarity_threshold=similarity_threshold,
            syllabus_relevance_threshold=relevance_threshold,
        )

        # Generate unique output filename
        unique_output_filename = generate_unique_filename(
            excel_file.filename, "syllabus_cleaned"
        )
        output_file_path = os.path.join(DATA_PROCESS_DIR, unique_output_filename)

        # Process complete pipeline
        results = checker.process_complete_pipeline(
            excel_path=saved_excel_path,
            master_questions=master_questions,
            syllabus_content=syllabus_content,
            output_path=output_file_path,
            question_column=question_column,
        )

        if not results["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Processing failed: {results.get('error', 'Unknown error')}",
            )

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        # Schedule cleanup of input files and output file after download
        background_tasks.add_task(cleanup_files, saved_excel_path, output_file_path)

        # Return the Excel file directly for download
        return FileResponse(
            path=output_file_path,
            filename=unique_output_filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "X-Processing-Time": str(processing_time),
                "X-Original-Count": str(
                    results["overall_statistics"]["original_questions"]
                ),
                "X-Final-Count": str(results["overall_statistics"]["final_questions"]),
                "X-Total-Removed": str(results["overall_statistics"]["total_removed"]),
                "X-Reduction-Percentage": str(
                    results["overall_statistics"]["reduction_percentage"]
                ),
            },
        )

    except HTTPException:
        # Cleanup on error
        cleanup_files(saved_excel_path, output_file_path)
        raise
    except Exception as e:
        # Cleanup on error
        cleanup_files(saved_excel_path, output_file_path)
        logger.error(f"Syllabus check with text failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Syllabus check with text failed: {str(e)}"
        )


@app.post("/syllabus-check-file")
async def syllabus_check_file(
    background_tasks: BackgroundTasks,
    excel_file: UploadFile = File(..., description="Excel file with questions"),
    syllabus_file: UploadFile = File(
        ..., description="Text file with syllabus content"
    ),
    master_questions: List[str] = Form(..., description="List of master questions"),
    question_column: str = Form(
        "Question", description="Name of question column in Excel"
    ),
    similarity_threshold: float = Form(
        0.8, description="Similarity threshold for master questions"
    ),
    relevance_threshold: float = Form(
        0.6, description="Relevance threshold for syllabus filtering"
    ),
):
    """
    Syllabus checker with .txt file syllabus content.

    Args:
        excel_file: Excel file with questions
        syllabus_file: Text file with syllabus content
        master_questions: List of master questions
        question_column: Name of question column
        similarity_threshold: Threshold for master question similarity
        relevance_threshold: Threshold for syllabus relevance

    Returns:
        FileResponse: Excel file with cleaned questions ready for download
    """
    start_time = datetime.now()
    saved_excel_path = None
    saved_syllabus_path = None
    output_file_path = None

    try:
        logger.info("Starting syllabus check with file content...")

        # Validate file types
        if not excel_file.filename.endswith((".xlsx", ".xls")):
            raise HTTPException(
                status_code=400, detail="Excel file must be .xlsx or .xls format"
            )

        if not syllabus_file.filename.endswith(".txt"):
            raise HTTPException(
                status_code=400, detail="Syllabus file must be .txt format"
            )

        # Save uploaded files
        saved_excel_path = save_uploaded_file(excel_file, DATA_PROCESS_DIR)
        saved_syllabus_path = save_uploaded_file(syllabus_file, DATA_PROCESS_DIR)

        # Create syllabus checker
        checker = create_syllabus_checker(
            similarity_threshold=similarity_threshold,
            syllabus_relevance_threshold=relevance_threshold,
        )

        # Generate unique output filename
        unique_output_filename = generate_unique_filename(
            excel_file.filename, "syllabus_cleaned"
        )
        output_file_path = os.path.join(DATA_PROCESS_DIR, unique_output_filename)

        # Process complete pipeline
        results = checker.process_complete_pipeline(
            excel_path=saved_excel_path,
            master_questions=master_questions,
            syllabus_content=saved_syllabus_path,  # Pass file path
            output_path=output_file_path,
            question_column=question_column,
        )

        if not results["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Processing failed: {results.get('error', 'Unknown error')}",
            )

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        # Schedule cleanup of input files and output file after download
        background_tasks.add_task(
            cleanup_files, saved_excel_path, saved_syllabus_path, output_file_path
        )

        # Return the Excel file directly for download
        return FileResponse(
            path=output_file_path,
            filename=unique_output_filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "X-Processing-Time": str(processing_time),
                "X-Original-Count": str(
                    results["overall_statistics"]["original_questions"]
                ),
                "X-Final-Count": str(results["overall_statistics"]["final_questions"]),
                "X-Total-Removed": str(results["overall_statistics"]["total_removed"]),
                "X-Reduction-Percentage": str(
                    results["overall_statistics"]["reduction_percentage"]
                ),
            },
        )

    except HTTPException:
        # Cleanup on error
        cleanup_files(saved_excel_path, saved_syllabus_path, output_file_path)
        raise
    except Exception as e:
        # Cleanup on error
        cleanup_files(saved_excel_path, saved_syllabus_path, output_file_path)
        logger.error(f"Syllabus check with file failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Syllabus check with file failed: {str(e)}"
        )


@app.post("/cleanup")
async def manual_cleanup():
    """
    Manually trigger cleanup of data_process folder.

    Returns:
        dict: Cleanup status
    """
    try:
        cleanup_folder(DATA_PROCESS_DIR)
        return {
            "success": True,
            "message": "Data process folder cleaned successfully",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Manual cleanup failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@app.get("/files")
async def list_files():
    """
    List files in data_process folder.

    Returns:
        dict: List of files
    """
    try:
        files = []
        if os.path.exists(DATA_PROCESS_DIR):
            files = [
                f
                for f in os.listdir(DATA_PROCESS_DIR)
                if os.path.isfile(os.path.join(DATA_PROCESS_DIR, f))
            ]

        return {
            "files": files,
            "count": len(files),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to list files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
