"""
Embeddings module for generating text embeddings using various sentence transformer models.
Supports multiple models with automatic downloading and caching functionality.
"""

import os
import logging
import shutil
import numpy as np
from typing import List, Union, Optional, Dict, Any
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Available embedding models with their configurations
AVAILABLE_MODELS = {
    "bge-large-en": {
        "model_name": "BAAI/bge-large-en-v1.5",
        "description": "Best overall performance",
        "dimension": 1024,
    },
    "gte-large": {
        "model_name": "thenlper/gte-large",
        "description": "Excellent speed",
        "dimension": 1024,
    },
    "e5-large-v2": {
        "model_name": "intfloat/e5-large-v2",
        "description": "Fast & reliable",
        "dimension": 1024,
    },
    "bge-m3": {
        "model_name": "BAAI/bge-m3",
        "description": "Multilingual support",
        "dimension": 1024,
    },
}


class EmbeddingGenerator:
    """
    A class to handle text embedding generation using various sentence transformer models.
    """

    def __init__(
        self, model_key: str = "bge-large-en", models_dir: str = "embeddingmodels"
    ):
        """
        Initialize the embedding generator.

        Args:
            model_key (str): Key for the model to use (from AVAILABLE_MODELS)
            models_dir (str): Directory to store downloaded models
        """
        self.models_dir = Path(models_dir)
        self.model_key = model_key
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Validate model key
        if model_key not in AVAILABLE_MODELS:
            raise ValueError(
                f"Model key '{model_key}' not found. Available models: {list(AVAILABLE_MODELS.keys())}"
            )

        self.model_config = AVAILABLE_MODELS[model_key]
        self.model_name = self.model_config["model_name"]

        # Create models directory if it doesn't exist
        self._ensure_models_directory()

        # Load the model
        self._load_model()

    def _ensure_models_directory(self):
        """Create the models directory if it doesn't exist."""
        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created models directory: {self.models_dir}")

    def _load_model(self):
        """Load the sentence transformer model with retry logic."""
        model_path = self.models_dir / self.model_key
        max_retries = 2

        for attempt in range(max_retries + 1):
            try:
                # Check if model exists locally and is valid
                if model_path.exists() and self._is_valid_model_directory(model_path):
                    logger.info(f"Loading model from local cache: {model_path}")
                    try:
                        self.model = SentenceTransformer(str(model_path), device=self.device)
                        logger.info(f"Model loaded successfully from cache on device: {self.device}")
                        logger.info(f"Model dimension: {self.model_config['dimension']}")
                        return
                    except Exception as cache_error:
                        logger.warning(f"Failed to load cached model (attempt {attempt + 1}): {cache_error}")
                        if attempt < max_retries:
                            logger.info("Removing corrupted cache and retrying...")
                            if model_path.exists():
                                shutil.rmtree(model_path, ignore_errors=True)
                        else:
                            raise cache_error

                # Download model from HuggingFace
                logger.info(f"Downloading model: {self.model_name}")
                logger.info("This may take a few minutes for the first download...")
                
                self.model = SentenceTransformer(self.model_name, device=self.device)

                # Save model locally for future use
                try:
                    logger.info(f"Saving model to: {model_path}")
                    self.model.save(str(model_path))
                    logger.info("Model saved successfully to local cache")
                except Exception as save_error:
                    logger.warning(f"Failed to save model to cache: {save_error}")
                    # Continue even if saving fails

                logger.info(f"Model loaded successfully on device: {self.device}")
                logger.info(f"Model dimension: {self.model_config['dimension']}")
                return

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Error loading model (attempt {attempt + 1}): {str(e)}")
                    logger.info(f"Retrying... ({attempt + 1}/{max_retries})")
                else:
                    logger.error(f"Failed to load model {self.model_name} after {max_retries + 1} attempts: {str(e)}")
                    raise

    def _is_valid_model_directory(self, model_path: Path) -> bool:
        """Check if the model directory contains valid model files."""
        # Check for config.json (always required)
        if not (model_path / "config.json").exists():
            return False
        
        # Check for either pytorch_model.bin or model.safetensors
        has_pytorch_model = (model_path / "pytorch_model.bin").exists()
        has_safetensors_model = (model_path / "model.safetensors").exists()
        
        return has_pytorch_model or has_safetensors_model

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available models."""
        return AVAILABLE_MODELS.copy()

    def list_downloaded_models(self) -> List[str]:
        """List all models that are downloaded locally."""
        downloaded = []
        for model_key in AVAILABLE_MODELS.keys():
            model_path = self.models_dir / model_key
            if model_path.exists() and self._is_valid_model_directory(model_path):
                downloaded.append(model_key)
        return downloaded

    def switch_model(self, model_key: str):
        """
        Switch to a different model.

        Args:
            model_key (str): Key for the new model to use
        """
        if model_key not in AVAILABLE_MODELS:
            raise ValueError(
                f"Model key '{model_key}' not found. Available models: {list(AVAILABLE_MODELS.keys())}"
            )

        if model_key == self.model_key:
            logger.info(f"Model '{model_key}' is already loaded")
            return

        logger.info(f"Switching from '{self.model_key}' to '{model_key}'")
        self.model_key = model_key
        self.model_config = AVAILABLE_MODELS[model_key]
        self.model_name = self.model_config["model_name"]
        self._load_model()

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text (str): Input text to embed

        Returns:
            np.ndarray: Embedding vector
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Please initialize the EmbeddingGenerator properly."
            )

        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        try:
            # Generate embedding
            embedding = self.model.encode(
                text.strip(), convert_to_numpy=True, show_progress_bar=False
            )
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def generate_embeddings_batch(
        self, texts: List[str], batch_size: int = 32, show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts (List[str]): List of input texts to embed
            batch_size (int): Batch size for processing
            show_progress (bool): Whether to show progress bar

        Returns:
            np.ndarray: Array of embedding vectors
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Please initialize the EmbeddingGenerator properly."
            )

        if not texts:
            raise ValueError("Input texts list cannot be empty")

        # Filter out empty texts
        valid_texts = [text.strip() for text in texts if text and text.strip()]

        if not valid_texts:
            raise ValueError("No valid texts found after filtering empty strings")

        try:
            logger.info(
                f"Generating embeddings for {len(valid_texts)} texts using {self.model_key}"
            )

            # Generate embeddings in batches
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                device=self.device,
            )

            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the current model's embeddings."""
        return self.model_config["dimension"]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model."""
        return {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "description": self.model_config["description"],
            "dimension": self.model_config["dimension"],
            "device": self.device,
            "is_loaded": self.model is not None,
        }


# Utility functions
def create_embedding_generator(
    model_key: str = "bge-large-en", models_dir: str = "embeddingmodels"
) -> EmbeddingGenerator:
    """
    Factory function to create an EmbeddingGenerator instance.

    Args:
        model_key (str): Key for the model to use
        models_dir (str): Directory to store models

    Returns:
        EmbeddingGenerator: Initialized embedding generator
    """
    return EmbeddingGenerator(model_key=model_key, models_dir=models_dir)


def get_available_models_info() -> Dict[str, Dict[str, Any]]:
    """Get information about all available models without initializing them."""
    return AVAILABLE_MODELS.copy()


def compare_embeddings(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.

    Args:
        embedding1 (np.ndarray): First embedding vector
        embedding2 (np.ndarray): Second embedding vector

    Returns:
        float: Cosine similarity score (-1 to 1)
    """
    # Normalize the vectors
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
    return float(similarity)


def save_embeddings(embeddings: np.ndarray, filepath: str):
    """
    Save embeddings to a file.

    Args:
        embeddings (np.ndarray): Embeddings array to save
        filepath (str): Path to save the embeddings
    """
    try:
        np.save(filepath, embeddings)
        logger.info(f"Embeddings saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving embeddings: {str(e)}")
        raise


def load_embeddings(filepath: str) -> np.ndarray:
    """
    Load embeddings from a file.

    Args:
        filepath (str): Path to load the embeddings from

    Returns:
        np.ndarray: Loaded embeddings array
    """
    try:
        embeddings = np.load(filepath)
        logger.info(f"Embeddings loaded from {filepath}")
        return embeddings
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        raise


# Question Matching Functions
def parse_master_questions(master_questions_text: List[str]) -> List[Dict[str, str]]:
    """
    Parse master questions from the custom format.
    
    Supports two formats:
    Format 1: "Why is my animation lagging?<br>SUB_TOPIC_JS_PERFORMANCE"
    Format 2: "How do I center a div?\tSUB_TOPIC_CSS_LAYOUT"
    
    Args:
        master_questions_text (List[str]): List of master question entries
        
    Returns:
        List[Dict[str, str]]: List of parsed questions with 'question' and 'topic' keys
    """
    parsed_questions = []
    
    for entry in master_questions_text:
        if not entry or not entry.strip():
            continue
            
        entry = entry.strip()
        
        # Try Format 1: question<br>topic
        if '<br>' in entry:
            parts = entry.split('<br>', 1)
            if len(parts) == 2:
                question = parts[0].strip()
                topic = parts[1].strip()
                parsed_questions.append({
                    'question': question,
                    'topic': topic,
                    'original': entry
                })
                continue
        
        # Try Format 2: question\ttopic
        if '\t' in entry:
            parts = entry.split('\t', 1)
            if len(parts) == 2:
                question = parts[0].strip()
                topic = parts[1].strip()
                parsed_questions.append({
                    'question': question,
                    'topic': topic,
                    'original': entry
                })
                continue
        
        # If no separator found, treat entire entry as question
        parsed_questions.append({
            'question': entry,
            'topic': '',
            'original': entry
        })
    
    return parsed_questions


def find_question_matches(
    master_questions: List[str], 
    user_questions: List[str], 
    similarity_threshold: float = 0.8,
    model_key: str = "bge-large-en"
) -> Dict[str, Any]:
    """
    Find matches between master questions and user questions using embeddings.
    
    Args:
        master_questions (List[str]): Master questions in custom format
        user_questions (List[str]): User questions to match
        similarity_threshold (float): Minimum similarity score for a match
        model_key (str): Embedding model to use
        
    Returns:
        Dict[str, Any]: Results containing matches and statistics
    """
    # Parse master questions
    parsed_master = parse_master_questions(master_questions)
    master_question_texts = [item['question'] for item in parsed_master]
    
    if not master_question_texts or not user_questions:
        return {
            'matches': [],
            'statistics': {
                'total_user_questions': len(user_questions),
                'total_master_questions': len(master_question_texts),
                'total_matches': 0,
                'match_percentage': 0.0
            },
            'parsed_master_questions': parsed_master
        }
    
    # Initialize embedding generator
    generator = create_embedding_generator(model_key=model_key)
    
    # Generate embeddings
    logger.info(f"Generating embeddings for {len(master_question_texts)} master questions...")
    master_embeddings = generator.generate_embeddings_batch(master_question_texts, show_progress=True)
    
    logger.info(f"Generating embeddings for {len(user_questions)} user questions...")
    user_embeddings = generator.generate_embeddings_batch(user_questions, show_progress=True)
    
    # Find matches
    matches = []
    for user_idx, user_question in enumerate(user_questions):
        user_embedding = user_embeddings[user_idx]
        
        best_match = None
        best_similarity = -1
        best_master_idx = -1
        
        # Compare with all master questions
        for master_idx, master_embedding in enumerate(master_embeddings):
            similarity = compare_embeddings(user_embedding, master_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = parsed_master[master_idx]
                best_master_idx = master_idx
        
        # Add match if above threshold
        if best_similarity >= similarity_threshold:
            matches.append({
                'user_question': user_question,
                'user_index': user_idx,
                'master_question': best_match['question'],
                'master_topic': best_match['topic'],
                'master_original': best_match['original'],
                'master_index': best_master_idx,
                'similarity_score': float(best_similarity)
            })
    
    # Calculate statistics
    total_matches = len(matches)
    match_percentage = (total_matches / len(user_questions)) * 100 if user_questions else 0
    
    results = {
        'matches': matches,
        'statistics': {
            'total_user_questions': len(user_questions),
            'total_master_questions': len(master_question_texts),
            'total_matches': total_matches,
            'match_percentage': match_percentage,
            'similarity_threshold': similarity_threshold,
            'model_used': model_key
        },
        'parsed_master_questions': parsed_master
    }
    
    logger.info(f"Found {total_matches} matches out of {len(user_questions)} user questions ({match_percentage:.1f}%)")
    
    return results


# Example usage and demo function
def demo_embedding_generation():
    """
    Demonstration function showing how to use the EmbeddingGenerator.
    """
    print("=== Embedding Generation Demo ===\n")

    # Show available models
    print("Available models:")
    for key, config in AVAILABLE_MODELS.items():
        print(f"  {key}: {config['model_name']} - {config['description']}")
    print()

    # Create embedding generator with default model
    print("Creating embedding generator with default model (bge-large-en)...")
    generator = create_embedding_generator()

    # Show model info
    info = generator.get_model_info()
    print(f"Loaded model: {info['model_name']}")
    print(f"Device: {info['device']}")
    print(f"Dimension: {info['dimension']}")
    print()

    # Generate single embedding
    sample_text = "This is a sample text for embedding generation."
    print(f"Generating embedding for: '{sample_text}'")
    embedding = generator.generate_embedding(sample_text)
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")
    print()

    # Generate batch embeddings
    sample_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing deals with text analysis.",
        "Deep learning uses neural networks with multiple layers.",
    ]
    print(f"Generating embeddings for {len(sample_texts)} texts...")
    embeddings = generator.generate_embeddings_batch(sample_texts, show_progress=False)
    print(f"Batch embeddings shape: {embeddings.shape}")
    print()

    # Compare embeddings
    similarity = compare_embeddings(embeddings[0], embeddings[1])
    print(f"Similarity between first two texts: {similarity:.4f}")
    print()

    print("Demo completed successfully!")


def demo_question_matching():
    """
    Demonstration function showing how to use the question matching functionality.
    """
    print("=== Question Matching Demo ===\n")
    
    # Sample master questions in different formats
    master_questions = [
        "Why is my animation lagging?<br>SUB_TOPIC_JS_PERFORMANCE",
        "How do I center a div?\tSUB_TOPIC_CSS_LAYOUT",
        "What is a closure in JavaScript?<br>SUB_TOPIC_JS_FUNDAMENTALS",
        "How to make responsive design?\tSUB_TOPIC_CSS_RESPONSIVE"
    ]
    
    # Sample user questions to match
    user_questions = [
        "My CSS animations are slow, what can I do?",
        "How can I center an element horizontally?",
        "What are JavaScript closures?",
        "How to create a mobile-friendly layout?",
        "What is machine learning?"  # This one shouldn't match
    ]
    
    print("Master Questions:")
    for i, q in enumerate(master_questions, 1):
        print(f"  {i}. {q}")
    print()
    
    print("User Questions:")
    for i, q in enumerate(user_questions, 1):
        print(f"  {i}. {q}")
    print()
    
    # Find matches
    print("Finding matches...")
    results = find_question_matches(
        master_questions=master_questions,
        user_questions=user_questions,
        similarity_threshold=0.7
    )
    
    # Display results
    print("\n=== RESULTS ===")
    print(f"Statistics:")
    stats = results['statistics']
    print(f"  Total user questions: {stats['total_user_questions']}")
    print(f"  Total master questions: {stats['total_master_questions']}")
    print(f"  Total matches found: {stats['total_matches']}")
    print(f"  Match percentage: {stats['match_percentage']:.1f}%")
    print()
    
    print("Matches found:")
    for i, match in enumerate(results['matches'], 1):
        print(f"  {i}. User: \"{match['user_question']}\"")
        print(f"     Master: \"{match['master_question']}\"")
        print(f"     Topic: {match['master_topic']}")
        print(f"     Similarity: {match['similarity_score']:.3f}")
        print()
    
    print("Demo completed successfully!")


def ensure_model_downloaded(model_key: str = "bge-large-en", models_dir: str = "embeddingmodels") -> bool:
    """
    Ensure a specific model is downloaded and available.
    
    Args:
        model_key (str): Key for the model to download
        models_dir (str): Directory to store models
        
    Returns:
        bool: True if model is available, False if download failed
    """
    try:
        logger.info(f"Checking model availability: {model_key}")
        generator = EmbeddingGenerator(model_key=model_key, models_dir=models_dir)
        logger.info(f"Model {model_key} is ready for use")
        return True
    except Exception as e:
        logger.error(f"Failed to ensure model {model_key} is available: {str(e)}")
        return False


def preload_default_models(models_dir: str = "embeddingmodels") -> Dict[str, bool]:
    """
    Preload all available models to ensure they're downloaded.
    
    Args:
        models_dir (str): Directory to store models
        
    Returns:
        Dict[str, bool]: Status of each model download
    """
    results = {}
    logger.info("Starting preload of all available models...")
    
    for model_key in AVAILABLE_MODELS.keys():
        logger.info(f"Preloading model: {model_key}")
        results[model_key] = ensure_model_downloaded(model_key, models_dir)
        
    successful_downloads = sum(results.values())
    total_models = len(results)
    
    logger.info(f"Preload completed: {successful_downloads}/{total_models} models ready")
    return results


if __name__ == "__main__":
    # Run both demos
    demo_embedding_generation()
    print("\n" + "="*50 + "\n")
    demo_question_matching()
