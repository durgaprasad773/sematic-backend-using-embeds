"""
Syllabus Check Application

This module provides a comprehensive pipeline for processing question banks:
1. Load questions from Excel files
2. Remove duplicate/similar questions using master question list
3. Filter questions based on syllabus content relevance
4. Generate cleaned Excel output with only relevant questions

Process Flow:
1. User uploads Excel file with questions
2. User provides master questions list
3. Remove questions similar to master questions (Phase 1 cleanup)
4. User uploads syllabus content (text or .txt file)
5. Remove questions not related to syllabus (Phase 2 cleanup)
6. Export final cleaned Excel file
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
import json
from datetime import datetime

# Import our similarity checker
from similarity import QuestionSimilarityChecker, create_similarity_checker
from embeddings import EmbeddingGenerator, compare_embeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyllabusChecker:
    """
    Main class for comprehensive question bank processing and syllabus-based filtering.
    """

    def __init__(
        self,
        model_key: str = "bge-large-en",
        similarity_threshold: float = 0.8,
        syllabus_relevance_threshold: float = 0.6,
        models_dir: str = "embeddingmodels",
    ):
        """
        Initialize the Syllabus Checker.

        Args:
            model_key (str): Embedding model to use
            similarity_threshold (float): Threshold for master question similarity (0-1)
            syllabus_relevance_threshold (float): Threshold for syllabus relevance (0-1)
            models_dir (str): Directory containing embedding models
        """
        self.similarity_threshold = similarity_threshold
        self.syllabus_relevance_threshold = syllabus_relevance_threshold

        # Initialize similarity checker for master questions comparison
        self.similarity_checker = QuestionSimilarityChecker(
            model_key=model_key,
            similarity_threshold=similarity_threshold,
            models_dir=models_dir,
        )

        # Initialize embedding generator for syllabus comparison
        self.embedding_generator = EmbeddingGenerator(
            model_key=model_key, models_dir=models_dir
        )

        # Cache for embeddings
        self.syllabus_embeddings_cache = {}

        logger.info(f"SyllabusChecker initialized with model: {model_key}")
        logger.info(f"Master question similarity threshold: {similarity_threshold}")
        logger.info(f"Syllabus relevance threshold: {syllabus_relevance_threshold}")

    def load_excel_questions(
        self,
        excel_path: str,
        question_column: str = "Question",
        sheet_name: Union[str, int] = 0,
    ) -> pd.DataFrame:
        """
        Load questions from Excel file.

        Args:
            excel_path (str): Path to Excel file
            question_column (str): Name of column containing questions
            sheet_name (Union[str, int]): Sheet name or index

        Returns:
            pd.DataFrame: Loaded DataFrame with questions
        """
        try:
            logger.info(f"Loading Excel file: {excel_path}")
            df = self.similarity_checker.load_excel_file(
                excel_path, question_column, sheet_name
            )
            return df
        except Exception as e:
            logger.error(f"Error loading Excel questions: {str(e)}")
            raise

    def load_master_questions(
        self, master_source: Union[str, List[str]], source_type: str = "auto"
    ) -> List[str]:
        """
        Load master questions from various sources.

        Args:
            master_source (Union[str, List[str]]): Source of master questions
            source_type (str): Type of source ('file', 'excel', 'list', 'auto')

        Returns:
            List[str]: List of master questions
        """
        try:
            if source_type == "list" or isinstance(master_source, list):
                logger.info(
                    f"Using provided list of {len(master_source)} master questions"
                )
                return master_source

            elif source_type == "file" or (
                isinstance(master_source, str) and master_source.endswith(".txt")
            ):
                logger.info(f"Loading master questions from text file: {master_source}")
                return self._load_master_questions_from_txt(master_source)

            elif source_type == "excel" or (
                isinstance(master_source, str)
                and master_source.endswith((".xlsx", ".xls"))
            ):
                logger.info(
                    f"Loading master questions from Excel file: {master_source}"
                )
                return self._load_master_questions_from_excel(master_source)

            else:
                # Auto-detect based on file extension
                if isinstance(master_source, str):
                    if master_source.endswith(".txt"):
                        return self._load_master_questions_from_txt(master_source)
                    elif master_source.endswith((".xlsx", ".xls")):
                        return self._load_master_questions_from_excel(master_source)

                raise ValueError(
                    f"Unable to determine master questions source type: {master_source}"
                )

        except Exception as e:
            logger.error(f"Error loading master questions: {str(e)}")
            raise

    def _load_master_questions_from_txt(self, file_path: str) -> List[str]:
        """Load master questions from text file (one per line)."""
        with open(file_path, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(questions)} master questions from text file")
        return questions

    def _load_master_questions_from_excel(
        self, excel_path: str, question_column: str = "Question"
    ) -> List[str]:
        """Load master questions from Excel file."""
        df = pd.read_excel(excel_path)
        if question_column not in df.columns:
            raise ValueError(
                f"Column '{question_column}' not found in master questions Excel"
            )

        questions = df[question_column].dropna().astype(str).tolist()
        questions = [q.strip() for q in questions if q.strip()]
        logger.info(f"Loaded {len(questions)} master questions from Excel file")
        return questions

    def remove_similar_to_master(
        self,
        df: pd.DataFrame,
        master_questions: List[str],
        question_column: str = "Question",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Phase 1: Remove questions similar to master questions.

        Args:
            df (pd.DataFrame): DataFrame with questions
            master_questions (List[str]): Master questions list
            question_column (str): Name of question column

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame and statistics
        """
        try:
            logger.info(
                "=== Phase 1: Removing questions similar to master questions ==="
            )

            filtered_df, stats = self.similarity_checker.remove_similar_questions(
                df, master_questions, question_column
            )

            logger.info(f"Phase 1 complete: {stats['removed_count']} questions removed")
            return filtered_df, stats

        except Exception as e:
            logger.error(f"Error in Phase 1 processing: {str(e)}")
            raise

    def load_syllabus_content(self, syllabus_source: str) -> str:
        """
        Load syllabus content from text file or direct text.

        Args:
            syllabus_source (str): Path to syllabus file or direct text content

        Returns:
            str: Syllabus content
        """
        try:
            # Check if it's a file path
            if os.path.exists(syllabus_source):
                logger.info(f"Loading syllabus from file: {syllabus_source}")
                with open(syllabus_source, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                logger.info(f"Loaded syllabus content ({len(content)} characters)")
                return content
            else:
                # Treat as direct text content
                logger.info(
                    f"Using provided syllabus text ({len(syllabus_source)} characters)"
                )
                return syllabus_source.strip()

        except Exception as e:
            logger.error(f"Error loading syllabus content: {str(e)}")
            raise

    def check_syllabus_relevance(
        self, questions: List[str], syllabus_content: str
    ) -> Dict[int, Dict[str, Any]]:
        """
        Check relevance of questions against syllabus content.

        Args:
            questions (List[str]): List of questions to check
            syllabus_content (str): Syllabus content

        Returns:
            Dict[int, Dict[str, Any]]: Relevance scores and details
        """
        try:
            logger.info("Checking question relevance against syllabus...")

            # Generate embeddings for questions
            question_embeddings = self.embedding_generator.generate_embeddings_batch(
                questions, batch_size=32, show_progress=True
            )

            # Generate embedding for syllabus content
            # Split syllabus into chunks if too long
            syllabus_chunks = self._split_syllabus_content(syllabus_content)
            syllabus_embeddings = self.embedding_generator.generate_embeddings_batch(
                syllabus_chunks, batch_size=16, show_progress=True
            )

            relevance_scores = {}

            for q_idx, question_embedding in enumerate(question_embeddings):
                max_relevance = -1
                best_chunk_idx = -1

                # Compare with all syllabus chunks and take maximum similarity
                for chunk_idx, syllabus_embedding in enumerate(syllabus_embeddings):
                    relevance = compare_embeddings(
                        question_embedding, syllabus_embedding
                    )
                    if relevance > max_relevance:
                        max_relevance = relevance
                        best_chunk_idx = chunk_idx

                relevance_scores[q_idx] = {
                    "question": questions[q_idx],
                    "relevance_score": max_relevance,
                    "is_relevant": max_relevance >= self.syllabus_relevance_threshold,
                    "best_matching_chunk": (
                        syllabus_chunks[best_chunk_idx] if best_chunk_idx >= 0 else ""
                    ),
                    "chunk_index": best_chunk_idx,
                }

            relevant_count = sum(
                1 for score in relevance_scores.values() if score["is_relevant"]
            )
            logger.info(
                f"Relevance check complete: {relevant_count}/{len(questions)} questions are relevant"
            )

            return relevance_scores

        except Exception as e:
            logger.error(f"Error checking syllabus relevance: {str(e)}")
            raise

    def _split_syllabus_content(
        self, content: str, max_chunk_length: int = 1000
    ) -> List[str]:
        """
        Split syllabus content into manageable chunks for embedding generation.

        Args:
            content (str): Syllabus content
            max_chunk_length (int): Maximum characters per chunk

        Returns:
            List[str]: List of content chunks
        """
        if len(content) <= max_chunk_length:
            return [content]

        # Split by paragraphs first
        paragraphs = content.split("\n\n")
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk + paragraph) <= max_chunk_length:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        # If chunks are still too long, split by sentences
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_chunk_length:
                final_chunks.append(chunk)
            else:
                sentences = chunk.split(". ")
                current_sentence_chunk = ""
                for sentence in sentences:
                    if len(current_sentence_chunk + sentence) <= max_chunk_length:
                        current_sentence_chunk += sentence + ". "
                    else:
                        if current_sentence_chunk:
                            final_chunks.append(current_sentence_chunk.strip())
                        current_sentence_chunk = sentence + ". "
                if current_sentence_chunk:
                    final_chunks.append(current_sentence_chunk.strip())

        logger.info(f"Split syllabus content into {len(final_chunks)} chunks")
        return final_chunks

    def filter_by_syllabus_relevance(
        self,
        df: pd.DataFrame,
        syllabus_content: str,
        question_column: str = "Question",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Phase 2: Filter questions based on syllabus relevance.

        Args:
            df (pd.DataFrame): DataFrame with questions
            syllabus_content (str): Syllabus content
            question_column (str): Name of question column

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Filtered DataFrame and statistics
        """
        try:
            logger.info("=== Phase 2: Filtering questions by syllabus relevance ===")

            questions = df[question_column].astype(str).tolist()
            original_count = len(df)

            # Check relevance
            relevance_scores = self.check_syllabus_relevance(
                questions, syllabus_content
            )

            # Get indices of irrelevant questions
            irrelevant_indices = [
                idx
                for idx, score in relevance_scores.items()
                if not score["is_relevant"]
            ]

            # Remove irrelevant questions
            filtered_df = df.drop(irrelevant_indices).reset_index(drop=True)
            removed_count = len(irrelevant_indices)
            remaining_count = len(filtered_df)

            # Prepare statistics
            stats = {
                "original_count": original_count,
                "removed_count": removed_count,
                "remaining_count": remaining_count,
                "removal_percentage": (
                    (removed_count / original_count) * 100 if original_count > 0 else 0
                ),
                "relevance_threshold": self.syllabus_relevance_threshold,
                "relevance_details": relevance_scores,
                "irrelevant_questions": {
                    idx: relevance_scores[idx] for idx in irrelevant_indices
                },
            }

            logger.info(f"Phase 2 complete:")
            logger.info(f"  Original questions: {original_count}")
            logger.info(f"  Removed irrelevant: {removed_count}")
            logger.info(f"  Remaining relevant: {remaining_count}")
            logger.info(f"  Removal percentage: {stats['removal_percentage']:.2f}%")

            return filtered_df, stats

        except Exception as e:
            logger.error(f"Error in Phase 2 processing: {str(e)}")
            raise

    def save_final_results(
        self,
        df: pd.DataFrame,
        output_path: str,
        phase1_stats: Dict[str, Any] = None,
        phase2_stats: Dict[str, Any] = None,
    ) -> str:
        """
        Save final filtered results to Excel with comprehensive statistics.

        Args:
            df (pd.DataFrame): Final filtered DataFrame
            output_path (str): Output file path
            phase1_stats (Dict[str, Any]): Phase 1 statistics
            phase2_stats (Dict[str, Any]): Phase 2 statistics

        Returns:
            str: Path to saved file
        """
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # Add timestamp if needed
            if not output_path.endswith((".xlsx", ".xls")):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"{output_path}_final_cleaned_{timestamp}.xlsx"

            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                # Main results sheet
                df.to_excel(writer, sheet_name="Final_Cleaned_Questions", index=False)

                # Combined statistics sheet
                if phase1_stats or phase2_stats:
                    combined_stats = self._create_combined_stats(
                        phase1_stats, phase2_stats
                    )
                    combined_stats.to_excel(
                        writer, sheet_name="Processing_Summary", index=False
                    )

                # Phase 1 details (removed similar questions)
                if phase1_stats and "similar_questions_details" in phase1_stats:
                    phase1_details = self._create_phase1_details(
                        phase1_stats["similar_questions_details"]
                    )
                    phase1_details.to_excel(
                        writer, sheet_name="Phase1_Removed_Similar", index=False
                    )

                # Phase 2 details (removed irrelevant questions)
                if phase2_stats and "irrelevant_questions" in phase2_stats:
                    phase2_details = self._create_phase2_details(
                        phase2_stats["irrelevant_questions"]
                    )
                    phase2_details.to_excel(
                        writer, sheet_name="Phase2_Removed_Irrelevant", index=False
                    )

            logger.info(f"Final results saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error saving final results: {str(e)}")
            raise

    def _create_combined_stats(
        self, phase1_stats: Dict[str, Any], phase2_stats: Dict[str, Any]
    ) -> pd.DataFrame:
        """Create combined statistics DataFrame."""
        stats_data = []

        if phase1_stats:
            stats_data.extend(
                [
                    [
                        "Phase 1 - Original Questions",
                        phase1_stats.get("original_count", 0),
                    ],
                    ["Phase 1 - Removed Similar", phase1_stats.get("removed_count", 0)],
                    [
                        "Phase 1 - Remaining After Cleanup",
                        phase1_stats.get("remaining_count", 0),
                    ],
                    [
                        "Phase 1 - Similarity Threshold",
                        phase1_stats.get("similarity_threshold", 0),
                    ],
                    ["", ""],
                ]
            )

        if phase2_stats:
            stats_data.extend(
                [
                    [
                        "Phase 2 - Input Questions",
                        phase2_stats.get("original_count", 0),
                    ],
                    [
                        "Phase 2 - Removed Irrelevant",
                        phase2_stats.get("removed_count", 0),
                    ],
                    [
                        "Phase 2 - Final Relevant Questions",
                        phase2_stats.get("remaining_count", 0),
                    ],
                    [
                        "Phase 2 - Relevance Threshold",
                        phase2_stats.get("relevance_threshold", 0),
                    ],
                    ["", ""],
                ]
            )

        # Calculate overall statistics
        if phase1_stats and phase2_stats:
            original_total = phase1_stats.get("original_count", 0)
            final_total = phase2_stats.get("remaining_count", 0)
            total_removed = original_total - final_total

            stats_data.extend(
                [
                    ["=== OVERALL SUMMARY ===", ""],
                    ["Total Original Questions", original_total],
                    ["Total Questions Removed", total_removed],
                    ["Final Questions Remaining", final_total],
                    [
                        "Overall Reduction Percentage",
                        (
                            f"{(total_removed/original_total)*100:.2f}%"
                            if original_total > 0
                            else "0%"
                        ),
                    ],
                    ["Processing Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ]
            )

        return pd.DataFrame(stats_data, columns=["Metric", "Value"])

    def _create_phase1_details(
        self, similar_questions: Dict[int, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Create Phase 1 details DataFrame."""
        details_data = []
        for idx, details in similar_questions.items():
            details_data.append(
                {
                    "Original_Row_Index": idx,
                    "Removed_Question": details["excel_question"],
                    "Similar_Master_Question": details["master_question"],
                    "Similarity_Score": f"{details['similarity_score']:.4f}",
                    "Reason": "Too similar to master question",
                }
            )
        return pd.DataFrame(details_data)

    def _create_phase2_details(
        self, irrelevant_questions: Dict[int, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Create Phase 2 details DataFrame."""
        details_data = []
        for idx, details in irrelevant_questions.items():
            details_data.append(
                {
                    "Original_Row_Index": idx,
                    "Removed_Question": details["question"],
                    "Relevance_Score": f"{details['relevance_score']:.4f}",
                    "Threshold": self.syllabus_relevance_threshold,
                    "Reason": "Not relevant to syllabus content",
                }
            )
        return pd.DataFrame(details_data)

    def process_complete_pipeline(
        self,
        excel_path: str,
        master_questions: Union[str, List[str]],
        syllabus_content: str,
        output_path: str = None,
        question_column: str = "Question",
        sheet_name: Union[str, int] = 0,
    ) -> Dict[str, Any]:
        """
        Complete processing pipeline: Excel -> Remove Similar -> Filter by Syllabus -> Save

        Args:
            excel_path (str): Path to input Excel file
            master_questions (Union[str, List[str]]): Master questions (file path or list)
            syllabus_content (str): Syllabus content (file path or text)
            output_path (str): Output file path (optional)
            question_column (str): Name of question column
            sheet_name (Union[str, int]): Sheet to process

        Returns:
            Dict[str, Any]: Complete processing results
        """
        try:
            logger.info("=== Starting Complete Syllabus Check Pipeline ===")
            start_time = datetime.now()

            # Step 1: Load Excel file
            logger.info("Step 1: Loading Excel file...")
            df = self.load_excel_questions(excel_path, question_column, sheet_name)
            original_count = len(df)

            # Step 2: Load master questions
            logger.info("Step 2: Loading master questions...")
            master_questions_list = self.load_master_questions(master_questions)

            # Step 3: Remove questions similar to master questions (Phase 1)
            logger.info("Step 3: Removing questions similar to master questions...")
            df_after_phase1, phase1_stats = self.remove_similar_to_master(
                df, master_questions_list, question_column
            )

            # Step 4: Load syllabus content
            logger.info("Step 4: Loading syllabus content...")
            syllabus_text = self.load_syllabus_content(syllabus_content)

            # Step 5: Filter by syllabus relevance (Phase 2)
            logger.info("Step 5: Filtering questions by syllabus relevance...")
            final_df, phase2_stats = self.filter_by_syllabus_relevance(
                df_after_phase1, syllabus_text, question_column
            )

            # Step 6: Generate output path if not provided
            if not output_path:
                base_name = os.path.splitext(os.path.basename(excel_path))[0]
                output_dir = os.path.dirname(excel_path)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(
                    output_dir, f"{base_name}_syllabus_cleaned_{timestamp}.xlsx"
                )

            # Step 7: Save final results
            logger.info("Step 6: Saving final results...")
            saved_path = self.save_final_results(
                final_df, output_path, phase1_stats, phase2_stats
            )

            # Calculate overall statistics
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            final_count = len(final_df)
            total_removed = original_count - final_count

            results = {
                "success": True,
                "input_file": excel_path,
                "output_file": saved_path,
                "processing_time_seconds": processing_time,
                "overall_statistics": {
                    "original_questions": original_count,
                    "final_questions": final_count,
                    "total_removed": total_removed,
                    "reduction_percentage": (
                        (total_removed / original_count) * 100
                        if original_count > 0
                        else 0
                    ),
                },
                "phase1_statistics": phase1_stats,
                "phase2_statistics": phase2_stats,
                "final_dataframe": final_df,
            }

            logger.info("=== Complete Pipeline Finished Successfully ===")
            logger.info(f"Input file: {excel_path}")
            logger.info(f"Output file: {saved_path}")
            logger.info(f"Original questions: {original_count}")
            logger.info(f"Questions after Phase 1: {phase1_stats['remaining_count']}")
            logger.info(f"Final questions after Phase 2: {final_count}")
            logger.info(f"Total questions removed: {total_removed}")
            logger.info(
                f"Overall reduction: {results['overall_statistics']['reduction_percentage']:.2f}%"
            )
            logger.info(f"Processing time: {processing_time:.2f} seconds")

            return results

        except Exception as e:
            logger.error(f"Error in complete pipeline: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "input_file": excel_path,
            }


# Utility Functions
def create_syllabus_checker(
    similarity_threshold: float = 0.8,
    syllabus_relevance_threshold: float = 0.6,
    model_key: str = "bge-large-en",
) -> SyllabusChecker:
    """
    Factory function to create SyllabusChecker instance.

    Args:
        similarity_threshold (float): Threshold for master question similarity
        syllabus_relevance_threshold (float): Threshold for syllabus relevance
        model_key (str): Embedding model to use

    Returns:
        SyllabusChecker: Initialized checker instance
    """
    return SyllabusChecker(
        model_key=model_key,
        similarity_threshold=similarity_threshold,
        syllabus_relevance_threshold=syllabus_relevance_threshold,
    )


# Example usage function
def demo_syllabus_checker():
    """
    Demonstration of the complete syllabus checking pipeline.
    """
    print("=== Syllabus Checker Demo ===\n")

    # Create sample data
    sample_questions = {
        "ID": list(range(1, 11)),
        "Question": [
            "What is machine learning?",
            "How do neural networks work?",
            "What is the weather today?",
            "Explain supervised learning algorithms",
            "What is your favorite color?",
            "Define artificial intelligence",
            "How to cook pasta?",
            "What are decision trees in ML?",
            "Where is the nearest restaurant?",
            "Explain deep learning concepts",
        ],
        "Category": ["Tech"] * 10,
        "Difficulty": ["Medium"] * 10,
    }

    master_questions = [
        "What is machine learning?",
        "How do artificial neural networks function?",
    ]

    syllabus_content = """
    Machine Learning and Artificial Intelligence Course Syllabus
    
    Unit 1: Introduction to Machine Learning
    - Definition and types of machine learning
    - Supervised, unsupervised, and reinforcement learning
    - Applications of machine learning
    
    Unit 2: Neural Networks and Deep Learning
    - Introduction to neural networks
    - Deep learning architectures
    - Backpropagation algorithm
    
    Unit 3: Machine Learning Algorithms
    - Decision trees and random forests
    - Support vector machines
    - Clustering algorithms
    """

    # Save sample data
    df = pd.DataFrame(sample_questions)
    excel_path = "sample_questions.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"Created sample Excel file: {excel_path}")

    # Create syllabus checker
    checker = create_syllabus_checker(
        similarity_threshold=0.7, syllabus_relevance_threshold=0.5
    )

    # Run complete pipeline
    results = checker.process_complete_pipeline(
        excel_path=excel_path,
        master_questions=master_questions,
        syllabus_content=syllabus_content,
        output_path="cleaned_questions.xlsx",
    )

    if results["success"]:
        stats = results["overall_statistics"]
        print(f"\n=== Pipeline Results ===")
        print(f"Original questions: {stats['original_questions']}")
        print(f"Final questions: {stats['final_questions']}")
        print(f"Total removed: {stats['total_removed']}")
        print(f"Reduction: {stats['reduction_percentage']:.2f}%")
        print(f"Output saved to: {results['output_file']}")
    else:
        print(f"Pipeline failed: {results['error']}")


if __name__ == "__main__":
    demo_syllabus_checker()
