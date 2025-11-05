"""
Question Similarity Checker Application

This module provides functionality to:
1. Load questions from Excel files
2. Compare questions using semantic similarity with embeddings
3. Remove duplicate/similar questions based on master question list
4. Save filtered results back to Excel

Uses the embeddings module for generating semantic embeddings for similarity comparison.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
import json
from datetime import datetime

# Import our embeddings module
from embeddings import EmbeddingGenerator, compare_embeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuestionSimilarityChecker:
    """
    A class to handle question similarity checking and duplicate removal.
    """

    def __init__(
        self,
        model_key: str = "bge-large-en",
        similarity_threshold: float = 0.8,
        models_dir: str = "embeddingmodels",
    ):
        """
        Initialize the Question Similarity Checker.

        Args:
            model_key (str): Embedding model to use for similarity calculation
            similarity_threshold (float): Threshold for considering questions similar (0-1)
            models_dir (str): Directory to store embedding models
        """
        self.similarity_threshold = similarity_threshold
        self.embedding_generator = EmbeddingGenerator(
            model_key=model_key, models_dir=models_dir
        )

        # Cache for embeddings to avoid recomputation
        self.question_embeddings_cache = {}
        self.master_embeddings_cache = {}

        logger.info(f"QuestionSimilarityChecker initialized with model: {model_key}")
        logger.info(f"Similarity threshold: {similarity_threshold}")

    def load_excel_file(
        self,
        excel_path: str,
        question_column: str = "Question",
        sheet_name: Union[str, int] = 0,
    ) -> pd.DataFrame:
        """
        Load Excel file and extract questions.

        Args:
            excel_path (str): Path to the Excel file
            question_column (str): Name of the column containing questions
            sheet_name (Union[str, int]): Sheet name or index to read

        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        try:
            if not os.path.exists(excel_path):
                raise FileNotFoundError(f"Excel file not found: {excel_path}")

            # Read Excel file
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            logger.info(f"Loaded Excel file: {excel_path}")
            logger.info(f"Shape: {df.shape}")

            # Check if question column exists
            if question_column not in df.columns:
                available_columns = list(df.columns)
                raise ValueError(
                    f"Column '{question_column}' not found. Available columns: {available_columns}"
                )

            # Remove rows with empty questions
            initial_count = len(df)
            df = df.dropna(subset=[question_column])
            df = df[df[question_column].astype(str).str.strip() != ""]
            final_count = len(df)

            if initial_count != final_count:
                logger.info(
                    f"Removed {initial_count - final_count} rows with empty questions"
                )

            logger.info(f"Valid questions found: {final_count}")
            return df

        except Exception as e:
            logger.error(f"Error loading Excel file: {str(e)}")
            raise

    def generate_embeddings_for_questions(
        self, questions: List[str], cache_key: str = None
    ) -> np.ndarray:
        """
        Generate embeddings for a list of questions with caching support.

        Args:
            questions (List[str]): List of questions
            cache_key (str): Key for caching embeddings

        Returns:
            np.ndarray: Array of embeddings
        """
        try:
            # Check cache first
            if cache_key and cache_key in self.question_embeddings_cache:
                logger.info(f"Using cached embeddings for {cache_key}")
                return self.question_embeddings_cache[cache_key]

            # Generate embeddings
            logger.info(f"Generating embeddings for {len(questions)} questions...")
            embeddings = self.embedding_generator.generate_embeddings_batch(
                questions, batch_size=32, show_progress=True
            )

            # Cache if key provided
            if cache_key:
                self.question_embeddings_cache[cache_key] = embeddings
                logger.info(f"Cached embeddings with key: {cache_key}")

            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def find_similar_questions(
        self, excel_questions: List[str], master_questions: List[str]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Find similar questions between Excel questions and master questions.

        Args:
            excel_questions (List[str]): Questions from Excel file
            master_questions (List[str]): Master questions to compare against

        Returns:
            Dict[int, Dict[str, Any]]: Dictionary mapping Excel question indices to match info
        """
        try:
            logger.info("Starting similarity comparison...")

            # Generate embeddings for both sets
            excel_embeddings = self.generate_embeddings_for_questions(
                excel_questions, cache_key="excel_questions"
            )
            master_embeddings = self.generate_embeddings_for_questions(
                master_questions, cache_key="master_questions"
            )

            similar_questions = {}
            total_comparisons = len(excel_questions) * len(master_questions)
            logger.info(f"Performing {total_comparisons} similarity comparisons...")

            # Compare each Excel question with all master questions
            for excel_idx, excel_embedding in enumerate(excel_embeddings):
                best_match = None
                best_similarity = -1
                best_master_idx = -1

                for master_idx, master_embedding in enumerate(master_embeddings):
                    similarity = compare_embeddings(excel_embedding, master_embedding)

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = master_questions[master_idx]
                        best_master_idx = master_idx

                # Check if similarity exceeds threshold
                if best_similarity >= self.similarity_threshold:
                    similar_questions[excel_idx] = {
                        "excel_question": excel_questions[excel_idx],
                        "master_question": best_match,
                        "master_index": best_master_idx,
                        "similarity_score": best_similarity,
                    }

                    logger.debug(
                        f"Match found - Excel[{excel_idx}] matches Master[{best_master_idx}] "
                        f"with similarity {best_similarity:.4f}"
                    )

            logger.info(
                f"Found {len(similar_questions)} similar questions above threshold {self.similarity_threshold}"
            )
            return similar_questions

        except Exception as e:
            logger.error(f"Error finding similar questions: {str(e)}")
            raise

    def remove_similar_questions(
        self,
        df: pd.DataFrame,
        master_questions: List[str],
        question_column: str = "Question",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Remove questions from DataFrame that are similar to master questions.

        Args:
            df (pd.DataFrame): DataFrame containing questions
            master_questions (List[str]): List of master questions
            question_column (str): Name of the question column

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Filtered DataFrame and removal statistics
        """
        try:
            if question_column not in df.columns:
                raise ValueError(f"Column '{question_column}' not found in DataFrame")

            # Extract questions
            excel_questions = df[question_column].astype(str).tolist()
            original_count = len(df)

            logger.info(f"Processing {original_count} questions from Excel file")
            logger.info(f"Comparing against {len(master_questions)} master questions")

            # Find similar questions
            similar_questions = self.find_similar_questions(
                excel_questions, master_questions
            )

            # Get indices to remove (questions that are similar to master questions)
            indices_to_remove = list(similar_questions.keys())

            # Remove similar questions
            filtered_df = df.drop(indices_to_remove).reset_index(drop=True)
            removed_count = len(indices_to_remove)
            remaining_count = len(filtered_df)

            # Prepare statistics
            stats = {
                "original_count": original_count,
                "removed_count": removed_count,
                "remaining_count": remaining_count,
                "removal_percentage": (
                    (removed_count / original_count) * 100 if original_count > 0 else 0
                ),
                "similar_questions_details": similar_questions,
                "similarity_threshold": self.similarity_threshold,
                "model_used": self.embedding_generator.model_key,
            }

            logger.info(f"Removal complete:")
            logger.info(f"  Original questions: {original_count}")
            logger.info(f"  Removed questions: {removed_count}")
            logger.info(f"  Remaining questions: {remaining_count}")
            logger.info(f"  Removal percentage: {stats['removal_percentage']:.2f}%")

            return filtered_df, stats

        except Exception as e:
            logger.error(f"Error removing similar questions: {str(e)}")
            raise

    def save_filtered_excel(
        self, df: pd.DataFrame, output_path: str, stats: Dict[str, Any] = None
    ) -> str:
        """
        Save filtered DataFrame to Excel file.

        Args:
            df (pd.DataFrame): Filtered DataFrame to save
            output_path (str): Path for output Excel file
            stats (Dict[str, Any]): Optional statistics to include

        Returns:
            str: Path of saved file
        """
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # Create output filename with timestamp if not specified
            if not output_path.endswith((".xlsx", ".xls")):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"{output_path}_filtered_{timestamp}.xlsx"

            # Save main DataFrame
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Filtered_Questions", index=False)

                # Add statistics sheet if provided
                if stats:
                    stats_df = self._create_stats_dataframe(stats)
                    stats_df.to_excel(
                        writer, sheet_name="Removal_Statistics", index=False
                    )

                    # Add detailed matches sheet
                    if (
                        "similar_questions_details" in stats
                        and stats["similar_questions_details"]
                    ):
                        matches_df = self._create_matches_dataframe(
                            stats["similar_questions_details"]
                        )
                        matches_df.to_excel(
                            writer, sheet_name="Removed_Questions", index=False
                        )

            logger.info(f"Filtered Excel file saved: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error saving filtered Excel file: {str(e)}")
            raise

    def _create_stats_dataframe(self, stats: Dict[str, Any]) -> pd.DataFrame:
        """Create DataFrame with removal statistics."""
        stats_data = [
            ["Original Questions Count", stats.get("original_count", 0)],
            ["Removed Questions Count", stats.get("removed_count", 0)],
            ["Remaining Questions Count", stats.get("remaining_count", 0)],
            ["Removal Percentage", f"{stats.get('removal_percentage', 0):.2f}%"],
            ["Similarity Threshold", stats.get("similarity_threshold", 0)],
            ["Embedding Model Used", stats.get("model_used", "Unknown")],
            ["Processing Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ]

        return pd.DataFrame(stats_data, columns=["Metric", "Value"])

    def _create_matches_dataframe(
        self, matches: Dict[int, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Create DataFrame with detailed match information."""
        matches_data = []
        for excel_idx, match_info in matches.items():
            matches_data.append(
                {
                    "Excel_Row_Index": excel_idx,
                    "Excel_Question": match_info["excel_question"],
                    "Matched_Master_Question": match_info["master_question"],
                    "Master_Question_Index": match_info["master_index"],
                    "Similarity_Score": f"{match_info['similarity_score']:.4f}",
                }
            )

        return pd.DataFrame(matches_data)

    def process_excel_file(
        self,
        excel_path: str,
        master_questions: List[str],
        output_path: str = None,
        question_column: str = "Question",
        sheet_name: Union[str, int] = 0,
    ) -> Dict[str, Any]:
        """
        Complete pipeline to process Excel file and remove similar questions.

        Args:
            excel_path (str): Path to input Excel file
            master_questions (List[str]): List of master questions
            output_path (str): Path for output file (optional)
            question_column (str): Name of question column
            sheet_name (Union[str, int]): Sheet to process

        Returns:
            Dict[str, Any]: Processing results and statistics
        """
        try:
            logger.info("=== Starting Excel Processing Pipeline ===")

            # Load Excel file
            df = self.load_excel_file(excel_path, question_column, sheet_name)

            # Remove similar questions
            filtered_df, stats = self.remove_similar_questions(
                df, master_questions, question_column
            )

            # Generate output path if not provided
            if not output_path:
                base_name = os.path.splitext(os.path.basename(excel_path))[0]
                output_dir = os.path.dirname(excel_path)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(
                    output_dir, f"{base_name}_filtered_{timestamp}.xlsx"
                )

            # Save filtered results
            saved_path = self.save_filtered_excel(filtered_df, output_path, stats)

            # Prepare final results
            results = {
                "success": True,
                "input_file": excel_path,
                "output_file": saved_path,
                "statistics": stats,
                "filtered_dataframe": filtered_df,
            }

            logger.info("=== Excel Processing Complete ===")
            logger.info(f"Input file: {excel_path}")
            logger.info(f"Output file: {saved_path}")
            logger.info(f"Questions processed: {stats['original_count']}")
            logger.info(f"Questions removed: {stats['removed_count']}")
            logger.info(f"Questions remaining: {stats['remaining_count']}")

            return results

        except Exception as e:
            logger.error(f"Error in Excel processing pipeline: {str(e)}")
            return {"success": False, "error": str(e), "input_file": excel_path}


# Utility Functions
def create_similarity_checker(
    model_key: str = "bge-large-en", similarity_threshold: float = 0.8
) -> QuestionSimilarityChecker:
    """
    Factory function to create a QuestionSimilarityChecker instance.

    Args:
        model_key (str): Embedding model to use
        similarity_threshold (float): Similarity threshold (0-1)

    Returns:
        QuestionSimilarityChecker: Initialized checker instance
    """
    return QuestionSimilarityChecker(
        model_key=model_key, similarity_threshold=similarity_threshold
    )


def load_master_questions_from_file(file_path: str) -> List[str]:
    """
    Load master questions from a text file (one question per line).

    Args:
        file_path (str): Path to text file containing master questions

    Returns:
        List[str]: List of master questions
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(questions)} master questions from {file_path}")
        return questions
    except Exception as e:
        logger.error(f"Error loading master questions from file: {str(e)}")
        raise


def load_master_questions_from_excel(
    excel_path: str, question_column: str = "Question", sheet_name: Union[str, int] = 0
) -> List[str]:
    """
    Load master questions from an Excel file.

    Args:
        excel_path (str): Path to Excel file
        question_column (str): Name of question column
        sheet_name (Union[str, int]): Sheet name or index

    Returns:
        List[str]: List of master questions
    """
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        if question_column not in df.columns:
            raise ValueError(f"Column '{question_column}' not found in Excel file")

        questions = df[question_column].dropna().astype(str).tolist()
        questions = [q.strip() for q in questions if q.strip()]
        logger.info(f"Loaded {len(questions)} master questions from Excel file")
        return questions
    except Exception as e:
        logger.error(f"Error loading master questions from Excel: {str(e)}")
        raise


def batch_process_excel_files(
    excel_files: List[str],
    master_questions: List[str],
    output_dir: str = "filtered_results",
    model_key: str = "bge-large-en",
    similarity_threshold: float = 0.8,
    question_column: str = "Question",
) -> Dict[str, Any]:
    """
    Process multiple Excel files in batch.

    Args:
        excel_files (List[str]): List of Excel file paths
        master_questions (List[str]): Master questions list
        output_dir (str): Directory for output files
        model_key (str): Embedding model to use
        similarity_threshold (float): Similarity threshold
        question_column (str): Name of question column

    Returns:
        Dict[str, Any]: Batch processing results
    """
    try:
        # Create similarity checker
        checker = create_similarity_checker(model_key, similarity_threshold)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        results = {
            "total_files": len(excel_files),
            "successful_files": 0,
            "failed_files": 0,
            "file_results": {},
            "overall_stats": {
                "total_original_questions": 0,
                "total_removed_questions": 0,
                "total_remaining_questions": 0,
            },
        }

        for excel_file in excel_files:
            try:
                logger.info(f"Processing file: {excel_file}")

                # Generate output path
                base_name = os.path.splitext(os.path.basename(excel_file))[0]
                output_path = os.path.join(output_dir, f"{base_name}_filtered.xlsx")

                # Process file
                file_result = checker.process_excel_file(
                    excel_path=excel_file,
                    master_questions=master_questions,
                    output_path=output_path,
                    question_column=question_column,
                )

                if file_result["success"]:
                    results["successful_files"] += 1
                    stats = file_result["statistics"]
                    results["overall_stats"]["total_original_questions"] += stats[
                        "original_count"
                    ]
                    results["overall_stats"]["total_removed_questions"] += stats[
                        "removed_count"
                    ]
                    results["overall_stats"]["total_remaining_questions"] += stats[
                        "remaining_count"
                    ]
                else:
                    results["failed_files"] += 1

                results["file_results"][excel_file] = file_result

            except Exception as e:
                logger.error(f"Error processing {excel_file}: {str(e)}")
                results["failed_files"] += 1
                results["file_results"][excel_file] = {
                    "success": False,
                    "error": str(e),
                }

        logger.info(
            f"Batch processing complete: {results['successful_files']}/{len(excel_files)} files processed successfully"
        )
        return results

    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise


# Example usage and demo function
def demo_similarity_checker():
    """
    Demonstration function showing how to use the QuestionSimilarityChecker.
    """
    print("=== Question Similarity Checker Demo ===\n")

    # Sample master questions
    master_questions = [
        "What is machine learning?",
        "How does artificial intelligence work?",
        "What are the benefits of cloud computing?",
        "Explain neural networks in detail",
        "What is data science?",
    ]

    print("Master Questions:")
    for i, q in enumerate(master_questions, 1):
        print(f"  {i}. {q}")
    print()

    # Sample Excel data (simulating Excel file content)
    sample_data = {
        "ID": [1, 2, 3, 4, 5, 6, 7],
        "Question": [
            "What is machine learning and how is it used?",  # Similar to master #1
            "How do neural networks function?",  # Similar to master #4
            "What is the weather like today?",  # Not similar
            "Can you explain AI systems?",  # Similar to master #2
            "What are programming languages?",  # Not similar
            "Benefits of using cloud technology",  # Similar to master #3
            "What is quantum computing?",  # Not similar
        ],
        "Category": ["Tech", "AI", "General", "AI", "Programming", "Cloud", "Tech"],
        "Difficulty": ["Medium", "Hard", "Easy", "Medium", "Easy", "Medium", "Hard"],
    }

    # Create sample Excel file
    df = pd.DataFrame(sample_data)
    sample_excel_path = "sample_questions.xlsx"
    df.to_excel(sample_excel_path, index=False)
    print(f"Created sample Excel file: {sample_excel_path}")

    # Create similarity checker
    print("Creating similarity checker...")
    checker = create_similarity_checker(
        similarity_threshold=0.7
    )  # Lower threshold for demo

    # Process the Excel file
    print("Processing Excel file...")
    results = checker.process_excel_file(
        excel_path=sample_excel_path,
        master_questions=master_questions,
        output_path="filtered_sample_questions.xlsx",
    )

    if results["success"]:
        stats = results["statistics"]
        print("\n=== Processing Results ===")
        print(f"Original questions: {stats['original_count']}")
        print(f"Removed questions: {stats['removed_count']}")
        print(f"Remaining questions: {stats['remaining_count']}")
        print(f"Removal percentage: {stats['removal_percentage']:.2f}%")
        print(f"Output file: {results['output_file']}")

        print("\n=== Removed Questions Details ===")
        for idx, match_info in stats["similar_questions_details"].items():
            print(f"Row {idx}: '{match_info['excel_question']}'")
            print(f"  Matched with: '{match_info['master_question']}'")
            print(f"  Similarity: {match_info['similarity_score']:.4f}")
            print()
    else:
        print(f"Processing failed: {results['error']}")

    print("Demo completed!")


if __name__ == "__main__":
    # Run the demo
    demo_similarity_checker()
