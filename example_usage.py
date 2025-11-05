"""
Example Usage Script for Syllabus Checker

This script demonstrates how to use the SyllabusChecker class for
processing question banks with two-phase filtering:
1. Remove questions similar to master questions
2. Remove questions not relevant to syllabus content
"""

from syllabus_check import create_syllabus_checker
import pandas as pd


def example_usage():
    """
    Example of how to use the syllabus checker in practice.
    """
    print("=== Syllabus Checker Usage Example ===\n")

    # Step 1: Create the checker instance with desired thresholds
    checker = create_syllabus_checker(
        similarity_threshold=0.8,  # 80% similarity to consider questions as duplicates
        syllabus_relevance_threshold=0.6,  # 60% relevance to syllabus to keep questions
        model_key="bge-large-en",  # Embedding model to use
    )

    # Step 2: Define your inputs
    excel_file_path = "your_questions.xlsx"  # Path to your Excel file with questions
    master_questions_file = "master_questions.txt"  # Path to master questions file
    # OR you can provide master questions as a list:
    # master_questions_list = ["What is AI?", "How does ML work?", ...]

    syllabus_file_path = "course_syllabus.txt"  # Path to syllabus text file
    # OR you can provide syllabus content directly:
    # syllabus_content = "Course covers AI, ML, Deep Learning..."

    output_file_path = "cleaned_questions.xlsx"  # Where to save the final results

    # Step 3: Run the complete pipeline
    try:
        results = checker.process_complete_pipeline(
            excel_path=excel_file_path,
            master_questions=master_questions_file,  # Can be file path or list
            syllabus_content=syllabus_file_path,  # Can be file path or text
            output_path=output_file_path,
            question_column="Question",  # Name of column with questions
            sheet_name=0,  # Excel sheet to process
        )

        if results["success"]:
            # Print summary
            print("✅ Processing completed successfully!")
            print(f"📊 Results Summary:")
            print(
                f"   Original questions: {results['overall_statistics']['original_questions']}"
            )
            print(
                f"   Final questions: {results['overall_statistics']['final_questions']}"
            )
            print(f"   Total removed: {results['overall_statistics']['total_removed']}")
            print(
                f"   Reduction percentage: {results['overall_statistics']['reduction_percentage']:.2f}%"
            )
            print(f"💾 Output saved to: {results['output_file']}")

            # Phase-wise breakdown
            print(f"\n📈 Phase-wise Details:")
            phase1 = results["phase1_statistics"]
            phase2 = results["phase2_statistics"]
            print(f"   Phase 1 (Similar to master): {phase1['removed_count']} removed")
            print(f"   Phase 2 (Not in syllabus): {phase2['removed_count']} removed")

        else:
            print(f"❌ Processing failed: {results['error']}")

    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")


def step_by_step_usage():
    """
    Example showing step-by-step usage for more control.
    """
    print("\n=== Step-by-Step Usage Example ===\n")

    # Initialize checker
    checker = create_syllabus_checker()

    try:
        # Step 1: Load Excel file
        print("Step 1: Loading Excel file...")
        df = checker.load_excel_questions("your_questions.xlsx", "Question")
        print(f"Loaded {len(df)} questions")

        # Step 2: Load master questions
        print("Step 2: Loading master questions...")
        master_questions = checker.load_master_questions("master_questions.txt")
        print(f"Loaded {len(master_questions)} master questions")

        # Step 3: Phase 1 - Remove similar questions
        print("Step 3: Removing questions similar to master questions...")
        df_phase1, phase1_stats = checker.remove_similar_to_master(df, master_questions)
        print(f"Phase 1: {phase1_stats['removed_count']} questions removed")

        # Step 4: Load syllabus
        print("Step 4: Loading syllabus content...")
        syllabus_content = checker.load_syllabus_content("course_syllabus.txt")
        print(f"Loaded syllabus ({len(syllabus_content)} characters)")

        # Step 5: Phase 2 - Filter by syllabus relevance
        print("Step 5: Filtering by syllabus relevance...")
        final_df, phase2_stats = checker.filter_by_syllabus_relevance(
            df_phase1, syllabus_content
        )
        print(f"Phase 2: {phase2_stats['removed_count']} questions removed")

        # Step 6: Save results
        print("Step 6: Saving final results...")
        output_path = checker.save_final_results(
            final_df, "step_by_step_output.xlsx", phase1_stats, phase2_stats
        )
        print(f"Results saved to: {output_path}")

        print(
            f"\n✅ Final Result: {len(final_df)} questions remain out of {len(df)} original"
        )

    except Exception as e:
        print(f"❌ Error in step-by-step processing: {str(e)}")


def format_requirements():
    """
    Information about required file formats.
    """
    print("\n=== File Format Requirements ===\n")

    print("📋 Excel File Format:")
    print("   • Must have a column with questions (default name: 'Question')")
    print("   • Can have additional columns (they will be preserved)")
    print("   • Supported formats: .xlsx, .xls")
    print("   • Example:")
    print("     | ID | Question | Category | Difficulty |")
    print("     |----|----------|----------|------------|")
    print("     | 1  | What is AI? | Tech | Easy |")
    print("     | 2  | How does ML work? | Tech | Medium |")

    print("\n📝 Master Questions File Format:")
    print("   • Text file (.txt) with one question per line")
    print("   • OR Excel file with question column")
    print("   • OR Python list of strings")
    print("   • Example (text file):")
    print("     What is artificial intelligence?")
    print("     How do machine learning algorithms work?")
    print("     Explain neural network architecture")

    print("\n📖 Syllabus Content Format:")
    print("   • Text file (.txt) with course syllabus content")
    print("   • OR direct text string")
    print("   • Should contain topics, concepts, keywords related to the course")
    print("   • Example:")
    print("     Machine Learning Course Syllabus")
    print("     Unit 1: Introduction to AI and ML")
    print("     Unit 2: Supervised Learning Algorithms")
    print("     Unit 3: Neural Networks and Deep Learning")


def main():
    """
    Main function demonstrating different usage patterns.
    """
    print("🚀 Syllabus Checker - Usage Examples\n")

    # Show format requirements
    format_requirements()

    # Show complete pipeline usage
    print("\n" + "=" * 50)
    example_usage()

    # Show step-by-step usage
    print("\n" + "=" * 50)
    step_by_step_usage()

    print("\n" + "=" * 50)
    print("📚 Additional Notes:")
    print("• Higher similarity_threshold = more strict duplicate removal")
    print("• Higher syllabus_relevance_threshold = more strict relevance filtering")
    print("• Processing time depends on number of questions and content length")
    print("• Results include detailed statistics and removed questions tracking")
    print("• Output Excel file contains multiple sheets with analysis details")


if __name__ == "__main__":
    main()
