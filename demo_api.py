"""
Simple Demo Script for Syllabus Checker API

This script demonstrates the API functionality with sample data.
Run this after starting the API server to see it in action.
"""

import requests
import pandas as pd
import os
import time
import json

# API Configuration
API_BASE = "http://localhost:8000"


def create_sample_data():
    """Create sample files for testing."""
    print("📝 Creating sample test data...")

    # Sample questions Excel file
    questions_data = {
        "ID": list(range(1, 16)),
        "Question": [
            "What is machine learning?",  # Similar to master
            "How do neural networks work?",  # Similar to master
            "What is the weather today?",  # Not relevant
            "Explain supervised learning algorithms",  # Relevant to syllabus
            "What is your favorite color?",  # Not relevant
            "Define artificial intelligence",  # Relevant to syllabus
            "How to cook pasta?",  # Not relevant
            "What are decision trees in ML?",  # Relevant to syllabus
            "Where is the nearest restaurant?",  # Not relevant
            "Explain deep learning concepts",  # Relevant to syllabus
            "What is unsupervised learning?",  # Relevant to syllabus
            "How does backpropagation work?",  # Relevant to syllabus
            "What time is it?",  # Not relevant
            "Explain convolutional neural networks",  # Relevant to syllabus
            "What is reinforcement learning?",  # Relevant to syllabus
        ],
        "Category": ["Tech"] * 15,
        "Difficulty": ["Medium"] * 15,
    }

    df = pd.DataFrame(questions_data)
    excel_path = "demo_questions.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"   ✅ Created: {excel_path}")

    # Sample syllabus content
    syllabus_content = """
    Machine Learning and Artificial Intelligence Course Syllabus
    
    Unit 1: Introduction to Machine Learning
    - Definition and types of machine learning
    - Supervised, unsupervised, and reinforcement learning
    - Applications of machine learning in various domains
    
    Unit 2: Neural Networks and Deep Learning
    - Introduction to artificial neural networks
    - Deep learning architectures and frameworks
    - Backpropagation algorithm and optimization
    - Convolutional Neural Networks (CNNs)
    - Recurrent Neural Networks (RNNs)
    
    Unit 3: Machine Learning Algorithms
    - Decision trees and random forests
    - Support vector machines
    - Clustering algorithms
    - Classification and regression techniques
    
    Unit 4: Advanced Topics
    - Model evaluation and validation
    - Feature selection and engineering
    - Ensemble methods
    - Transfer learning
    """

    syllabus_path = "demo_syllabus.txt"
    with open(syllabus_path, "w", encoding="utf-8") as f:
        f.write(syllabus_content)
    print(f"   ✅ Created: {syllabus_path}")

    # Master questions
    master_questions = [
        "What is machine learning?",
        "How do artificial neural networks function?",
        "What are the basic concepts of AI?",
    ]

    return excel_path, syllabus_path, syllabus_content, master_questions


def test_health_endpoint():
    """Test the health check endpoint."""
    print("\n🏥 Testing Health Check Endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=30)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Health check successful!")
            print(f"   📊 Overall status: {health_data['status']}")
            print(f"   🧠 LLM health: {health_data['llm_health']['status']}")
            print(
                f"   🔤 Embeddings health: {health_data['embeddings_health']['status']}"
            )
            return True
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Health check error: {str(e)}")
        return False


def test_similarity_check(excel_path, master_questions):
    """Test similarity check endpoint."""
    print("\n🔍 Testing Similarity Check Endpoint...")
    try:
        with open(excel_path, "rb") as f:
            files = {"excel_file": f}
            data = {
                "master_questions": master_questions,
                "question_column": "Question",
                "similarity_threshold": 0.7,
            }

            response = requests.post(
                f"{API_BASE}/similarity-check", files=files, data=data, timeout=60
            )

        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Similarity check successful!")
            print(f"   📄 Output file: {result['output_filename']}")
            print(f"   ⏱️  Processing time: {result['processing_time_seconds']:.2f}s")
            stats = result["processing_stats"]
            print(f"   📊 Original questions: {stats['original_count']}")
            print(f"   🗑️  Removed similar: {stats['removed_count']}")
            print(f"   📋 Remaining: {stats['remaining_count']}")
            return result["output_filename"]
        else:
            print(f"   ❌ Similarity check failed: {response.status_code}")
            print(f"   📝 Response: {response.text}")
            return None

    except Exception as e:
        print(f"   ❌ Similarity check error: {str(e)}")
        return None


def test_syllabus_check_text(excel_path, master_questions, syllabus_content):
    """Test syllabus check with text content."""
    print("\n📄 Testing Syllabus Check (Text Content)...")
    try:
        with open(excel_path, "rb") as f:
            files = {"excel_file": f}
            data = {
                "master_questions": master_questions,
                "syllabus_content": syllabus_content,
                "question_column": "Question",
                "similarity_threshold": 0.7,
                "relevance_threshold": 0.5,
            }

            response = requests.post(
                f"{API_BASE}/syllabus-check-text", files=files, data=data, timeout=90
            )

        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Syllabus check (text) successful!")
            print(f"   📄 Output file: {result['output_filename']}")
            print(f"   ⏱️  Processing time: {result['processing_time_seconds']:.2f}s")
            stats = result["processing_stats"]
            print(f"   📊 Original questions: {stats['original_questions']}")
            print(f"   📋 Final questions: {stats['final_questions']}")
            print(f"   🗑️  Total removed: {stats['total_removed']}")
            print(f"   📉 Reduction: {stats['reduction_percentage']:.1f}%")
            return result["output_filename"]
        else:
            print(f"   ❌ Syllabus check (text) failed: {response.status_code}")
            print(f"   📝 Response: {response.text}")
            return None

    except Exception as e:
        print(f"   ❌ Syllabus check (text) error: {str(e)}")
        return None


def check_server_running():
    """Check if the API server is running."""
    try:
        response = requests.get(f"{API_BASE}/", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    """Run the demo."""
    print("🚀 Syllabus Checker API - Live Demo")
    print("=" * 50)

    # Check if server is running
    print("🔍 Checking if API server is running...")
    if not check_server_running():
        print("❌ API server is not running!")
        print("\n💡 Please start the server first:")
        print("   python start.py")
        print("   or")
        print("   uvicorn api:app --host 0.0.0.0 --port 8000")
        print("\nThen run this demo again.")
        return

    print("✅ API server is running!")

    # Create sample data
    excel_path, syllabus_path, syllabus_content, master_questions = create_sample_data()

    try:
        # Test endpoints
        if not test_health_endpoint():
            print("❌ Health check failed. Cannot proceed with demo.")
            return

        # Test similarity check
        similarity_output = test_similarity_check(excel_path, master_questions)

        # Test full syllabus check
        full_output = test_syllabus_check_text(
            excel_path, master_questions, syllabus_content
        )

        # Show results
        print("\n🎉 Demo Completed Successfully!")
        print(f"📁 Files created in data_process folder:")
        if similarity_output:
            print(f"   • {similarity_output}")
        if full_output:
            print(f"   • {full_output}")

        print(f"\n💾 You can download files using:")
        print(f"   GET {API_BASE}/download/{{filename}}")

        print(f"\n🧹 Clean up files using:")
        print(f"   POST {API_BASE}/cleanup")

    finally:
        # Cleanup sample files
        for file_path in [excel_path, syllabus_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🧹 Cleaned up: {file_path}")


if __name__ == "__main__":
    main()
