"""
Test the Simple Question Matcher API
"""

import requests
import json

def test_simple_question_matcher():
    """Test the simple question matcher API functionality."""
    
    print("🧠 Testing Simple Question Matcher API")
    print("=" * 50)
    
    # API endpoint
    url = "http://127.0.0.1:8001/match-questions"
    
    # Test data
    master_questions = [
        "What is machine learning?",
        "How does artificial intelligence work?", 
        "What are neural networks?",
        "Explain deep learning concepts",
        "What is natural language processing?",
        "How do you train a model?",
        "What is supervised learning?",
        "What is unsupervised learning?"
    ]
    
    user_questions = [
        "Can you explain AI?",
        "What are the basics of ML?", 
        "How do computers understand language?",
        "Tell me about deep neural networks",
        "How to build machine learning models?",
        "What's the difference between supervised and unsupervised learning?"
    ]
    
    payload = {
        "master_questions": master_questions,
        "user_questions": user_questions,
        "similarity_threshold": 0.7,
        "model_key": "bge-large-en"
    }
    
    print("Master Questions:")
    for i, q in enumerate(master_questions, 1):
        print(f"  {i}. {q}")
    
    print("\nUser Questions:")
    for i, q in enumerate(user_questions, 1):
        print(f"  {i}. {q}")
    
    print(f"\nSimilarity Threshold: {payload['similarity_threshold']}")
    print(f"Model: {payload['model_key']}")
    
    try:
        print("\n🔍 Sending request to API...")
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n" + "=" * 50)
            print("✅ RESULTS")
            print("=" * 50)
            
            print(f"📊 Statistics:")
            print(f"  • Total User Questions: {data['total_user_questions']}")
            print(f"  • Total Master Questions: {data['total_master_questions']}")
            print(f"  • Matches Found: {data['total_matches']}")
            print(f"  • Match Percentage: {data['match_percentage']:.1f}%")
            
            if data['matches']:
                print(f"\n🎯 Matches Found:")
                for i, match in enumerate(data['matches'], 1):
                    print(f"\n  {i}. User Question: \"{match['user_question']}\"")
                    print(f"     ↳ Master Match: \"{match['matched_master_question']}\"")
                    print(f"     ↳ Similarity: {match['similarity_score']:.3f} ({match['similarity_score']*100:.1f}%)")
            else:
                print("\n❌ No matches found above the threshold")
                
            print(f"\n✅ Test completed successfully!")
            return True
            
        else:
            print(f"\n❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error: Could not connect to API server")
        print("Make sure the server is running on http://127.0.0.1:8001")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def test_health_check():
    """Test the health check endpoint."""
    try:
        print("\n🏥 Testing Health Check...")
        response = requests.get("http://127.0.0.1:8001/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Status: {data['status']}")
            if 'model_info' in data:
                model_info = data['model_info']
                print(f"📋 Model: {model_info['model_name']}")
                print(f"📐 Dimension: {model_info['dimension']}")
                print(f"💻 Device: {model_info['device']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting API Tests...")
    
    # Test health check first
    if test_health_check():
        print("\n" + "="*50)
        # Test main functionality
        test_simple_question_matcher()
    else:
        print("\n❌ Health check failed. Is the server running?")