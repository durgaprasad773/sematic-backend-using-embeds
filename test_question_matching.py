"""
Simple test script for the new question matching functionality
"""

from embeddings import find_question_matches

def test_question_matching():
    print("Testing Question Matching Functionality")
    print("=" * 50)
    
    # Sample master questions
    master_questions = [
        "Why is my animation lagging?<br>SUB_TOPIC_JS_PERFORMANCE",
        "How do I center a div?\tSUB_TOPIC_CSS_LAYOUT",
        "What is a closure in JavaScript?<br>SUB_TOPIC_JS_FUNDAMENTALS",
        "How to make responsive design?\tSUB_TOPIC_CSS_RESPONSIVE"
    ]
    
    # Sample user questions
    user_questions = [
        "My CSS animations are slow, what can I do?",
        "How can I center an element horizontally?",
        "What are JavaScript closures?",
        "How to create a mobile-friendly layout?",
        "What is machine learning?"
    ]
    
    print("Master Questions:")
    for i, q in enumerate(master_questions, 1):
        print(f"  {i}. {q}")
    
    print("\nUser Questions:")
    for i, q in enumerate(user_questions, 1):
        print(f"  {i}. {q}")
    
    print("\nFinding matches...")
    
    try:
        results = find_question_matches(
            master_questions=master_questions,
            user_questions=user_questions,
            similarity_threshold=0.7,
            model_key="bge-large-en"
        )
        
        print("\n" + "=" * 50)
        print("RESULTS")
        print("=" * 50)
        
        stats = results['statistics']
        print(f"Total User Questions: {stats['total_user_questions']}")
        print(f"Total Master Questions: {stats['total_master_questions']}")
        print(f"Matches Found: {stats['total_matches']}")
        print(f"Match Percentage: {stats['match_percentage']:.1f}%")
        print(f"Similarity Threshold: {stats['similarity_threshold']}")
        print(f"Model Used: {stats['model_used']}")
        
        if results['matches']:
            print("\nMatches:")
            for i, match in enumerate(results['matches'], 1):
                print(f"\n{i}. User: \"{match['user_question']}\"")
                print(f"   Master: \"{match['master_question']}\"")
                print(f"   Topic: {match['master_topic']}")
                print(f"   Similarity: {match['similarity_score']:.3f}")
        else:
            print("\nNo matches found above the threshold.")
            
        print("\nTest completed successfully! ✅")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_question_matching()