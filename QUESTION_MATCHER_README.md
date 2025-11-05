# Question Matcher - Enhanced Syllabus Checker

This enhanced version of the Syllabus Checker now includes a new **Question Matcher** functionality that allows you to compare master questions with user questions using semantic embeddings.

## 🎯 New Question Matcher Feature

### Overview
The Question Matcher uses the same powerful embedding models to find semantic similarities between master questions and user questions. It supports two input formats for master questions and provides detailed matching results.

### Supported Master Question Formats

**Format 1:** Using `<br>` as separator
```
Why is my animation lagging?<br>SUB_TOPIC_JS_PERFORMANCE
What is a closure in JavaScript?<br>SUB_TOPIC_JS_FUNDAMENTALS
```

**Format 2:** Using tab separator
```
How do I center a div?	SUB_TOPIC_CSS_LAYOUT
How to make responsive design?	SUB_TOPIC_CSS_RESPONSIVE
```

### How to Use

#### Option 1: Web Interface
1. Start the server: `python run_server.py`
2. Open your browser to: `http://127.0.0.1:8000/question-matcher.html`
3. Enter master questions (left side) and user questions (right side)
4. Adjust similarity threshold and model if needed
5. Click "Find Matches" to see results

#### Option 2: API Endpoint
Send a POST request to `/question-match` with JSON payload:

```json
{
    "master_questions": [
        "Why is my animation lagging?<br>SUB_TOPIC_JS_PERFORMANCE",
        "How do I center a div?\tSUB_TOPIC_CSS_LAYOUT"
    ],
    "user_questions": [
        "My CSS animations are slow, what can I do?",
        "How can I center an element horizontally?"
    ],
    "similarity_threshold": 0.8,
    "model_key": "bge-large-en"
}
```

#### Option 3: Python Code
```python
from embeddings import find_question_matches

master_questions = [
    "Why is my animation lagging?<br>SUB_TOPIC_JS_PERFORMANCE",
    "How do I center a div?\tSUB_TOPIC_CSS_LAYOUT"
]

user_questions = [
    "My CSS animations are slow, what can I do?",
    "How can I center an element horizontally?"
]

results = find_question_matches(
    master_questions=master_questions,
    user_questions=user_questions,
    similarity_threshold=0.8,
    model_key="bge-large-en"
)

print(f"Found {results['statistics']['total_matches']} matches")
for match in results['matches']:
    print(f"User: {match['user_question']}")
    print(f"Master: {match['master_question']}")
    print(f"Topic: {match['master_topic']}")
    print(f"Similarity: {match['similarity_score']:.3f}")
    print()
```

### Features

- **Semantic Similarity**: Uses advanced embedding models for accurate semantic matching
- **Multiple Formats**: Supports both `<br>` and tab-separated master question formats
- **Configurable Threshold**: Adjust similarity threshold from 0.0 to 1.0
- **Multiple Models**: Choose from 4 different embedding models
- **Detailed Results**: Get similarity scores, topics, and comprehensive statistics
- **Web Interface**: User-friendly HTML interface for easy testing
- **API Integration**: RESTful API for programmatic access

### Available Embedding Models

1. **bge-large-en** (Default) - Best overall performance
2. **gte-large** - Excellent speed
3. **e5-large-v2** - Fast & reliable  
4. **bge-m3** - Multilingual support

### Output Format

The matcher returns:
- **Matches**: List of matched question pairs with similarity scores
- **Statistics**: Total questions, match count, percentage, etc.
- **Parsed Master Questions**: Structured view of master questions with topics

### Example Output

```
Statistics:
  Total User Questions: 5
  Total Master Questions: 4
  Matches Found: 4
  Match Percentage: 80.0%

Matches:
1. User: "My CSS animations are slow, what can I do?"
   Master: "Why is my animation lagging?"
   Topic: SUB_TOPIC_JS_PERFORMANCE
   Similarity: 0.825

2. User: "How can I center an element horizontally?"
   Master: "How do I center a div?"
   Topic: SUB_TOPIC_CSS_LAYOUT
   Similarity: 0.848
```

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server**:
   ```bash
   python run_server.py
   ```

3. **Open the web interface**:
   Navigate to `http://127.0.0.1:8000/question-matcher.html`

4. **Test the functionality**:
   - Enter some master questions in the left panel
   - Enter user questions in the right panel  
   - Click "Find Matches" to see results

## 📚 Integration with Existing Features

The Question Matcher integrates seamlessly with the existing Syllabus Checker features:

- **Same Embedding Models**: Uses the same high-quality models as the syllabus checker
- **Consistent API**: Follows the same API patterns and error handling
- **Shared Infrastructure**: Utilizes the same model caching and optimization

## 🛠️ Technical Details

- **Embedding Generation**: Uses sentence-transformers for high-quality embeddings
- **Similarity Calculation**: Cosine similarity between embedding vectors
- **Caching**: Intelligent caching to avoid recomputing embeddings
- **Batch Processing**: Efficient batch processing for multiple questions
- **Error Handling**: Comprehensive error handling and logging

This enhancement makes the Syllabus Checker even more powerful for educational content analysis and question matching tasks!