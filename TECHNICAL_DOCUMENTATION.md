# 📚 Syllabus Checker - Complete Technical Documentation

## 🏗️ **Project Architecture Overview**

The Syllabus Checker is a comprehensive AI-powered question bank processing system built with FastAPI, featuring two-phase intelligent filtering:

1. **Phase 1**: Remove questions similar to master questions (duplicate removal)
2. **Phase 2**: Remove questions not relevant to syllabus content (relevance filtering)

### **🎯 System Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    Syllabus Checker System                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   FastAPI   │    │   Web UI    │    │   CLI Tool  │     │
│  │  REST API   │    │ (Optional)  │    │ (Scripts)   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                │                            │
├─────────────────────────────────┼────────────────────────────┤
│                Core Processing Layer                        │
│                                │                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Syllabus    │ ── │ Similarity  │ ── │ Embeddings  │     │
│  │ Checker     │    │ Checker     │    │ Generator   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                │                            │
├─────────────────────────────────┼────────────────────────────┤
│                    AI/ML Layer                              │
│                                │                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   LangChain │    │ Sentence    │    │    LLM      │     │
│  │ Integration │    │Transformers │    │ Providers   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                │                            │
├─────────────────────────────────┼────────────────────────────┤
│                 Storage Layer                               │
│                                │                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Excel I/O  │    │ File System │    │ Model Cache │     │
│  │   Processing│    │ Management  │    │   Storage   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📁 **Project Structure**

```
syllabuscheck/
├── 🔧 Core Modules
│   ├── main.py                    # FastAPI application entry point
│   ├── embeddings.py              # Text embedding generation
│   ├── similarity.py              # Question similarity checking
│   ├── syllabus_check.py          # Main processing pipeline
│   └── llm.py                     # LLM provider abstraction
│
├── ⚙️ Configuration
│   ├── config.py                  # System configuration
│   ├── properties.py              # API keys and secrets
│   └── requirements.txt           # Python dependencies
│
├── 🚀 Utility Scripts
│   ├── start.py                   # Server startup script
│   ├── status_check.py            # System health check
│   ├── demo_api.py                # API demonstration
│   └── example_usage.py           # Usage examples
│
├── 🤖 AI Models
│   └── embeddingmodels/           # Cached embedding models
│       └── bge-large-en/          # BGE model files
│
├── 📁 Data Processing
│   └── data_process/              # Temporary file storage
│
└── 📚 Documentation
    ├── README.md                  # Main documentation
    └── *.md                       # Additional docs
```

## 🧩 **Core Module Documentation**

### **📊 embeddings.py - Text Embedding Engine**

**Purpose**: Generates high-quality semantic embeddings for text similarity comparison.

#### **Key Classes:**

##### `EmbeddingGenerator`
```python
class EmbeddingGenerator:
    """
    Handles text embedding generation using sentence transformer models.
    Supports multiple models with automatic caching.
    """
    
    def __init__(self, model_key="bge-large-en", models_dir="embeddingmodels"):
        """
        Args:
            model_key: Model identifier from AVAILABLE_MODELS
            models_dir: Directory for model storage and caching
        """
```

#### **Supported Models:**
| Model Key | Model Name | Dimension | Best For |
|-----------|------------|-----------|----------|
| `bge-large-en` | BAAI/bge-large-en-v1.5 | 1024 | Overall performance |
| `gte-large` | thenlper/gte-large | 1024 | Speed optimization |
| `e5-large-v2` | intfloat/e5-large-v2 | 1024 | Reliability |
| `bge-m3` | BAAI/bge-m3 | 1024 | Multilingual support |

#### **Key Methods:**
- `generate_embeddings(texts)` - Generate embeddings for text list
- `generate_single_embedding(text)` - Generate embedding for single text
- `compare_embeddings(emb1, emb2)` - Calculate cosine similarity
- `get_model_info()` - Get model specifications

---

### **🔍 similarity.py - Question Similarity Checker**

**Purpose**: Identifies and removes duplicate/similar questions based on semantic similarity.

#### **Key Classes:**

##### `QuestionSimilarityChecker`
```python
class QuestionSimilarityChecker:
    """
    Handles question similarity checking and duplicate removal using embeddings.
    """
    
    def __init__(self, model_key="bge-large-en", similarity_threshold=0.8):
        """
        Args:
            model_key: Embedding model to use
            similarity_threshold: Similarity threshold (0-1)
        """
```

#### **Core Processing Pipeline:**
1. **Load Questions**: Import from Excel files
2. **Generate Embeddings**: Create semantic representations
3. **Compare Similarity**: Calculate cosine similarity scores
4. **Filter Duplicates**: Remove questions above threshold
5. **Export Results**: Save cleaned data with statistics

#### **Key Methods:**
- `process_excel_file()` - Complete pipeline for Excel processing
- `find_similar_questions()` - Identify similar question pairs
- `remove_similar_questions()` - Filter out duplicates
- `generate_report()` - Create processing statistics

---

### **📋 syllabus_check.py - Complete Processing Pipeline**

**Purpose**: Main orchestrator for two-phase question processing with syllabus relevance filtering.

#### **Key Classes:**

##### `SyllabusChecker`
```python
class SyllabusChecker:
    """
    Main class for comprehensive question bank processing and syllabus-based filtering.
    Combines similarity checking with relevance filtering.
    """
    
    def __init__(self, similarity_threshold=0.8, syllabus_relevance_threshold=0.6):
        """
        Args:
            similarity_threshold: Threshold for similarity detection
            syllabus_relevance_threshold: Threshold for syllabus relevance
        """
```

#### **Two-Phase Processing:**

##### **Phase 1: Similarity Filtering**
- Remove questions similar to master questions
- Use semantic embeddings for comparison
- Configurable similarity threshold

##### **Phase 2: Relevance Filtering**
- Remove questions not relevant to syllabus
- Use LLM-powered relevance scoring
- Configurable relevance threshold

#### **Key Methods:**
- `process_complete_pipeline()` - Full two-phase processing
- `phase1_similarity_check()` - Duplicate removal
- `phase2_relevance_check()` - Syllabus filtering
- `generate_comprehensive_report()` - Complete statistics

---

### **🚀 main.py - FastAPI Application**

**Purpose**: REST API server providing web interface for all processing capabilities.

#### **API Endpoints:**

##### **Health & Status**
- `GET /health` - System health check
- `GET /files` - List processed files
- `POST /cleanup` - Manual file cleanup

##### **Processing Endpoints (Direct Download)**
- `POST /similarity-check` - Phase 1 only (similarity filtering)
- `POST /syllabus-check-text` - Complete pipeline with text syllabus
- `POST /syllabus-check-file` - Complete pipeline with file syllabus

#### **Key Features:**
- **Direct File Download**: Returns Excel files directly
- **Background Cleanup**: Automatic file management
- **Processing Statistics**: Headers with processing info
- **CORS Support**: Cross-origin request handling
- **Error Handling**: Comprehensive error responses

---

### **🤖 llm.py - LLM Provider Abstraction**

**Purpose**: Unified interface for multiple LLM providers.

#### **Supported Providers:**
- **Groq**: Fast inference with various models
- **OpenAI**: GPT models via OpenAI API
- **Google**: Gemini models via Google AI

#### **Key Features:**
- Provider switching via configuration
- Unified interface across providers
- API key management
- Model selection per provider

---

## ⚙️ **Configuration System**

### **config.py - System Settings**
```python
LLM_PROVIDER = "groq"              # LLM provider selection
Groq_model = "openai/gpt-oss-20b"  # Groq model specification
OpenAI_model = "gpt-3.5-turbo"     # OpenAI model specification
Google_model = ""                   # Google model specification
```

### **properties.py - API Credentials**
```python
GROQ_API_KEY = "your_groq_key"      # Groq API key
OPEN_AI_API_KEY = "your_openai_key" # OpenAI API key
GOOGLE_API_KEY = "your_google_key"  # Google API key
```

## 🔄 **Data Flow Architecture**

```
Input Excel File
        │
        ▼
┌─────────────────┐
│  File Upload    │
│  & Validation   │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Phase 1:        │
│ Similarity      │
│ Detection       │
│                 │
│ 1. Load Excel   │
│ 2. Extract Q's  │
│ 3. Generate     │
│    Embeddings   │
│ 4. Compare      │
│    Similarity   │
│ 5. Filter       │
│    Duplicates   │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Phase 2:        │
│ Relevance       │
│ Filtering       │
│                 │
│ 1. Load         │
│    Syllabus     │
│ 2. Generate     │
│    Embeddings   │
│ 3. Calculate    │
│    Relevance    │
│ 4. Filter       │
│    Questions    │
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ Report          │
│ Generation      │
│                 │
│ 1. Statistics   │
│ 2. Removed Q's  │
│ 3. Final Data   │
│ 4. Excel Export │
└─────────────────┘
        │
        ▼
   Output Excel File
```

## 🎛️ **Processing Parameters**

### **Similarity Threshold (0.0 - 1.0)**
- **0.9+**: Very strict (only nearly identical questions)
- **0.8**: Recommended default (good balance)
- **0.7**: Moderate (catches more variations)
- **0.6-**: Liberal (may catch unrelated questions)

### **Relevance Threshold (0.0 - 1.0)**
- **0.8+**: Very strict (only highly relevant)
- **0.6**: Recommended default (good balance)
- **0.4**: Moderate (includes tangentially related)
- **0.3-**: Liberal (may include irrelevant content)

## 📊 **Output Data Structure**

### **Excel Output Sheets:**
1. **Cleaned_Questions**: Final filtered questions
2. **Removed_Similar**: Questions removed in Phase 1
3. **Removed_Irrelevant**: Questions removed in Phase 2
4. **Processing_Stats**: Detailed statistics
5. **Configuration**: Processing parameters used

### **Statistics Included:**
- Original question count
- Phase 1 removals (similarity)
- Phase 2 removals (relevance)
- Final question count
- Reduction percentage
- Processing time
- Configuration used

## 🛠️ **Utility Scripts**

### **start.py - Server Management**
- Server startup with proper configuration
- Dependency installation
- Health checking
- Error handling

### **status_check.py - System Validation**
- Module import validation
- API endpoint testing
- File system checking
- Comprehensive status reporting

### **demo_api.py - API Testing**
- Live API demonstration
- Sample data generation
- Endpoint testing
- Response validation

### **example_usage.py - Usage Examples**
- Python library usage examples
- Step-by-step processing demos
- Configuration examples
- Best practices

## 🔒 **Security Considerations**

### **API Key Management**
- Store keys in `properties.py`
- Use environment variables in production
- Never commit keys to version control
- Rotate keys regularly

### **File Processing Security**
- Temporary file cleanup
- Input validation
- Safe file operations
- Resource limits

### **API Security**
- CORS configuration
- Input sanitization
- Error message sanitization
- File upload limits

## 🚀 **Performance Optimization**

### **Embedding Caching**
- Local model storage
- Embedding result caching
- Batch processing optimization
- Memory management

### **Processing Efficiency**
- Vectorized operations
- Efficient similarity calculations
- Background processing
- Resource monitoring

### **API Performance**
- Async request handling
- Background task processing
- File streaming
- Response compression

This documentation provides a complete technical overview of the Syllabus Checker system, covering architecture, modules, configuration, and usage patterns.