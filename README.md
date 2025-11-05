# Syllabus Checker - Question Bank Processing Tool

A comprehensive Python tool and FastAPI service for processing and cleaning question banks using semantic similarity and syllabus-based filtering.

## 🚀 Available Interfaces

### 1. **FastAPI REST Service** (Recommended)
Complete REST API with 4 endpoints for web-based processing:
- Health checking for LLM and embeddings
- Similarity-only checking
- Complete syllabus checking (text input)
- Complete syllabus checking (file input)

### 2. **Python Library**
Direct Python integration for custom applications and batch processing.

## Features

### Two-Phase Processing Pipeline:
1. **Phase 1**: Remove questions similar to master questions (duplicate removal)
2. **Phase 2**: Remove questions not relevant to syllabus content (relevance filtering)

### Key Capabilities:
- ✅ **REST API Service** with automatic file management
- ✅ Load questions from Excel files (.xlsx, .xls)
- ✅ Compare questions using semantic embeddings (BGE-Large-EN model)
- ✅ Remove duplicate/similar questions based on master question list
- ✅ Filter questions based on syllabus content relevance
- ✅ Generate comprehensive Excel reports with statistics
- ✅ Support multiple input formats (Excel, text files, direct text)
- ✅ Configurable similarity and relevance thresholds
- ✅ Detailed processing statistics and removed questions tracking
- ✅ **Unique file naming** with 10-digit identifiers
- ✅ **Automatic cleanup** of processed files
- ✅ **Automatic model downloading** - no manual model setup required
- ✅ **Git-friendly** - embedding models excluded from repository

## Installation

```bash
# Install all dependencies
pip install -r requirements.txt
```

## 🤖 Embedding Models

### Automatic Model Management
The system automatically downloads and manages embedding models:

- **First Run**: Models are downloaded automatically when the server starts
- **No Manual Setup**: Models download seamlessly in the background
- **Offline Ready**: Once downloaded, models work offline
- **Git Safe**: Models are excluded from git repository (added to .gitignore)

### Available Models
- `bge-large-en` (default) - Best overall performance
- `gte-large` - Excellent speed  
- `e5-large-v2` - Fast & reliable
- `bge-m3` - Multilingual support

### Manual Model Management (Optional)
```bash
# List available models
python download_models.py --list

# Check downloaded models
python download_models.py --check

# Download specific model
python download_models.py --model bge-large-en

# Download all models (for offline environments)
python download_models.py --all
```

## 🚀 Quick Start - FastAPI Service (Recommended)

### Start the API Server
```bash
# Easy start
python start.py

# Or manually
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Note**: On first startup, the default embedding model will be downloaded automatically. This may take a few minutes depending on your internet connection.

### Access Points
- **API Base**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### API Usage Examples

**1. Similarity Check Only:**
```bash
curl -X POST "http://localhost:8000/similarity-check" \
  -F "excel_file=@questions.xlsx" \
  -F "master_questions=What is AI?" \
  -F "master_questions=How does ML work?" \
  -F "similarity_threshold=0.8"
```

**2. Complete Syllabus Check (Text Content):**
```bash
curl -X POST "http://localhost:8000/syllabus-check-text" \
  -F "excel_file=@questions.xlsx" \
  -F "master_questions=What is AI?" \
  -F "syllabus_content=Machine Learning course covers AI, neural networks..." \
  -F "similarity_threshold=0.8" \
  -F "relevance_threshold=0.6"
```

**3. Complete Syllabus Check (File Content):**
```bash
curl -X POST "http://localhost:8000/syllabus-check-file" \
  -F "excel_file=@questions.xlsx" \
  -F "syllabus_file=@syllabus.txt" \
  -F "master_questions=What is AI?" \
  -F "similarity_threshold=0.8" \
  -F "relevance_threshold=0.6"
```

## 📋 Python Library Usage

```python
from syllabus_check import create_syllabus_checker

# Create checker with custom thresholds
checker = create_syllabus_checker(
    similarity_threshold=0.8,          # 80% similarity for duplicate detection
    syllabus_relevance_threshold=0.6   # 60% relevance to syllabus content
)

# Process complete pipeline
results = checker.process_complete_pipeline(
    excel_path="questions.xlsx",           # Your Excel file with questions
    master_questions="master_questions.txt",  # Master questions file
    syllabus_content="syllabus.txt",       # Course syllabus file
    output_path="cleaned_questions.xlsx"   # Output file
)

if results["success"]:
    print(f"✅ Processing complete!")
    print(f"Original: {results['overall_statistics']['original_questions']} questions")
    print(f"Final: {results['overall_statistics']['final_questions']} questions")
    print(f"Removed: {results['overall_statistics']['total_removed']} questions")
else:
    print(f"❌ Error: {results['error']}")
```

## File Format Requirements

### Excel File (Questions)
- Must contain a column with questions (default: "Question")
- Can include additional columns (preserved in output)
- Supported formats: .xlsx, .xls

Example:
| ID | Question | Category | Difficulty |
|----|----------|----------|------------|
| 1 | What is AI? | Tech | Easy |
| 2 | How does ML work? | Tech | Medium |

### Master Questions
Can be provided as:
- **Text file**: One question per line (.txt)
- **Excel file**: With questions column
- **Python list**: List of strings

Example (text file):
```
What is artificial intelligence?
How do machine learning algorithms work?
Explain neural network architecture
```

### Syllabus Content
Can be provided as:
- **Text file**: Course syllabus content (.txt)
- **Direct text**: String with syllabus content

Example:
```
Machine Learning Course Syllabus

Unit 1: Introduction to AI and ML
- Definition and applications of AI
- Types of machine learning

Unit 2: Supervised Learning
- Classification algorithms
- Regression techniques

Unit 3: Neural Networks
- Perceptron and multilayer networks
- Deep learning fundamentals
```

## Advanced Usage

### Step-by-Step Processing
```python
# Initialize checker
checker = create_syllabus_checker()

# Step 1: Load Excel file
df = checker.load_excel_questions("questions.xlsx")

# Step 2: Load master questions
master_questions = checker.load_master_questions("master_questions.txt")

# Step 3: Phase 1 - Remove similar questions
df_phase1, phase1_stats = checker.remove_similar_to_master(df, master_questions)

# Step 4: Load syllabus
syllabus_content = checker.load_syllabus_content("syllabus.txt")

# Step 5: Phase 2 - Filter by relevance
final_df, phase2_stats = checker.filter_by_syllabus_relevance(df_phase1, syllabus_content)

# Step 6: Save results
output_path = checker.save_final_results(final_df, "output.xlsx", phase1_stats, phase2_stats)
```

### Custom Configuration
```python
checker = create_syllabus_checker(
    similarity_threshold=0.85,         # More strict duplicate detection
    syllabus_relevance_threshold=0.7,  # More strict relevance filtering
    model_key="bge-large-en"           # Embedding model to use
)
```

## Output

The tool generates an Excel file with multiple sheets:

1. **Final_Cleaned_Questions**: Final filtered questions
2. **Processing_Summary**: Overall statistics and metrics
3. **Phase1_Removed_Similar**: Questions removed in Phase 1 (duplicates)
4. **Phase2_Removed_Irrelevant**: Questions removed in Phase 2 (irrelevant)

## Processing Statistics

The tool provides comprehensive statistics including:
- Original question count
- Questions removed in each phase
- Final question count
- Removal percentages
- Similarity and relevance scores
- Processing time
- Detailed information about removed questions

## Configuration Options

### Similarity Threshold
- Range: 0.0 - 1.0
- Higher values = more strict duplicate detection
- Recommended: 0.7 - 0.9

### Syllabus Relevance Threshold  
- Range: 0.0 - 1.0
- Higher values = more strict relevance filtering
- Recommended: 0.5 - 0.8

### Embedding Models
- Default: "bge-large-en"
- Requires model files in `embeddingmodels/` directory

## Example Output

```
=== Processing Results ===
Original questions: 100
Questions after Phase 1: 75 (removed 25 similar to master questions)
Final questions after Phase 2: 60 (removed 15 irrelevant to syllabus)
Total reduction: 40%
Processing time: 45.2 seconds
Output saved to: cleaned_questions_20240321_143022.xlsx
```

## Error Handling

The tool includes comprehensive error handling for:
- Missing files or invalid paths
- Incorrect file formats
- Missing required columns
- Empty question lists
- Embedding generation failures

## Performance Notes

- Processing time depends on number of questions and content length
- Embedding generation is cached to improve performance
- Batch processing is used for efficiency
- Progress indicators show processing status

## Troubleshooting

### Common Issues:
1. **Import Error**: Ensure all dependencies are installed
2. **File Not Found**: Check file paths are correct
3. **Column Not Found**: Verify question column name
4. **Out of Memory**: Reduce batch size for large files
5. **Slow Processing**: Consider using smaller similarity thresholds

For more examples, see `example_usage.py`.

## License

This project is for educational and research purposes.