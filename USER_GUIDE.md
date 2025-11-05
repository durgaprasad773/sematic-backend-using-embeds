# 👨‍💻 Syllabus Checker - Complete User Guide

## 🎯 **What is Syllabus Checker?**

Syllabus Checker is an intelligent AI-powered tool that helps educators and content creators clean and optimize their question banks. It uses advanced machine learning to:

1. **Remove Duplicate Questions** - Identifies and removes questions that are too similar to your master questions
2. **Filter by Relevance** - Removes questions that don't align with your syllabus content
3. **Generate Reports** - Provides detailed statistics about the cleaning process

### **🔥 Key Benefits**
- ✅ **Save Time**: Automated processing instead of manual review
- ✅ **Improve Quality**: Remove irrelevant and duplicate questions
- ✅ **Maintain Standards**: Ensure questions align with curriculum
- ✅ **Get Insights**: Detailed statistics and processing reports
- ✅ **Easy to Use**: Simple web API or Python library interface

---

## 🚀 **Quick Start Guide**

### **📋 Prerequisites**
- Python 3.12+ installed
- Basic understanding of Excel files
- Internet connection (for first-time model download)

### **⚡ 30-Second Setup**

1. **Download and Install**:
```bash
# Clone or download the project
cd syllabuscheck

# Install dependencies
pip install -r requirements.txt
```

2. **Start the API Server**:
```bash
python start.py
```

3. **Open Your Browser**:
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### **🎉 You're Ready!**

---

## 📊 **Input Requirements**

### **Excel File Format**
Your Excel file should have questions in one of these formats:

#### **Option 1: Simple Format**
| Question |
|----------|
| What is machine learning? |
| How do neural networks work? |
| Explain supervised learning |

#### **Option 2: With ID Column**
| ID | Question |
|----|----------|
| 1 | What is machine learning? |
| 2 | How do neural networks work? |
| 3 | Explain supervised learning |

#### **Option 3: Any Column Name**
- The system will automatically detect columns containing questions
- Common column names: "Question", "Questions", "Q", "Query", etc.

### **Master Questions**
Provide a list of questions you want to compare against:
```
Examples:
- "What is artificial intelligence?"
- "How does machine learning work?"
- "Explain neural network architecture"
```

### **Syllabus Content**
Provide syllabus content in one of these ways:
- **Text**: Paste syllabus content directly
- **File**: Upload a .txt file with syllabus content

---

## 🎮 **How to Use - Web API**

### **🌐 Method 1: Interactive Web Interface**

1. **Start the server**: `python start.py`
2. **Open**: http://localhost:8000/docs
3. **Use the interactive interface** to test endpoints

### **📱 Method 2: cURL Commands**

#### **Similarity Check Only**:
```bash
curl -X POST "http://localhost:8000/similarity-check" \
  -F "excel_file=@your_questions.xlsx" \
  -F "master_questions=What is AI?" \
  -F "master_questions=How does ML work?" \
  -F "similarity_threshold=0.8" \
  --output "cleaned_questions.xlsx"
```

#### **Complete Syllabus Check**:
```bash
curl -X POST "http://localhost:8000/syllabus-check-text" \
  -F "excel_file=@your_questions.xlsx" \
  -F "master_questions=What is AI?" \
  -F "syllabus_content=This course covers machine learning basics..." \
  -F "similarity_threshold=0.8" \
  -F "relevance_threshold=0.6" \
  --output "syllabus_cleaned.xlsx"
```

### **🐍 Method 3: Python Code**

```python
import requests

# Simple similarity check
with open("questions.xlsx", "rb") as f:
    files = {"excel_file": f}
    data = {
        "master_questions": [
            "What is machine learning?",
            "How do neural networks work?"
        ],
        "similarity_threshold": 0.8
    }
    
    response = requests.post(
        "http://localhost:8000/similarity-check",
        files=files,
        data=data
    )
    
    if response.status_code == 200:
        # Save the cleaned Excel file
        with open("cleaned_questions.xlsx", "wb") as output:
            output.write(response.content)
            
        # Check processing statistics
        print(f"Processing time: {response.headers.get('X-Processing-Time')}s")
        print(f"Removed {response.headers.get('X-Removed-Count')} questions")
        print(f"Kept {response.headers.get('X-Remaining-Count')} questions")
```

---

## 🛠️ **How to Use - Python Library**

### **📚 Direct Library Usage**

```python
from syllabus_check import create_syllabus_checker
import pandas as pd

# Create a syllabus checker
checker = create_syllabus_checker(
    similarity_threshold=0.8,
    syllabus_relevance_threshold=0.6
)

# Load your questions
excel_path = "your_questions.xlsx"
master_questions = [
    "What is machine learning?",
    "How do neural networks work?"
]
syllabus_content = """
This course covers:
1. Introduction to AI and ML
2. Neural networks and deep learning  
3. Supervised learning algorithms
4. Model evaluation techniques
"""

# Process with complete pipeline
result = checker.process_complete_pipeline(
    excel_path=excel_path,
    master_questions=master_questions,
    syllabus_content=syllabus_content,
    output_path="cleaned_questions.xlsx"
)

# Print results
print(f"✅ Processing completed!")
print(f"📊 Original questions: {result['original_questions']}")
print(f"🗑️  Removed questions: {result['total_removed']}")
print(f"📋 Final questions: {result['final_questions']}")
print(f"📉 Reduction: {result['reduction_percentage']:.1f}%")
```

### **🔍 Step-by-Step Processing**

```python
from similarity import create_similarity_checker

# Step 1: Remove similar questions only
similarity_checker = create_similarity_checker(
    similarity_threshold=0.8
)

result1 = similarity_checker.process_excel_file(
    excel_path="questions.xlsx",
    master_questions=["What is AI?", "How does ML work?"],
    output_path="step1_cleaned.xlsx"
)

print(f"Step 1: Removed {result1['removed_count']} similar questions")

# Step 2: Continue with syllabus filtering if needed
# (Use the syllabus checker on the result from step 1)
```

---

## ⚙️ **Configuration Options**

### **🎛️ Similarity Threshold (0.0 - 1.0)**

| Value | Behavior | Use Case |
|-------|----------|----------|
| **0.9+** | Very strict - only nearly identical | Remove exact duplicates |
| **0.8** | Recommended default - good balance | General use |
| **0.7** | Moderate - catches more variations | Broader duplicate removal |
| **0.6-** | Liberal - may remove different questions | Be cautious |

### **📊 Relevance Threshold (0.0 - 1.0)**

| Value | Behavior | Use Case |
|-------|----------|----------|
| **0.8+** | Very strict - only highly relevant | Precise syllabus matching |
| **0.6** | Recommended default - good balance | General use |
| **0.4** | Moderate - includes related topics | Broader curriculum |
| **0.3-** | Liberal - may include unrelated | Be cautious |

### **🔧 Advanced Configuration**

```python
# Custom model selection
from embeddings import EmbeddingGenerator

# Use different embedding model
embedding_gen = EmbeddingGenerator(
    model_key="gte-large",  # Faster model
    models_dir="custom_models"
)

# Configure LLM provider
import os
os.environ['LLM_PROVIDER'] = 'openai'  # or 'groq', 'google'
```

---

## 📊 **Understanding Results**

### **📋 Output Excel File Structure**

Your processed file will contain multiple sheets:

#### **Sheet 1: "Cleaned_Questions"**
- ✅ Your final, cleaned question set
- Ready to use for exams or assessments

#### **Sheet 2: "Removed_Similar"**
- 🗑️ Questions removed for being too similar
- Shows which master question each was similar to
- Includes similarity scores

#### **Sheet 3: "Removed_Irrelevant"**
- 🚫 Questions removed for not matching syllabus
- Shows relevance scores
- Helps understand what was filtered out

#### **Sheet 4: "Processing_Stats"**
- 📊 Complete processing statistics
- Configuration used
- Timing information
- Reduction percentages

### **📈 Reading the Statistics**

```
Example Output:
✅ Processing completed in 3.45 seconds
📊 Original questions: 200
🗑️  Phase 1 removed: 25 (similar to master questions)
🚫 Phase 2 removed: 40 (not relevant to syllabus)
📋 Final questions: 135
📉 Total reduction: 32.5%
```

---

## 🎯 **Real-World Examples**

### **Example 1: Computer Science Course**

**Master Questions**:
```
- "What is object-oriented programming?"
- "How does inheritance work in OOP?"
- "Explain polymorphism with examples"
```

**Syllabus Content**:
```
This course covers object-oriented programming concepts including:
- Classes and objects
- Inheritance and polymorphism  
- Encapsulation and abstraction
- Design patterns
- Java programming language
```

**Expected Results**:
- Removes questions about web development (not in syllabus)
- Removes duplicate OOP questions
- Keeps relevant programming questions

### **Example 2: Biology Course**

**Master Questions**:
```
- "What is photosynthesis?"
- "How does cellular respiration work?"
- "Explain the structure of DNA"
```

**Syllabus Content**:
```
Introduction to Biology covering:
- Cell structure and function
- Genetics and DNA
- Plant biology and photosynthesis
- Energy metabolism
- Evolution principles
```

**Expected Results**:
- Removes questions about physics or chemistry
- Removes duplicate questions about photosynthesis
- Keeps relevant biology questions

---

## 🔧 **Troubleshooting**

### **❌ Common Issues**

#### **"No questions found in Excel file"**
**Solution**: 
- Ensure your Excel file has a column with questions
- Check column names (should contain "question", "Q", etc.)
- Verify file isn't corrupted

#### **"Model download failed"**
**Solution**:
- Check internet connection
- Ensure sufficient disk space (models are ~1GB)
- Try running again (download will resume)

#### **"Processing takes too long"**
**Solutions**:
- Use smaller question sets for testing
- Consider using `gte-large` model (faster)
- Increase thresholds to reduce processing

#### **"API server won't start"**
**Solutions**:
```bash
# Check if all dependencies installed
pip install -r requirements.txt

# Verify system status
python status_check.py

# Check for port conflicts
# Try different port: uvicorn main:app --port 8001
```

### **⚠️ Performance Tips**

#### **For Large Question Sets (500+)**:
- Start with higher thresholds (0.85+ similarity)
- Process in smaller batches
- Use the faster `gte-large` model
- Consider running on GPU if available

#### **For Production Use**:
- Use environment variables for API keys
- Set up proper error logging
- Consider database storage for results
- Implement request rate limiting

---

## 🎓 **Best Practices**

### **✅ Do's**
- **Start Conservative**: Use higher thresholds initially (0.8+)
- **Review Results**: Check removed questions to tune thresholds
- **Backup Original**: Always keep your original question files
- **Test Small**: Try with small datasets first
- **Check Quality**: Review the final output for accuracy

### **❌ Don'ts**  
- **Don't Use Very Low Thresholds**: Below 0.6 may remove good questions
- **Don't Skip Review**: Always review results before using
- **Don't Process Without Master Questions**: They're essential for good results
- **Don't Ignore Errors**: Check error messages for guidance
- **Don't Overload**: Process reasonable batch sizes

### **🎯 Workflow Recommendations**

1. **Preparation Phase**:
   - Organize your Excel file properly
   - Prepare clear master questions
   - Write detailed syllabus content

2. **Testing Phase**:
   - Start with high thresholds
   - Process small sample first
   - Review removed questions

3. **Fine-tuning Phase**:
   - Adjust thresholds based on results  
   - Re-process with optimized settings
   - Validate final output

4. **Production Phase**:
   - Process full dataset
   - Save all outputs and statistics
   - Document your settings for future use

---

## 🚀 **Advanced Usage**

### **🔄 Batch Processing**
```python
import os
from pathlib import Path

# Process multiple files
question_files = Path("question_files").glob("*.xlsx")
checker = create_syllabus_checker()

for file_path in question_files:
    print(f"Processing: {file_path.name}")
    
    result = checker.process_complete_pipeline(
        excel_path=str(file_path),
        master_questions=master_questions,
        syllabus_content=syllabus_content,
        output_path=f"cleaned_{file_path.name}"
    )
    
    print(f"✅ Reduced from {result['original_questions']} to {result['final_questions']}")
```

### **📊 Custom Reporting**
```python
# Generate custom analysis
results = []
for threshold in [0.7, 0.8, 0.9]:
    result = checker.process_complete_pipeline(
        excel_path="questions.xlsx",
        master_questions=master_questions,
        syllabus_content=syllabus_content,
        similarity_threshold=threshold,
        output_path=f"cleaned_t{threshold}.xlsx"
    )
    
    results.append({
        'threshold': threshold,
        'reduction': result['reduction_percentage'],
        'final_count': result['final_questions']
    })

# Find optimal threshold
optimal = min(results, key=lambda x: abs(x['reduction'] - 30))  # Target 30% reduction
print(f"Optimal threshold: {optimal['threshold']} (reduces {optimal['reduction']:.1f}%)")
```

---

## 🎉 **You're All Set!**

This user guide covers everything you need to effectively use the Syllabus Checker. Remember:

- **Start simple** with default settings
- **Review results** and adjust as needed  
- **Keep backups** of your original files
- **Experiment** with different thresholds
- **Check the docs** if you need more technical details

### **📞 Need Help?**
- Check the `status_check.py` output for system issues
- Review the `API_REFERENCE.md` for detailed API info
- Look at `example_usage.py` for more code examples
- Run `demo_api.py` to see the system in action

**Happy question cleaning!** 🎯✨