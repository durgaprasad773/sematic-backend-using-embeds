# 📖 Syllabus Checker - Complete API Reference

## 🚀 **FastAPI Application Overview**

The Syllabus Checker API provides REST endpoints for intelligent question bank processing with two-phase filtering:
- **Phase 1**: Remove questions similar to master questions
- **Phase 2**: Remove questions not relevant to syllabus content

**Base URL**: `http://localhost:8000`  
**Interactive Docs**: `http://localhost:8000/docs`  
**OpenAPI Spec**: `http://localhost:8000/openapi.json`

---

## 🔍 **API Endpoints Reference**

### **🏥 Health & System Management**

#### `GET /health`
**Purpose**: Check system health and component availability.

**Response**: `200 OK`
```json
{
    "status": "healthy",
    "timestamp": "2025-09-22T10:30:00",
    "components": {
        "embedding_generator": "✅ Ready",
        "similarity_checker": "✅ Ready", 
        "syllabus_checker": "✅ Ready",
        "llm_provider": "✅ Ready (groq)"
    },
    "models": {
        "embedding_model": "bge-large-en",
        "model_dimension": 1024,
        "device": "cpu"
    },
    "system_info": {
        "python_version": "3.12.9",
        "torch_version": "2.6.0",
        "transformers_version": "4.51.3"
    }
}
```

**Error Response**: `503 Service Unavailable`
```json
{
    "status": "unhealthy",
    "error": "Failed to initialize embedding generator",
    "timestamp": "2025-09-22T10:30:00"
}
```

---

#### `GET /files`
**Purpose**: List all files in the processing directory.

**Response**: `200 OK`
```json
{
    "files": [
        "questions_similarity_1234567890.xlsx",
        "cleaned_syllabus_0987654321.xlsx"
    ],
    "count": 2,
    "timestamp": "2025-09-22T10:30:00"
}
```

---

#### `POST /cleanup`
**Purpose**: Manually clean up all processed files.

**Response**: `200 OK`
```json
{
    "message": "Cleanup completed successfully",
    "files_removed": 3,
    "timestamp": "2025-09-22T10:30:00"
}
```

---

### **🔍 Processing Endpoints (Direct Download)**

#### `POST /similarity-check`
**Purpose**: Phase 1 only - Remove questions similar to master questions.

**Content-Type**: `multipart/form-data`

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `excel_file` | File | ✅ Yes | - | Excel file with questions (.xlsx, .xls) |
| `master_questions` | List[str] | ✅ Yes | - | List of master questions to compare against |
| `similarity_threshold` | float | ❌ No | 0.8 | Similarity threshold (0.0-1.0) |

**Request Example (cURL)**:
```bash
curl -X POST "http://localhost:8000/similarity-check" \
  -F "excel_file=@questions.xlsx" \
  -F "master_questions=What is machine learning?" \
  -F "master_questions=How do neural networks work?" \
  -F "similarity_threshold=0.8" \
  --output "cleaned_questions.xlsx"
```

**Request Example (Python)**:
```python
import requests

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
        with open("cleaned_questions.xlsx", "wb") as output:
            output.write(response.content)
```

**Response**: `200 OK`
- **Content-Type**: `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`
- **Content-Disposition**: `attachment; filename="questions_similarity_1234567890.xlsx"`

**Response Headers**:
```
X-Processing-Time: 2.45
X-Original-Count: 150
X-Removed-Count: 23
X-Remaining-Count: 127
```

**Error Response**: `400 Bad Request`
```json
{
    "detail": "No valid questions found in the Excel file"
}
```

---

#### `POST /syllabus-check-text`
**Purpose**: Complete two-phase processing with text-based syllabus.

**Content-Type**: `multipart/form-data`

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `excel_file` | File | ✅ Yes | - | Excel file with questions |
| `master_questions` | List[str] | ✅ Yes | - | Master questions for similarity check |
| `syllabus_content` | str | ✅ Yes | - | Syllabus text content |
| `similarity_threshold` | float | ❌ No | 0.8 | Similarity threshold (0.0-1.0) |
| `relevance_threshold` | float | ❌ No | 0.6 | Relevance threshold (0.0-1.0) |

**Request Example (cURL)**:
```bash
curl -X POST "http://localhost:8000/syllabus-check-text" \
  -F "excel_file=@questions.xlsx" \
  -F "master_questions=What is AI?" \
  -F "syllabus_content=This course covers machine learning, neural networks, and AI algorithms..." \
  -F "similarity_threshold=0.8" \
  -F "relevance_threshold=0.6" \
  --output "syllabus_cleaned.xlsx"
```

**Request Example (Python)**:
```python
import requests

syllabus_text = """
This machine learning course covers:
1. Introduction to AI and ML
2. Neural networks and deep learning
3. Supervised and unsupervised learning
4. Model evaluation and optimization
5. Real-world applications
"""

with open("questions.xlsx", "rb") as f:
    files = {"excel_file": f}
    data = {
        "master_questions": ["What is AI?", "How does ML work?"],
        "syllabus_content": syllabus_text,
        "similarity_threshold": 0.8,
        "relevance_threshold": 0.6
    }
    
    response = requests.post(
        "http://localhost:8000/syllabus-check-text",
        files=files,
        data=data
    )
```

**Response**: `200 OK`
- **Content-Type**: `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`

**Response Headers**:
```
X-Processing-Time: 5.23
X-Original-Count: 150
X-Final-Count: 89
X-Total-Removed: 61
X-Reduction-Percentage: 40.7
```

---

#### `POST /syllabus-check-file`
**Purpose**: Complete two-phase processing with file-based syllabus.

**Content-Type**: `multipart/form-data`

**Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `excel_file` | File | ✅ Yes | - | Excel file with questions |
| `syllabus_file` | File | ✅ Yes | - | Syllabus text file (.txt) |
| `master_questions` | List[str] | ✅ Yes | - | Master questions for similarity check |
| `similarity_threshold` | float | ❌ No | 0.8 | Similarity threshold (0.0-1.0) |
| `relevance_threshold` | float | ❌ No | 0.6 | Relevance threshold (0.0-1.0) |

**Request Example (cURL)**:
```bash
curl -X POST "http://localhost:8000/syllabus-check-file" \
  -F "excel_file=@questions.xlsx" \
  -F "syllabus_file=@syllabus.txt" \
  -F "master_questions=What is AI?" \
  -F "similarity_threshold=0.8" \
  -F "relevance_threshold=0.6" \
  --output "syllabus_cleaned.xlsx"
```

**Request Example (Python)**:
```python
import requests

with open("questions.xlsx", "rb") as f1, open("syllabus.txt", "rb") as f2:
    files = {
        "excel_file": f1,
        "syllabus_file": f2
    }
    data = {
        "master_questions": ["What is AI?", "How does ML work?"],
        "similarity_threshold": 0.8,
        "relevance_threshold": 0.6
    }
    
    response = requests.post(
        "http://localhost:8000/syllabus-check-file",
        files=files,
        data=data
    )
```

---

## 📊 **Response Data Structures**

### **Excel Output Structure**

The processed Excel file contains multiple sheets:

#### **Sheet 1: "Cleaned_Questions"**
| Column | Description |
|---------|-------------|
| ID | Original question ID |
| Question | Question text |
| Original_Index | Index from original file |
| Processing_Notes | Any processing annotations |

#### **Sheet 2: "Removed_Similar"** (Phase 1 removals)
| Column | Description |
|---------|-------------|
| ID | Question ID |
| Question | Removed question text |
| Similar_To | Master question it was similar to |
| Similarity_Score | Similarity score (0.0-1.0) |
| Removal_Reason | "Similar to master question" |

#### **Sheet 3: "Removed_Irrelevant"** (Phase 2 removals)
| Column | Description |
|---------|-------------|
| ID | Question ID |
| Question | Removed question text |
| Relevance_Score | Relevance score (0.0-1.0) |
| Removal_Reason | "Not relevant to syllabus" |

#### **Sheet 4: "Processing_Stats"**
| Metric | Value |
|---------|--------|
| Original_Questions | 150 |
| Phase1_Removed | 23 |
| Phase2_Removed | 38 |
| Final_Questions | 89 |
| Total_Removed | 61 |
| Reduction_Percentage | 40.7% |
| Processing_Time | 5.23 seconds |
| Similarity_Threshold | 0.8 |
| Relevance_Threshold | 0.6 |
| Model_Used | bge-large-en |
| Timestamp | 2025-09-22T10:30:00 |

---

## ⚠️ **Error Handling**

### **Common Error Responses**

#### **400 Bad Request**
```json
{
    "detail": "No valid questions found in the Excel file"
}
```

#### **422 Unprocessable Entity**
```json
{
    "detail": [
        {
            "loc": ["form", "similarity_threshold"],
            "msg": "ensure this value is less than or equal to 1.0",
            "type": "value_error.number.not_le",
            "ctx": {"limit_value": 1.0}
        }
    ]
}
```

#### **500 Internal Server Error**
```json
{
    "detail": "Failed to process questions: Model initialization error"
}
```

### **Input Validation Rules**

#### **File Validation**:
- Excel files must be `.xlsx` or `.xls` format
- Text files must be `.txt` format
- Files must not be empty
- Excel files must contain valid question data

#### **Parameter Validation**:
- `similarity_threshold`: 0.0 ≤ value ≤ 1.0
- `relevance_threshold`: 0.0 ≤ value ≤ 1.0  
- `master_questions`: Must be non-empty list
- `syllabus_content`: Must be non-empty string

---

## 🔧 **API Configuration**

### **CORS Settings**
```python
# Allows all origins, methods, and headers
allow_origins=["*"]
allow_credentials=True
allow_methods=["*"]
allow_headers=["*"]
```

### **File Upload Limits**
- Maximum file size: Not explicitly limited (depends on server config)
- Supported Excel formats: `.xlsx`, `.xls`
- Supported text formats: `.txt`

### **Processing Limits**
- Maximum questions: No hard limit (depends on memory)
- Processing timeout: No explicit timeout
- Concurrent requests: Handled by FastAPI async capabilities

---

## 🧪 **Testing the API**

### **Health Check Test**
```bash
curl -X GET "http://localhost:8000/health"
```

### **Complete API Test Script**
```python
import requests
import pandas as pd

# Create sample data
questions_data = {
    "ID": [1, 2, 3, 4, 5],
    "Question": [
        "What is machine learning?",
        "How do neural networks work?",
        "What is the weather today?",
        "Explain supervised learning",
        "What is your favorite color?"
    ]
}

# Save to Excel
df = pd.DataFrame(questions_data)
df.to_excel("test_questions.xlsx", index=False)

# Test similarity check
with open("test_questions.xlsx", "rb") as f:
    files = {"excel_file": f}
    data = {
        "master_questions": ["What is ML?", "How do NNs work?"],
        "similarity_threshold": 0.7
    }
    
    response = requests.post(
        "http://localhost:8000/similarity-check",
        files=files,
        data=data
    )
    
    if response.status_code == 200:
        print("✅ Similarity check successful")
        print(f"Processing time: {response.headers.get('X-Processing-Time')}s")
        print(f"Removed: {response.headers.get('X-Removed-Count')} questions")
        
        # Save result
        with open("result.xlsx", "wb") as output:
            output.write(response.content)
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())
```

---

## 📈 **Performance Metrics**

### **Typical Processing Times**
| Questions | Similarity Only | Full Pipeline | 
|-----------|----------------|---------------|
| 50 | ~1-2 seconds | ~3-5 seconds |
| 100 | ~2-4 seconds | ~5-8 seconds |
| 500 | ~8-15 seconds | ~15-30 seconds |
| 1000+ | ~20+ seconds | ~40+ seconds |

### **Memory Usage**
- Base API: ~200-500 MB
- With models loaded: ~1-2 GB
- Per request processing: +50-200 MB (temporary)

### **Response Headers for Monitoring**
```
X-Processing-Time: Processing duration in seconds
X-Original-Count: Original number of questions
X-Removed-Count: Number of questions removed (similarity only)
X-Remaining-Count: Number of questions remaining (similarity only)
X-Final-Count: Final number of questions (full pipeline)
X-Total-Removed: Total questions removed (full pipeline)
X-Reduction-Percentage: Overall reduction percentage
```

This comprehensive API reference provides complete information for developers to integrate with and use the Syllabus Checker API effectively.