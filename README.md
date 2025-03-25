
# DocumentRetrievalChatbot

A **Retrieval-Augmented Generation (RAG)** assistant that handles multiple document formats including **PDFs, DOCX, CSVs, images, and videos**—all **without using LangChain**. The project uses **Streamlit** for the frontend and integrates **Claude API** for intelligent response generation.

---

##  Architecture Overview

### 1. Main Application (`app.py`)
- Built with **Streamlit**
- Handles:
  - File uploads
  - User interaction
  - Document processing pipeline
  - Claude API response generation

### 2. Document Processors
- **PDF**: `PyPDF2` – Text extraction  
- **DOCX**: `python-docx` – Paragraph parsing  
- **CSV**: `pandas` – Structured data handling  
- **Images**: `OpenCV + Tesseract OCR` – OCR & vision analysis  
- **Videos**: `OpenCV + Whisper` – Audio transcription  

### 3. Core Components
- **RAG Pipeline**: Custom implementation using **FAISS**
- **Embedding Model**: Custom embedding generation
- **Claude API**: Direct integration (no LangChain)

---

##  Features

### LangChain-Free Implementation
- Entirely custom pipeline
- No dependency on third-party orchestration

### Multi-Modal File Support
- PDFs, DOCX  
- CSV files  
- Images (via OCR)  
- Videos (via Whisper transcription)

### Custom RAG Features
- Chunking with overlap for better context
- FAISS-based vector similarity search
- Context-aware Claude API response generation

---

## Technical Breakdown

### 1. Document Processing
def process(self, file_path: str) -> List[str]:
    # Processes document and returns text chunks
### 2. RAG Pipeline (FAISS)
class RAGPipeline:
    def __init__(self, embedding_model):
        self.index = faiss.IndexFlatL2(dimension)

    def retrieve(self, query: str, top_k: int = 5):
        # Perform FAISS-based similarity search
#### 3. Embedding Generation
def embed_texts(self, texts: List[str]) -> np.ndarray:
    # Generate embeddings for text chunks
### Deployment - Streamlit
![image](https://github.com/user-attachments/assets/b5978a60-2ed4-40c9-9eb7-1b3c11815263)
![image](https://github.com/user-attachments/assets/e4457c70-a7b1-4e05-9479-ced6830a6c2b)
![image](https://github.com/user-attachments/assets/8ab71912-d34e-44fe-ac96-61468ad88543)
![image](https://github.com/user-attachments/assets/bdacca25-9d7c-4064-bbfb-b78712344b82)



