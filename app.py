import streamlit as st
import os
import tempfile
import pandas as pd
from io import BytesIO

# Import processors
from processors.pdf_processor import PDFProcessor
from processors.docx_processor import DOCXProcessor
from processors.csv_processor import CSVProcessor
from processors.image_processor import ImageProcessor
from processors.video_processor import VideoProcessor

# Import RAG utils
from utils.embeddings import EmbeddingModel
from utils.rag import RAGPipeline
from utils.claude_api import ClaudeAPI

# Page configuration
st.set_page_config(
    page_title="Multi-Document RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_store" not in st.session_state:
    st.session_state.document_store = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = []

if "file_processors" not in st.session_state:
    st.session_state.file_processors = {
        "pdf": PDFProcessor(),
        "docx": DOCXProcessor(),
        "csv": CSVProcessor(),
        "image": ImageProcessor(),
        "video": VideoProcessor()
    }

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = EmbeddingModel()

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline(st.session_state.embedding_model)

if "claude_api" not in st.session_state:
    st.session_state.claude_api = ClaudeAPI()

# Display title and introduction
st.title("📄 Multi-Document RAG Chatbot")
st.markdown(
    """
    This chatbot can answer questions based on uploaded documents, images, and videos.
    Upload your files, and start asking questions!
    """
)

# File uploader section
st.sidebar.header("Upload Documents")

# Function to process uploaded files
def process_file(uploaded_file):
    try:
        # Get file extension
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        # Create a temporary file to process
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process file based on its type
        if file_extension in ["pdf"]:
            processor = st.session_state.file_processors["pdf"]
            chunks = processor.process(tmp_path)
        elif file_extension in ["docx", "doc"]:
            processor = st.session_state.file_processors["docx"]
            chunks = processor.process(tmp_path)
        elif file_extension in ["csv"]:
            processor = st.session_state.file_processors["csv"]
            chunks = processor.process(tmp_path)
        elif file_extension in ["jpg", "jpeg", "png"]:
            processor = st.session_state.file_processors["image"]
            chunks = processor.process(tmp_path)
        elif file_extension in ["mp4", "avi", "mov"]:
            processor = st.session_state.file_processors["video"]
            chunks = processor.process(tmp_path)
        else:
            st.sidebar.error(f"Unsupported file type: {file_extension}")
            os.unlink(tmp_path)  # Clean up temp file
            return None
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Add processed chunks to document store with metadata
        for i, chunk in enumerate(chunks):
            doc_with_metadata = {
                "content": chunk,
                "metadata": {
                    "source": uploaded_file.name,
                    "chunk_id": i,
                    "file_type": file_extension
                }
            }
            st.session_state.document_store.append(doc_with_metadata)
        
        # Create embeddings for the chunks
        st.session_state.rag_pipeline.add_documents(chunks, [{"source": uploaded_file.name, "chunk_id": i} for i in range(len(chunks))])
        
        return len(chunks)
    
    except Exception as e:
        st.sidebar.error(f"Error processing file: {e}")
        return None

# Upload files
uploaded_files = st.sidebar.file_uploader(
    "Upload your documents (PDF, DOCX, CSV, Images, Videos)",
    type=["pdf", "docx", "csv", "jpg", "jpeg", "png", "mp4", "avi", "mov"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.sidebar.expander("Uploaded Files", expanded=True):
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in [doc["metadata"]["source"] for doc in st.session_state.document_store]:
                st.info(f"Processing {uploaded_file.name}...")
                chunks_count = process_file(uploaded_file)
                if chunks_count:
                    st.success(f"Processed {uploaded_file.name} into {chunks_count} chunks")
            else:
                st.info(f"{uploaded_file.name} already processed")

# Display document statistics
if st.session_state.document_store:
    with st.sidebar.expander("Document Statistics", expanded=True):
        total_chunks = len(st.session_state.document_store)
        unique_sources = set([doc["metadata"]["source"] for doc in st.session_state.document_store])
        
        st.write(f"📊 Total chunks: {total_chunks}")
        st.write(f"📚 Unique documents: {len(unique_sources)}")
        
        # Group by file type
        file_types = {}
        for doc in st.session_state.document_store:
            file_type = doc["metadata"]["file_type"]
            if file_type in file_types:
                file_types[file_type] += 1
            else:
                file_types[file_type] = 1
        
        for file_type, count in file_types.items():
            st.write(f"- {file_type.upper()}: {count} chunks")

# Clear documents button
if st.session_state.document_store:
    if st.sidebar.button("Clear All Documents"):
        st.session_state.document_store = []
        st.session_state.rag_pipeline.clear_index()
        st.sidebar.success("All documents cleared!")
        st.rerun()

# Chat interface
st.header("Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Show sources if available
        if "sources" in message and message["sources"]:
            with st.expander("View sources"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}:** {source['source']}")
                    st.markdown(f"*Excerpt:* {source['content'][:200]}...")

# User input
if user_input := st.chat_input("Ask a question about your documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if not st.session_state.document_store:
                response = "Please upload some documents first so I can answer your questions."
                sources = []
            else:
                # Get relevant documents
                relevant_docs = st.session_state.rag_pipeline.retrieve(user_input, top_k=5)
                
                # Format context from relevant documents
                context = "\n\n".join([doc["content"] for doc in relevant_docs])
                
                # Generate response using Claude API
                response = st.session_state.claude_api.generate_response(user_input, context)
                
                # Extract sources
                sources = [{"source": doc["metadata"]["source"], "content": doc["content"]} for doc in relevant_docs]
            
            # Display response
            st.write(response)
            
            # Show sources if available
            if sources:
                with st.expander("View sources"):
                    for i, source in enumerate(sources):
                        st.markdown(f"**Source {i+1}:** {source['source']}")
                        st.markdown(f"*Excerpt:* {source['content'][:200]}...")
            
            # Add assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": sources
            })
