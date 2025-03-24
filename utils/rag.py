import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from utils.embeddings import EmbeddingModel

class RAGPipeline:
    def __init__(self, embedding_model: EmbeddingModel):
        """
        Initialize the RAG pipeline
        
        Args:
            embedding_model (EmbeddingModel): Model for generating embeddings
        """
        self.embedding_model = embedding_model
        self.documents = []
        self.document_metadatas = []
        self.index = None
        self.dimension = None
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]):
        """
        Add documents to the RAG pipeline
        
        Args:
            documents (List[str]): List of document texts
            metadatas (List[Dict[str, Any]]): List of metadata for each document
        """
        try:
            # Check if documents and metadatas have the same length
            if len(documents) != len(metadatas):
                raise ValueError("Documents and metadatas must have the same length")
            
            # Extend documents and metadatas
            self.documents.extend(documents)
            self.document_metadatas.extend(metadatas)
            
            # Generate embeddings for the documents
            embeddings = self.embedding_model.embed_texts(documents)
            
            # Initialize or update the index
            if self.index is None:
                self.dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index.add(embeddings.astype(np.float32))
            else:
                if embeddings.shape[1] != self.dimension:
                    raise ValueError(f"Expected embedding dimension {self.dimension}, got {embeddings.shape[1]}")
                self.index.add(embeddings.astype(np.float32))
        
        except Exception as e:
            raise Exception(f"Error adding documents: {str(e)}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query (str): Query text
            top_k (int): Number of documents to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of retrieved documents with their content and metadata
        """
        try:
            if not self.documents or self.index is None:
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # Reshape embedding for faiss
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            
            # Search for similar documents
            distances, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
            
            # Get the retrieved documents
            retrieved_docs = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):  # Safety check
                    retrieved_docs.append({
                        "content": self.documents[idx],
                        "metadata": self.document_metadatas[idx],
                        "score": float(distances[0][i])  # Convert to Python float
                    })
            
            return retrieved_docs
        
        except Exception as e:
            raise Exception(f"Error retrieving documents: {str(e)}")
    
    def clear_index(self):
        """
        Clear the index and all stored documents
        """
        self.documents = []
        self.document_metadatas = []
        
        if self.dimension is not None:
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.index = None
