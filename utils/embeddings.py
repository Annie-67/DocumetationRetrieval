import numpy as np
import hashlib
from typing import List, Dict, Any

class EmbeddingModel:
    def __init__(self, embedding_dim: int = 512):
        """
        Initialize a simple embedding model.
        
        This uses a hash-based approach for generating embeddings 
        when the sentence-transformers package is not available.
        
        Args:
            embedding_dim (int): Dimension of the embeddings to generate
        """
        self.embedding_dim = embedding_dim
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """
        Convert text to a deterministic embedding vector using a hash-based approach
        
        Args:
            text (str): Text to convert to embedding
            
        Returns:
            np.ndarray: Embedding vector
        """
        # Create a deterministic seed from the text
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        seed = int(text_hash, 16) % (2**32)
        
        # Use the seed to generate a random-like but deterministic vector
        np.random.seed(seed)
        embedding = np.random.normal(size=self.embedding_dim)
        
        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            np.ndarray: Array of embeddings
        """
        try:
            # Generate embeddings for each text
            embeddings = np.array([self._text_to_embedding(text) for text in texts])
            return embeddings
        
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query text
        
        Args:
            query (str): Query text to embed
            
        Returns:
            np.ndarray: Query embedding
        """
        try:
            # Generate embedding for the query
            embedding = self._text_to_embedding(query)
            return embedding
        
        except Exception as e:
            raise Exception(f"Error generating query embedding: {str(e)}")
