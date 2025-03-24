import os
import anthropic
from anthropic import Anthropic
import json
from typing import List, Dict, Any

class ClaudeAPI:
    def __init__(self, max_tokens: int = 1000):
        """
        Initialize the Claude API client
        
        Args:
            max_tokens (int): Maximum number of tokens for the response
        """
        # Get API key from environment variable
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        # Note: the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
        self.model = "claude-3-5-sonnet-20241022"
        self.max_tokens = max_tokens
        
        # Initialize Anthropic client
        self.client = Anthropic(api_key=self.api_key)
    
    def generate_response(self, query: str, context: str) -> str:
        """
        Generate a response using Claude API
        
        Args:
            query (str): User query
            context (str): Context from retrieved documents
            
        Returns:
            str: Generated response
        """
        try:
            # Prepare system message with instructions
            system_message = (
                "You are a helpful assistant that answers questions based on the provided context. "
                "If the answer is not in the context, say 'I don't have enough information to answer this question'. "
                "Do not make up information. Always provide accurate information based only on the given context."
            )
            
            # Prepare user message with context and query
            user_message = f"Context:\n{context}\n\nQuestion: {query}"
            
            # Generate response using Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_message,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            
            # Extract the text content
            if response.content:
                return response.content[0].text
            else:
                return "I'm sorry, I couldn't generate a response. Please try again."
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
