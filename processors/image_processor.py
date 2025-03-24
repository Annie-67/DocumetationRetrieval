import os
import pytesseract
from PIL import Image
import cv2
import numpy as np
from typing import List

class ImageProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the Image processor
        
        Args:
            chunk_size (int): Maximum size of each text chunk
            chunk_overlap (int): Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Configure pytesseract path if needed
        # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    
    def process(self, file_path: str) -> List[str]:
        """
        Process an image file and extract text
        
        Args:
            file_path (str): Path to the image file
            
        Returns:
            List[str]: List of text chunks extracted from the image
        """
        try:
            # Extract text from image
            text = self._extract_text_from_image(file_path)
            
            # Get image description
            description = self._generate_image_description(file_path)
            
            # Combine text and description
            combined_text = f"Image Description: {description}\n\n" + text
            
            # Split text into chunks if it's large
            if len(combined_text) > self.chunk_size:
                chunks = self._chunk_text(combined_text)
            else:
                chunks = [combined_text]
            
            return chunks
        
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")
    
    def _extract_text_from_image(self, file_path: str) -> str:
        """
        Extract text from an image using Tesseract OCR
        
        Args:
            file_path (str): Path to the image file
            
        Returns:
            str: Extracted text from the image
        """
        # Open image with PIL
        image = Image.open(file_path)
        
        # Preprocess image for better OCR
        preprocessed_image = self._preprocess_image(image)
        
        # Extract text using Tesseract OCR
        extracted_text = pytesseract.image_to_string(preprocessed_image)
        
        if not extracted_text.strip():
            return "No text detected in the image."
        
        return "Extracted Text from Image:\n" + extracted_text
    
    def _preprocess_image(self, image):
        """
        Preprocess image to improve OCR quality
        
        Args:
            image: PIL Image object
            
        Returns:
            PIL Image: Preprocessed image
        """
        # Convert PIL image to OpenCV format
        image_np = np.array(image)
        
        # Convert to grayscale if the image is in color
        if len(image_np.shape) > 2 and image_np.shape[2] == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        # Apply thresholding to get binary image
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Apply noise reduction
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        # Convert back to PIL image
        preprocessed_image = Image.fromarray(denoised)
        
        return preprocessed_image
    
    def _generate_image_description(self, file_path: str) -> str:
        """
        Generate a basic description of the image
        
        Args:
            file_path (str): Path to the image file
            
        Returns:
            str: Basic description of the image
        """
        try:
            # Open the image
            image = Image.open(file_path)
            
            # Get basic image information
            width, height = image.size
            format_type = image.format
            mode = image.mode
            
            # Analyze image content
            image_np = np.array(image)
            
            # Check if image is grayscale or color
            image_type = "Grayscale" if len(image_np.shape) < 3 else "Color"
            
            # Check if image might contain text
            has_text = "Yes" if self._might_contain_text(image_np) else "Unknown"
            
            # File information
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / 1024  # Size in KB
            
            # Generate description
            description = (
                f"Filename: {file_name}, "
                f"Dimensions: {width}x{height} pixels, "
                f"Format: {format_type}, "
                f"Mode: {mode}, "
                f"Type: {image_type}, "
                f"May contain text: {has_text}, "
                f"File size: {file_size:.1f} KB"
            )
            
            return description
        
        except Exception as e:
            return f"Unable to generate image description: {str(e)}"
    
    def _might_contain_text(self, image_np):
        """
        Simple heuristic to check if an image might contain text
        
        Args:
            image_np: Image as numpy array
            
        Returns:
            bool: True if the image might contain text
        """
        # Convert to grayscale if needed
        if len(image_np.shape) > 2 and image_np.shape[2] == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        # Apply edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Count the number of edges
        edge_count = np.count_nonzero(edges)
        
        # Calculate edge density
        edge_density = edge_count / (gray.shape[0] * gray.shape[1])
        
        # Images with text typically have higher edge density
        return edge_density > 0.05
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text (str): Text to be chunked
            
        Returns:
            List[str]: List of text chunks
        """
        chunks = []
        
        if len(text) <= self.chunk_size:
            chunks.append(text)
            return chunks
        
        # Split text into chunks with overlap
        start = 0
        while start < len(text):
            # Find end of chunk
            end = start + self.chunk_size
            
            # Adjust end to not break mid-sentence if possible
            if end < len(text):
                # Try to end at a period, question mark, or exclamation point
                punctuation = max(text.rfind('. ', start, end), 
                                text.rfind('? ', start, end),
                                text.rfind('! ', start, end))
                
                if punctuation != -1:
                    end = punctuation + 1
            
            # Get chunk and add to list
            chunk = text[start:min(end, len(text))].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move start position for next chunk
            start = end - self.chunk_overlap
            
            # Handle case where overlap might create a start position that's invalid
            if start < 0 or start >= len(text):
                break
        
        return chunks
