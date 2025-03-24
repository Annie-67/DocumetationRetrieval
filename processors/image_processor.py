import os
import pytesseract
from PIL import Image
import cv2
import numpy as np
import base64
import anthropic
from anthropic import Anthropic
from typing import List, Dict, Any

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
        self.use_claude_vision = True
        
        # Get API key from environment variable for Claude Vision
        try:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self.claude_client = Anthropic(api_key=api_key)
                # Note: the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
                self.claude_model = "claude-3-5-sonnet-20241022"
            else:
                self.use_claude_vision = False
        except Exception:
            self.use_claude_vision = False
        
        # Configure pytesseract path if needed
        # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    
    def process(self, file_path: str) -> List[str]:
        """
        Process an image file and extract text and content description
        
        Args:
            file_path (str): Path to the image file
            
        Returns:
            List[str]: List of text chunks extracted from the image
        """
        try:
            # Get basic image metadata
            metadata = self._generate_image_metadata(file_path)
            
            # Process content with Claude Vision if available
            if self.use_claude_vision:
                # Generate detailed description using Claude's vision capabilities
                vision_description = self._analyze_image_with_claude(file_path)
                
                # Extract text from image using OCR as backup
                ocr_text = self._extract_text_from_image(file_path)
                
                # Combine all information
                if vision_description:
                    combined_text = f"Image Metadata: {metadata}\n\n{vision_description}"
                    
                    # Only add OCR text if it's not already captured in the vision description
                    if ocr_text and "No text detected" not in ocr_text:
                        combined_text += f"\n\nOCR Results: {ocr_text}"
                else:
                    # Fallback to just OCR and metadata if Claude Vision fails
                    combined_text = f"Image Metadata: {metadata}\n\n{ocr_text}"
            else:
                # Fallback to just OCR and metadata if Claude Vision is not available
                ocr_text = self._extract_text_from_image(file_path)
                combined_text = f"Image Metadata: {metadata}\n\n{ocr_text}"
            
            # Split text into chunks if it's large
            if len(combined_text) > self.chunk_size:
                chunks = self._chunk_text(combined_text)
            else:
                chunks = [combined_text]
            
            return chunks
        
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")
    
    def _analyze_image_with_claude(self, file_path: str) -> str:
        """
        Analyze image content using Claude's vision capabilities
        
        Args:
            file_path (str): Path to the image file
            
        Returns:
            str: Detailed description of the image content
        """
        try:
            if not self.use_claude_vision:
                return ""
            
            # Read image file and encode as base64
            with open(file_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")
            
            # Create message with image content
            message = self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=1000,
                system="You are an image analysis assistant. Provide detailed descriptions of images, focusing on key elements, text content, visual style, and overall composition. Be comprehensive but concise.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image
                                }
                            },
                            {
                                "type": "text",
                                "text": "Please provide a detailed description of this image, including any visible text, objects, people, scenes, colors, and other relevant details."
                            }
                        ]
                    }
                ]
            )
            
            # Extract the text response
            if message.content:
                return "Image Content Analysis:\n" + message.content[0].text
            else:
                return ""
                
        except Exception as e:
            return f"Claude Vision analysis unavailable: {str(e)}"
    
    def _extract_text_from_image(self, file_path: str) -> str:
        """
        Extract text from an image using Tesseract OCR
        
        Args:
            file_path (str): Path to the image file
            
        Returns:
            str: Extracted text from the image
        """
        try:
            # Open image with PIL
            image = Image.open(file_path)
            
            # Preprocess image for better OCR
            preprocessed_image = self._preprocess_image(image)
            
            # Extract text using Tesseract OCR
            extracted_text = pytesseract.image_to_string(preprocessed_image)
            
            if not extracted_text.strip():
                return "No text detected in the image using OCR."
            
            return "Extracted Text from Image (OCR):\n" + extracted_text
        except Exception as e:
            return f"OCR text extraction failed: {str(e)}"
    
    def _preprocess_image(self, image):
        """
        Preprocess image to improve OCR quality
        
        Args:
            image: PIL Image object
            
        Returns:
            PIL Image: Preprocessed image
        """
        try:
            # Convert PIL image to OpenCV format
            image_np = np.array(image)
            
            # Handle images with transparency (4 channels) or other color formats
            if len(image_np.shape) > 2:
                if image_np.shape[2] == 4:  # RGBA image
                    # Convert RGBA to RGB first
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
                    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                elif image_np.shape[2] == 3:  # RGB image
                    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                else:
                    # For other unusual formats, just take the first channel
                    gray = image_np[:, :, 0]
            else:
                # Already grayscale
                gray = image_np
            
            # Apply thresholding to get binary image
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            # Apply noise reduction
            denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
            
            # Convert back to PIL image
            preprocessed_image = Image.fromarray(denoised)
            
            return preprocessed_image
        except Exception:
            # If preprocessing fails, return the original image
            return image
    
    def _generate_image_metadata(self, file_path: str) -> str:
        """
        Generate metadata about the image file
        
        Args:
            file_path (str): Path to the image file
            
        Returns:
            str: Basic metadata of the image
        """
        try:
            # Open the image
            image = Image.open(file_path)
            
            # Get basic image information
            width, height = image.size
            format_type = image.format
            mode = image.mode
            
            # File information
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / 1024  # Size in KB
            
            # Generate basic description
            metadata = (
                f"Filename: {file_name}, "
                f"Dimensions: {width}x{height} pixels, "
                f"Format: {format_type}, "
                f"Mode: {mode}, "
                f"File size: {file_size:.1f} KB"
            )
            
            return metadata
        
        except Exception as e:
            return f"Basic image information: {os.path.basename(file_path)}, Error: {str(e)}"
    
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
