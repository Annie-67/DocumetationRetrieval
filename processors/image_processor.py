import os
import pytesseract
from PIL import Image
import cv2
import numpy as np
import base64
import anthropic
from anthropic import Anthropic
from typing import List, Dict, Any, Tuple
import math
import json
import requests

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
        self.use_huggingface_vision = True
        
        # Get API key from environment variable for Claude Vision
        try:
            anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
            if anthropic_api_key:
                self.claude_client = Anthropic(api_key=anthropic_api_key)
                # Note: the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
                self.claude_model = "claude-3-5-sonnet-20241022"
            else:
                self.use_claude_vision = False
        except Exception:
            self.use_claude_vision = False
            
        # Get API key from environment variable for Hugging Face
        try:
            self.hf_api_key = os.environ.get("HUGGINGFACE_API_KEY")
            if not self.hf_api_key:
                self.use_huggingface_vision = False
        except Exception:
            self.use_huggingface_vision = False
            
        # Define Hugging Face models to use
        self.image_captioning_model = "Salesforce/blip-image-captioning-large"
        self.image_classification_model = "google/vit-base-patch16-224"
        self.object_detection_model = "facebook/detr-resnet-50"
        self.ocr_model = "microsoft/trocr-base-handwritten"
    
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
            
            # Perform Hugging Face analysis if available
            hf_analysis = ""
            if self.use_huggingface_vision:
                hf_analysis = self._analyze_image_with_huggingface(file_path)
            
            # If Hugging Face analysis is not available, use pixel-level analysis
            if not hf_analysis:
                detailed_analysis = self._analyze_image_pixel_level(file_path)
            else:
                detailed_analysis = hf_analysis
            
            # Extract text using OCR
            ocr_text = self._extract_text_from_image(file_path)
            
            # Also try to get Claude Vision analysis if available
            claude_analysis = ""
            if self.use_claude_vision:
                claude_analysis = self._analyze_image_with_claude(file_path)
            
            # Combine all information
            combined_text = f"Image Metadata: {metadata}\n\n"
            combined_text += f"Detailed Image Analysis:\n{detailed_analysis}\n\n"
            
            if ocr_text and "No text detected" not in ocr_text:
                combined_text += f"{ocr_text}\n\n"
                
            if claude_analysis:
                combined_text += f"{claude_analysis}"
            
            # Split text into chunks if it's large
            if len(combined_text) > self.chunk_size:
                chunks = self._chunk_text(combined_text)
            else:
                chunks = [combined_text]
            
            return chunks
        
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")
    
    def _analyze_image_pixel_level(self, file_path: str) -> str:
        """
        Perform detailed pixel-level analysis of the image
        
        Args:
            file_path (str): Path to the image file
            
        Returns:
            str: Detailed analysis of the image content
        """
        try:
            # Open image
            image = Image.open(file_path)
            image_np = np.array(image)
            
            # Handle different image modes
            if len(image_np.shape) > 2:
                if image_np.shape[2] == 4:  # RGBA image
                    # Convert RGBA to RGB
                    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
                elif image_np.shape[2] == 3:  # RGB image
                    image_rgb = image_np
                else:
                    # For other unusual formats, just take the first channel
                    image_rgb = np.stack([image_np[:, :, 0]] * 3, axis=2)
            else:
                # Convert grayscale to RGB
                image_rgb = np.stack([image_np] * 3, axis=2)
            
            # 1. Color analysis
            color_analysis = self._analyze_color_distribution(image_rgb)
            
            # 2. Edge detection for shapes/objects
            shape_analysis = self._detect_shapes_and_objects(image_rgb)
            
            # 3. Texture analysis
            texture_analysis = self._analyze_texture(image_rgb)
            
            # 4. Detect faces if any
            face_analysis = self._detect_faces(image_rgb)
            
            # 5. Composition analysis
            composition_analysis = self._analyze_composition(image_rgb)
            
            # Combine all analyses
            detailed_analysis = (
                f"{color_analysis}\n"
                f"{shape_analysis}\n"
                f"{texture_analysis}\n"
                f"{face_analysis}\n"
                f"{composition_analysis}"
            )
            
            return detailed_analysis
            
        except Exception as e:
            return f"Error in pixel-level analysis: {str(e)}"
    
    def _analyze_color_distribution(self, image_np: np.ndarray) -> str:
        """
        Analyze the color distribution in the image
        
        Args:
            image_np (np.ndarray): Image as a numpy array
            
        Returns:
            str: Color analysis description
        """
        try:
            # Convert to HSV for better color analysis
            hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
            
            # Get average hue, saturation, value
            avg_hue = np.mean(hsv_image[:, :, 0])
            avg_saturation = np.mean(hsv_image[:, :, 1])
            avg_brightness = np.mean(hsv_image[:, :, 2])
            
            # Determine dominant colors by binning hues
            hue_ranges = [
                (0, 15, "red"),
                (15, 25, "orange"),
                (25, 35, "yellow"),
                (35, 75, "green"),
                (75, 95, "teal"),
                (95, 135, "blue"),
                (135, 165, "purple"),
                (165, 180, "pink/red")
            ]
            
            # Only consider pixels with sufficient saturation and brightness
            valid_pixels = hsv_image[(hsv_image[:, :, 1] > 50) & (hsv_image[:, :, 2] > 50)]
            
            if len(valid_pixels) > 0:
                hues = valid_pixels[:, 0]
                
                # Count pixels in each hue range
                color_counts = {}
                for hue_min, hue_max, color_name in hue_ranges:
                    count = np.sum((hues >= hue_min) & (hues < hue_max))
                    if count > 0:
                        color_counts[color_name] = count
                
                # Sort colors by frequency
                sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
                
                if sorted_colors:
                    total_color_pixels = sum(count for _, count in sorted_colors)
                    dominant_colors = ", ".join([f"{color} ({count/total_color_pixels*100:.1f}%)" 
                                              for color, count in sorted_colors[:3] 
                                              if count/total_color_pixels > 0.05])
                else:
                    dominant_colors = "no dominant colors detected"
            else:
                # Likely a grayscale or low saturation image
                dominant_colors = "grayscale or muted colors"
            
            # Get brightness description
            if avg_brightness < 50:
                brightness = "dark"
            elif avg_brightness > 200:
                brightness = "very bright"
            elif avg_brightness > 150:
                brightness = "bright"
            else:
                brightness = "moderate brightness"
            
            # Get saturation description
            if avg_saturation < 30:
                saturation = "desaturated/grayscale"
            elif avg_saturation < 100:
                saturation = "moderately saturated"
            else:
                saturation = "highly saturated"
            
            # Assemble color analysis
            color_analysis = (
                f"Color Analysis: This image is {brightness} and {saturation}. "
                f"Dominant colors: {dominant_colors}."
            )
            
            return color_analysis
            
        except Exception as e:
            return f"Color analysis unavailable: {str(e)}"
    
    def _detect_shapes_and_objects(self, image_np: np.ndarray) -> str:
        """
        Detect shapes and potential objects in the image
        
        Args:
            image_np (np.ndarray): Image as a numpy array
            
        Returns:
            str: Shape and object analysis
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Use Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter out tiny contours
            significant_contours = [c for c in contours if cv2.contourArea(c) > 100]
            
            # Count objects of different sizes
            small_objects = 0
            medium_objects = 0
            large_objects = 0
            
            image_area = image_np.shape[0] * image_np.shape[1]
            
            for contour in significant_contours:
                area = cv2.contourArea(contour)
                area_ratio = area / image_area
                
                if area_ratio < 0.01:
                    small_objects += 1
                elif area_ratio < 0.1:
                    medium_objects += 1
                else:
                    large_objects += 1
            
            # Determine if image has a main subject
            has_main_subject = large_objects > 0
            
            # Check for geometric shapes
            shapes = []
            for contour in significant_contours:
                # Approximate the contour
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                
                # Identify shape based on vertices and properties
                vertices = len(approx)
                shape_name = "unknown"
                
                if vertices == 3:
                    shape_name = "triangle"
                elif vertices == 4:
                    # Check if it's a square or rectangle
                    if 0.95 <= aspect_ratio <= 1.05:
                        shape_name = "square"
                    else:
                        shape_name = "rectangle"
                elif vertices == 5:
                    shape_name = "pentagon"
                elif vertices == 6:
                    shape_name = "hexagon"
                elif vertices > 10:
                    # Check circularity
                    area = cv2.contourArea(contour)
                    circularity = 4 * math.pi * area / (perimeter * perimeter)
                    if circularity > 0.8:
                        shape_name = "circle"
                
                if shape_name != "unknown":
                    shapes.append(shape_name)
            
            # Count shape frequencies
            shape_counts = {}
            for shape in shapes:
                if shape in shape_counts:
                    shape_counts[shape] += 1
                else:
                    shape_counts[shape] = 1
            
            # Format shape information
            shape_info = ", ".join([f"{count} {shape}{'s' if count > 1 else ''}" 
                                 for shape, count in shape_counts.items()])
            
            # Assemble object analysis
            object_analysis = (
                f"Object Analysis: Detected approximately {len(significant_contours)} distinct elements "
                f"({small_objects} small, {medium_objects} medium, {large_objects} large). "
            )
            
            if has_main_subject:
                object_analysis += "The image appears to have a main subject. "
            else:
                object_analysis += "The image appears to contain multiple elements or a scene. "
                
            if shape_info:
                object_analysis += f"Detected shapes: {shape_info}."
            
            return object_analysis
            
        except Exception as e:
            return f"Shape/object analysis unavailable: {str(e)}"
    
    def _analyze_texture(self, image_np: np.ndarray) -> str:
        """
        Analyze the texture characteristics of the image
        
        Args:
            image_np (np.ndarray): Image as a numpy array
            
        Returns:
            str: Texture analysis
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            # Calculate local standard deviation as a measure of texture
            local_std = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Interpret the texture
            if local_std < 100:
                texture_type = "smooth"
            elif local_std < 500:
                texture_type = "moderate"
            elif local_std < 2000:
                texture_type = "textured"
            else:
                texture_type = "highly detailed"
            
            # Get gradient magnitude using Sobel filters
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            avg_gradient = np.mean(gradient_magnitude)
            
            # Interpret gradient
            if avg_gradient < 10:
                gradient_desc = "soft transitions"
            elif avg_gradient < 30:
                gradient_desc = "moderate contrast"
            else:
                gradient_desc = "sharp contrasts"
            
            # Assemble texture analysis
            texture_analysis = (
                f"Texture Analysis: The image has a {texture_type} texture with {gradient_desc}. "
                f"Detail level: {int(local_std)}/2000."
            )
            
            return texture_analysis
            
        except Exception as e:
            return f"Texture analysis unavailable: {str(e)}"
    
    def _detect_faces(self, image_np: np.ndarray) -> str:
        """
        Attempt to detect faces in the image
        
        Args:
            image_np (np.ndarray): Image as a numpy array
            
        Returns:
            str: Face detection results
        """
        try:
            # Load the face cascade classifier
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            
            # Check if the cascade file exists
            if not os.path.exists(face_cascade_path):
                return "Face detection unavailable: Classifier file not found."
                
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                return f"People: Detected {len(faces)} {'person' if len(faces) == 1 else 'people'} in the image."
            else:
                # Check for potential people using HOG descriptor (simplified)
                # This is a rough heuristic that might indicate people in the image
                win_size = (64, 128)
                hog = cv2.HOGDescriptor()
                hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                
                # Resize image if it's too large for HOG
                if image_np.shape[0] > 800 or image_np.shape[1] > 800:
                    scale = 800 / max(image_np.shape[0], image_np.shape[1])
                    resized = cv2.resize(image_np, (0, 0), fx=scale, fy=scale)
                else:
                    resized = image_np
                    
                # Detect people
                bodies, _ = hog.detectMultiScale(resized, winStride=(8, 8), padding=(4, 4), scale=1.05)
                
                if len(bodies) > 0:
                    return f"People: Potentially detected {len(bodies)} human figures in the image."
                else:
                    return "People: No people detected in the image."
                    
        except Exception as e:
            return f"Face/people detection unavailable: {str(e)}"
    
    def _analyze_composition(self, image_np: np.ndarray) -> str:
        """
        Analyze the composition of the image
        
        Args:
            image_np (np.ndarray): Image as a numpy array
            
        Returns:
            str: Composition analysis
        """
        try:
            height, width = image_np.shape[:2]
            
            # Check aspect ratio
            aspect_ratio = width / height
            
            if 0.95 <= aspect_ratio <= 1.05:
                aspect_desc = "square"
            elif aspect_ratio > 1:
                if aspect_ratio >= 1.9:
                    aspect_desc = "panoramic landscape"
                else:
                    aspect_desc = "landscape"
            else:
                if aspect_ratio <= 0.6:
                    aspect_desc = "tall portrait"
                else:
                    aspect_desc = "portrait"
            
            # Analyze complexity by dividing the image into regions and measuring variety
            regions_h = 3
            regions_w = 3
            region_height = height // regions_h
            region_width = width // regions_w
            
            region_stats = []
            for y in range(regions_h):
                for x in range(regions_w):
                    y1 = y * region_height
                    y2 = min((y + 1) * region_height, height)
                    x1 = x * region_width
                    x2 = min((x + 1) * region_width, width)
                    
                    region = image_np[y1:y2, x1:x2]
                    region_mean = np.mean(region, axis=(0, 1))
                    region_std = np.std(region, axis=(0, 1))
                    
                    region_stats.append((region_mean, region_std))
            
            # Calculate variance between regions
            means = np.array([stat[0] for stat in region_stats])
            region_variance = np.mean(np.var(means, axis=0))
            
            # Interpret composition based on region variance
            if region_variance < 100:
                composition_desc = "minimalist/uniform"
            elif region_variance < 500:
                composition_desc = "balanced"
            elif region_variance < 1000:
                composition_desc = "varied"
            else:
                composition_desc = "high contrast/complex"
            
            # Determine focus area
            # Get saliency map using gradient magnitude to estimate where the focus might be
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(sobelx, sobely)
            
            # Get the region with highest average gradient magnitude
            region_gradients = []
            for y in range(regions_h):
                for x in range(regions_w):
                    y1 = y * region_height
                    y2 = min((y + 1) * region_height, height)
                    x1 = x * region_width
                    x2 = min((x + 1) * region_width, width)
                    
                    region_grad = np.mean(magnitude[y1:y2, x1:x2])
                    region_gradients.append((x, y, region_grad))
            
            # Find region with highest gradient
            focus_region = max(region_gradients, key=lambda x: x[2])
            focus_x, focus_y = focus_region[0], focus_region[1]
            
            # Determine focus position
            if focus_y == 0:
                v_pos = "top"
            elif focus_y == 1:
                v_pos = "middle"
            else:
                v_pos = "bottom"
                
            if focus_x == 0:
                h_pos = "left"
            elif focus_x == 1:
                h_pos = "center"
            else:
                h_pos = "right"
            
            focus_position = f"{v_pos} {h_pos}"
            
            # Assemble composition analysis
            composition_analysis = (
                f"Composition Analysis: This is a {aspect_desc} format image with a {composition_desc} composition. "
                f"The main area of visual interest appears to be in the {focus_position} of the frame."
            )
            
            return composition_analysis
            
        except Exception as e:
            return f"Composition analysis unavailable: {str(e)}"
    
    def _analyze_image_with_huggingface(self, file_path: str) -> str:
        """
        Analyze image using Hugging Face vision models
        
        Args:
            file_path (str): Path to the image file
            
        Returns:
            str: Detailed analysis from Hugging Face models
        """
        try:
            if not self.use_huggingface_vision or not self.hf_api_key:
                return ""
            
            # Read image file and encode as base64
            with open(file_path, "rb") as img_file:
                image_bytes = img_file.read()
                base64_image = base64.b64encode(image_bytes).decode("utf-8")
            
            # 1. Image captioning using BLIP model
            caption = self._get_image_caption(image_bytes)
            
            # 2. Image classification using ViT model
            classifications = self._classify_image(image_bytes)
            
            # 3. Object detection using DETR model
            objects = self._detect_objects(image_bytes)
            
            # Combine all analyses
            hf_analysis = f"Image Caption: {caption}\n\n"
            
            if classifications:
                hf_analysis += "Classification Results:\n"
                for label, score in classifications:
                    hf_analysis += f"- {label}: {score:.1f}%\n"
                hf_analysis += "\n"
            
            if objects:
                hf_analysis += "Detected Objects:\n"
                for obj in objects:
                    hf_analysis += f"- {obj['label']} (confidence: {obj['score']:.1f}%)\n"
            
            return hf_analysis
            
        except Exception as e:
            return f"Hugging Face analysis unavailable: {str(e)}"
    
    def _get_image_caption(self, image_bytes: bytes) -> str:
        """
        Get image caption using Hugging Face's BLIP model
        
        Args:
            image_bytes (bytes): Raw image bytes
            
        Returns:
            str: Generated caption
        """
        try:
            headers = {"Authorization": f"Bearer {self.hf_api_key}"}
            API_URL = f"https://api-inference.huggingface.co/models/{self.image_captioning_model}"
            
            response = requests.post(API_URL, headers=headers, data=image_bytes)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "No caption generated")
                elif isinstance(result, dict):
                    return result.get("generated_text", "No caption generated")
                else:
                    return "Caption unavailable"
            else:
                return f"Caption request failed with status {response.status_code}"
        
        except Exception as e:
            return f"Caption generation error: {str(e)}"
    
    def _classify_image(self, image_bytes: bytes) -> List[Tuple[str, float]]:
        """
        Classify image using Hugging Face's ViT model
        
        Args:
            image_bytes (bytes): Raw image bytes
            
        Returns:
            List[Tuple[str, float]]: List of (label, score) tuples
        """
        try:
            headers = {"Authorization": f"Bearer {self.hf_api_key}"}
            API_URL = f"https://api-inference.huggingface.co/models/{self.image_classification_model}"
            
            response = requests.post(API_URL, headers=headers, data=image_bytes)
            
            if response.status_code == 200:
                results = response.json()
                
                # Format results as list of (label, score) tuples
                return [(item["label"], item["score"] * 100) for item in results[:5]]
            else:
                return []
        
        except Exception:
            return []
    
    def _detect_objects(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Detect objects in image using Hugging Face's DETR model
        
        Args:
            image_bytes (bytes): Raw image bytes
            
        Returns:
            List[Dict[str, Any]]: List of detected objects with labels and scores
        """
        try:
            headers = {"Authorization": f"Bearer {self.hf_api_key}"}
            API_URL = f"https://api-inference.huggingface.co/models/{self.object_detection_model}"
            
            response = requests.post(API_URL, headers=headers, data=image_bytes)
            
            if response.status_code == 200:
                results = response.json()
                
                # Format results as list of detection dictionaries
                detections = []
                for item in results:
                    detections.append({
                        "label": item["label"],
                        "score": item["score"] * 100,
                        "box": item["box"]
                    })
                
                # Sort by confidence score
                detections.sort(key=lambda x: x["score"], reverse=True)
                
                return detections[:10]  # Return top 10 detections
            else:
                return []
        
        except Exception:
            return []
    
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
                return "Claude Vision Analysis:\n" + message.content[0].text
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
