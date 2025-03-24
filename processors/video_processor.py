import os
import cv2
import tempfile
import numpy as np
import pytesseract
from PIL import Image
import subprocess
import base64
import json
import requests
from typing import List, Dict, Any, Optional

class VideoProcessor:
    def __init__(self, frame_interval: int = 30, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the Video processor
        
        Args:
            frame_interval (int): Number of frames to skip between processing
            chunk_size (int): Maximum size of each text chunk
            chunk_overlap (int): Number of characters to overlap between chunks
        """
        self.frame_interval = frame_interval
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_huggingface = True
        
        # Get API key from environment variable for Hugging Face
        try:
            self.hf_api_key = os.environ.get("HUGGINGFACE_API_KEY")
            if not self.hf_api_key:
                self.use_huggingface = False
        except Exception:
            self.use_huggingface = False
        
        # Define Hugging Face models to use
        self.speech_recognition_model = "facebook/wav2vec2-large-960h-lv60-self"
        self.video_classification_model = "MCG-NJU/videomae-base-finetuned-kinetics"
    
    def process(self, file_path: str) -> List[str]:
        """
        Process a video file and extract content
        
        Args:
            file_path (str): Path to the video file
            
        Returns:
            List[str]: List of text chunks extracted from the video
        """
        try:
            # Extract video metadata
            metadata = self._extract_video_metadata(file_path)
            
            # Sample frames and extract content
            frame_contents = self._process_video_frames(file_path)
            
            # Extract audio and transcribe if Hugging Face API is available
            audio_text = ""
            if self.use_huggingface:
                audio_text = self._extract_and_transcribe_audio(file_path)
            
            # Classify video content if Hugging Face API is available
            video_classification = ""
            if self.use_huggingface:
                video_classification = self._classify_video_content(file_path)
            
            # Combine all information
            all_text = metadata + "\n\n"
            
            if video_classification:
                all_text += "Video Classification:\n" + video_classification + "\n\n"
                
            all_text += "Visual Content Analysis:\n" + "\n\n".join(frame_contents) + "\n\n"
            
            if audio_text:
                all_text += "Audio Transcription:\n" + audio_text
            
            # Split into chunks
            if len(all_text) > self.chunk_size:
                chunks = self._chunk_text(all_text)
            else:
                chunks = [all_text]
            
            return chunks
        
        except Exception as e:
            raise Exception(f"Error processing video: {str(e)}")
    
    def _extract_video_metadata(self, file_path: str) -> str:
        """
        Extract metadata from a video file
        
        Args:
            file_path (str): Path to the video file
            
        Returns:
            str: Text describing the video metadata
        """
        # Open the video file
        video = cv2.VideoCapture(file_path)
        
        if not video.isOpened():
            return "Error: Could not open video file."
        
        # Get basic video properties
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Get file information
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        
        # Release the video
        video.release()
        
        # Format metadata
        metadata = (
            f"Video Metadata:\n"
            f"Filename: {file_name}\n"
            f"Resolution: {width}x{height}\n"
            f"FPS: {fps:.2f}\n"
            f"Duration: {duration:.2f} seconds\n"
            f"Frame Count: {frame_count}\n"
            f"File Size: {file_size:.2f} MB"
        )
        
        return metadata
    
    def _process_video_frames(self, file_path: str) -> List[str]:
        """
        Process frames from a video file
        
        Args:
            file_path (str): Path to the video file
            
        Returns:
            List[str]: List of text descriptions for each processed frame
        """
        # Open the video file
        video = cv2.VideoCapture(file_path)
        
        if not video.isOpened():
            return ["Error: Could not open video file."]
        
        frame_contents = []
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process frames at regular intervals
        for frame_idx in range(0, frame_count, self.frame_interval):
            # Set video position
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # Read frame
            success, frame = video.read()
            
            if not success:
                continue
            
            # Process frame
            frame_desc = self._analyze_frame(frame, frame_idx)
            frame_contents.append(frame_desc)
        
        # Release the video
        video.release()
        
        return frame_contents
    
    def _analyze_frame(self, frame, frame_idx: int) -> str:
        """
        Analyze a video frame
        
        Args:
            frame: OpenCV image frame
            frame_idx (int): Frame index
            
        Returns:
            str: Description of the frame
        """
        try:
            # Basic frame description that will be available even if analysis fails
            frame_desc = f"Frame {frame_idx}:\n"
            
            try:
                # Convert frame to PIL Image for OCR
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Extract text using OCR
                extracted_text = pytesseract.image_to_string(pil_image).strip()
                
                if extracted_text:
                    frame_desc += f"- Text detected: {extracted_text}\n"
                else:
                    frame_desc += "- No text detected\n"
            except Exception:
                frame_desc += "- Text analysis unavailable\n"
            
            try:
                # Basic frame analysis
                frame_brightness = np.mean(frame)
                frame_desc += f"- Brightness: {frame_brightness:.1f}/255\n"
                
                # Convert BGR to HSV for color analysis
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Calculate color distribution
                hue_values = hsv_frame[:, :, 0]
                saturation_values = hsv_frame[:, :, 1]
                avg_hue = np.mean(hue_values)
                avg_saturation = np.mean(saturation_values)
                
                # Determine dominant color
                if avg_saturation < 50:
                    if frame_brightness < 50:
                        dominant_color = "dark/black"
                    elif frame_brightness > 200:
                        dominant_color = "white"
                    else:
                        dominant_color = "gray"
                else:
                    if 0 <= avg_hue < 30 or 150 <= avg_hue <= 180:
                        dominant_color = "red"
                    elif 30 <= avg_hue < 90:
                        dominant_color = "green/yellow"
                    elif 90 <= avg_hue < 150:
                        dominant_color = "blue"
                    else:
                        dominant_color = "unknown"
                
                frame_desc += f"- Dominant color: {dominant_color}\n"
            except Exception:
                frame_desc += "- Color analysis unavailable\n"
            
            try:
                # Detect edges for object estimation
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                edge_count = np.count_nonzero(edges)
                
                # Estimate scene complexity
                complexity = "high" if edge_count > (frame.shape[0] * frame.shape[1] * 0.1) else "medium" if edge_count > (frame.shape[0] * frame.shape[1] * 0.05) else "low"
                
                frame_desc += f"- Scene complexity: {complexity}\n"
            except Exception:
                frame_desc += "- Scene complexity analysis unavailable\n"
            
            return frame_desc
        
        except Exception as e:
            # Fallback if all else fails
            return f"Frame {frame_idx}: Unable to analyze (Error: {str(e)})"
    
    def _extract_and_transcribe_audio(self, video_path: str) -> str:
        """
        Extract audio from video and transcribe it using Hugging Face ASR model
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            str: Transcribed audio text
        """
        try:
            if not self.use_huggingface or not self.hf_api_key:
                return "Audio transcription unavailable (Hugging Face API not configured)"
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract audio to temporary file
                audio_path = os.path.join(temp_dir, "audio.wav")
                self._extract_audio_from_video(video_path, audio_path)
                
                # Check if audio extraction was successful
                if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                    return "Audio extraction failed or video contains no audio"
                
                # Read the audio file
                with open(audio_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                
                # Send to Hugging Face API for transcription
                transcription = self._transcribe_audio(audio_bytes)
                
                return transcription
        except Exception as e:
            return f"Audio transcription failed: {str(e)}"
    
    def _extract_audio_from_video(self, video_path: str, output_audio_path: str) -> bool:
        """
        Extract audio track from video file using FFmpeg
        
        Args:
            video_path (str): Path to input video
            output_audio_path (str): Path to output audio file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if ffmpeg is installed
            ffmpeg_path = "/usr/bin/ffmpeg"
            if not os.path.exists(ffmpeg_path):
                return False
                
            # Construct FFmpeg command to extract audio
            command = [
                ffmpeg_path,
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit little-endian
                "-ar", "16000",  # 16kHz sample rate (common for speech recognition)
                "-ac", "1",  # Mono channel
                "-y",  # Overwrite output file if exists
                output_audio_path
            ]
            
            # Execute command and capture output
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            _, _ = process.communicate()
            
            # Check if output file was created
            return os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0
            
        except Exception:
            return False
    
    def _transcribe_audio(self, audio_bytes: bytes) -> str:
        """
        Transcribe audio using Hugging Face Speech-to-Text API
        
        Args:
            audio_bytes (bytes): Raw audio bytes
            
        Returns:
            str: Transcribed text
        """
        try:
            headers = {"Authorization": f"Bearer {self.hf_api_key}"}
            API_URL = f"https://api-inference.huggingface.co/models/{self.speech_recognition_model}"
            
            response = requests.post(API_URL, headers=headers, data=audio_bytes)
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract transcription from response
                if isinstance(result, dict) and "text" in result:
                    return result["text"]
                else:
                    return "Transcription format not recognized"
            else:
                return f"Transcription request failed with status {response.status_code}"
            
        except Exception as e:
            return f"Transcription error: {str(e)}"
    
    def _classify_video_content(self, video_path: str) -> str:
        """
        Classify video content using Hugging Face video classification model
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            str: Video classification results
        """
        try:
            if not self.use_huggingface or not self.hf_api_key:
                return ""
            
            # Sample a few frames from the video
            video = cv2.VideoCapture(video_path)
            
            if not video.isOpened():
                return "Video classification failed: Could not open video file"
            
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
            
            # Sample frames at regular intervals (try to get about 8 frames)
            sample_interval = max(1, frame_count // 8)
            frames = []
            
            for frame_idx in range(0, frame_count, sample_interval):
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = video.read()
                
                if success:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(rgb_frame)
                
                if len(frames) >= 8:
                    break
            
            video.release()
            
            if not frames:
                return "No frames could be extracted for classification"
            
            # Create a temporary file to store frames
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                # Save the middle frame for classification
                middle_frame = frames[len(frames)//2]
                pil_image = Image.fromarray(middle_frame)
                pil_image.save(temp_file.name)
                temp_file_path = temp_file.name
            
            # Read the image file
            with open(temp_file_path, "rb") as img_file:
                image_bytes = img_file.read()
            
            # Remove temporary file
            os.unlink(temp_file_path)
            
            # Send to Hugging Face API for classification
            headers = {"Authorization": f"Bearer {self.hf_api_key}"}
            API_URL = f"https://api-inference.huggingface.co/models/{self.video_classification_model}"
            
            response = requests.post(API_URL, headers=headers, data=image_bytes)
            
            if response.status_code == 200:
                results = response.json()
                
                # Format classification results
                classifications = ""
                if isinstance(results, list):
                    for result in results[:5]:  # Top 5 classifications
                        label = result.get("label", "Unknown")
                        score = result.get("score", 0) * 100
                        classifications += f"- {label}: {score:.1f}%\n"
                
                return classifications
            else:
                return f"Classification request failed with status {response.status_code}"
            
        except Exception as e:
            return f"Video classification error: {str(e)}"
    
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
