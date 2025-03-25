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
import openai
from io import BytesIO
from typing import List, Dict, Any, Optional

class VideoProcessor:
    def __init__(self, frame_interval: int = 60, chunk_size: int = 1000, chunk_overlap: int = 200, max_frames: int = 10):
        """
        Initialize the Video processor
        
        Args:
            frame_interval (int): Number of frames to skip between processing
            chunk_size (int): Maximum size of each text chunk
            chunk_overlap (int): Number of characters to overlap between chunks
            max_frames (int): Maximum number of frames to process
        """
        self.frame_interval = frame_interval
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_frames = max_frames
        
        # Flags for API availability
        self.use_huggingface = True
        self.use_openai = True
        
        # Get API keys from environment variables
        try:
            self.hf_api_key = os.environ.get("HUGGINGFACE_API_KEY")
            if not self.hf_api_key:
                self.use_huggingface = False
                print("Warning: HUGGINGFACE_API_KEY not set, using OpenAI for video analysis")
        except Exception:
            self.use_huggingface = False
        
        try:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not self.openai_api_key:
                self.use_openai = False
                print("Warning: OPENAI_API_KEY not set, using Hugging Face for video analysis")
            else:
                # Initialize OpenAI client
                self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        except Exception as e:
            print(f"Error initializing OpenAI: {e}")
            self.use_openai = False
        
        # Define model names
        self.speech_recognition_model = "facebook/wav2vec2-large-960h-lv60-self"  # Hugging Face fallback
        self.video_classification_model = "MCG-NJU/videomae-base-finetuned-kinetics"  # Hugging Face fallback
        self.openai_vision_model = "gpt-4-vision-preview"  # For analyzing frames
        self.openai_whisper_model = "whisper-1"  # For audio transcription
    
    def process(self, file_path: str) -> List[str]:
        """
        Process a video file and extract content
        
        Args:
            file_path (str): Path to the video file
            
        Returns:
            List[str]: List of text chunks extracted from the video
        """
        # Fallback result in case of errors
        error_result = [f"Error processing video: {os.path.basename(file_path)}. Video analysis features might not be fully available."]
        
        try:
            # Check if file exists and is readable
            if not os.path.exists(file_path):
                print(f"Error: Video file does not exist: {file_path}")
                return error_result
                
            if not os.access(file_path, os.R_OK):
                print(f"Error: Video file is not readable: {file_path}")
                return error_result
            
            # Basic validation - try opening the file with OpenCV
            try:
                video = cv2.VideoCapture(file_path)
                if not video.isOpened():
                    print(f"Error: Could not open video file with OpenCV: {file_path}")
                    return error_result
                video.release()
                print(f"Successfully opened video: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Error opening video with OpenCV: {e}")
                return error_result
                
            # Process each component with robust error handling
            
            # 1. Metadata extraction
            try:
                print(f"Extracting metadata for {os.path.basename(file_path)}")
                metadata = self._extract_video_metadata(file_path)
                print(f"Metadata extraction successful")
            except Exception as e:
                print(f"Error extracting video metadata: {e}")
                metadata = f"Video: {os.path.basename(file_path)}"
            
            # 2. Frame content analysis
            try:
                print(f"Processing video frames for {os.path.basename(file_path)}")
                frame_contents = self._process_video_frames(file_path)
                print(f"Processed {len(frame_contents)} frames")
            except Exception as e:
                print(f"Error processing video frames: {e}")
                frame_contents = ["Frame analysis unavailable"]
            
            # 3. Audio transcription
            audio_text = ""
            try:
                print(f"Attempting audio extraction and transcription for {os.path.basename(file_path)}")
                if self.use_openai:  # Prefer OpenAI Whisper if available
                    audio_text = self._extract_and_transcribe_audio_with_whisper(file_path)
                elif self.use_huggingface:  # Fall back to Hugging Face
                    audio_text = self._extract_and_transcribe_audio(file_path)
                else:
                    audio_text = "Audio transcription unavailable (no API keys configured)"
                print(f"Audio transcription completed")
            except Exception as e:
                print(f"Error in audio extraction/transcription: {e}")
                audio_text = "Audio transcription unavailable"
            
            # 4. Video classification and content analysis
            video_classification = ""
            try:
                print(f"Attempting video classification for {os.path.basename(file_path)}")
                if self.use_openai:  # Prefer OpenAI Vision API if available
                    video_classification = self._classify_video_with_vision(file_path)
                elif self.use_huggingface:  # Fall back to Hugging Face
                    video_classification = self._classify_video_content(file_path)
                else:
                    video_classification = "Video classification unavailable (no API keys configured)"
                print(f"Video classification completed")
            except Exception as e:
                print(f"Error in video classification: {e}")
            
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
            
            print(f"Successfully processed video: {os.path.basename(file_path)}, generated {len(chunks)} chunks")
            return chunks
        
        except Exception as e:
            print(f"Unexpected error processing video: {e}")
            return error_result
    
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
        duration = frame_count / video.get(cv2.CAP_PROP_FPS) if video.get(cv2.CAP_PROP_FPS) > 0 else 0
        
        # Calculate optimal frame interval based on video duration
        # For longer videos, we'll sample fewer frames
        if duration > 60:  # If video is longer than 1 minute
            # Dynamically adjust frame interval to get enough samples
            optimal_interval = max(self.frame_interval, int(frame_count / self.max_frames))
        else:
            optimal_interval = self.frame_interval
            
        # Process frames at regular intervals, limited by max_frames
        frames_processed = 0
        for frame_idx in range(0, frame_count, optimal_interval):
            # Stop once we reach max_frames
            if frames_processed >= self.max_frames:
                break
                
            # Set video position
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # Read frame
            success, frame = video.read()
            
            if not success:
                continue
            
            # Process frame
            frame_desc = self._analyze_frame(frame, frame_idx)
            frame_contents.append(frame_desc)
            frames_processed += 1
            
            # Add progress indicator for long videos
            if frames_processed % 3 == 0:
                print(f"Processed {frames_processed}/{self.max_frames} frames")
        
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
            
            # Optimize OCR by only running on every 3rd frame to save processing time
            # Also, only run OCR if the frame appears to have text (high contrast areas)
            try:
                # Check if frame likely contains text by looking at edge detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges_for_ocr = cv2.Canny(gray, 100, 200)  # Renamed to avoid confusion
                edge_density = np.count_nonzero(edges_for_ocr) / (frame.shape[0] * frame.shape[1])
                
                # Only run OCR if the frame has a reasonable edge density
                # or if this is a key frame (first, middle, or last frame)
                if edge_density > 0.05 or frame_idx % (self.frame_interval * 3) == 0:
                    # Resize image to speed up OCR (smaller image is faster)
                    height, width = frame.shape[:2]
                    scale_factor = min(1.0, 800 / max(width, height))
                    if scale_factor < 1.0:
                        resized_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
                    else:
                        resized_frame = frame
                    
                    # Convert frame to PIL Image for OCR
                    pil_image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
                    
                    # Extract text using OCR
                    extracted_text = pytesseract.image_to_string(pil_image).strip()
                    
                    if extracted_text:
                        frame_desc += f"- Text detected: {extracted_text}\n"
                    else:
                        frame_desc += "- No text detected\n"
                else:
                    # Skip OCR for this frame
                    frame_desc += "- Text analysis skipped (optimization)\n"
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
                # Just compute the edges directly - simpler and more reliable
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges_for_complexity = cv2.Canny(gray, 100, 200)
                edge_count = np.count_nonzero(edges_for_complexity)
                
                # Estimate scene complexity
                total_pixels = frame.shape[0] * frame.shape[1]
                edge_ratio = edge_count / total_pixels
                
                if edge_ratio > 0.1:
                    complexity = "high"
                elif edge_ratio > 0.05:
                    complexity = "medium"
                else:
                    complexity = "low"
                
                frame_desc += f"- Scene complexity: {complexity}\n"
            except Exception as e:
                print(f"Scene complexity analysis error: {e}")
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
            
            # Get a single representative frame instead of sampling multiple frames
            video = cv2.VideoCapture(video_path)
            
            if not video.isOpened():
                return "Video classification failed: Could not open video file"
            
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Just get one frame from 1/3 into the video
            target_frame = min(frame_count - 1, frame_count // 3)
            video.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            success, frame = video.read()
            video.release()
            
            if not success:
                return "Could not extract frame for classification"
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create a temporary file to store the frame
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                # Resize image to reduce file size (speed up API call)
                height, width = rgb_frame.shape[:2]
                scale_factor = min(1.0, 800 / max(width, height))
                if scale_factor < 1.0:
                    resized_frame = cv2.resize(rgb_frame, (0, 0), fx=scale_factor, fy=scale_factor)
                else:
                    resized_frame = rgb_frame
                
                # Save the frame
                pil_image = Image.fromarray(resized_frame)
                pil_image.save(temp_file.name, quality=85)  # Lower quality for smaller file
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
        
    def _extract_and_transcribe_audio_with_whisper(self, video_path: str) -> str:
        """
        Extract audio from video and transcribe it using OpenAI's Whisper model
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            str: Transcribed audio text
        """
        try:
            if not self.use_openai:
                return "Audio transcription unavailable (OpenAI API not configured)"
            
            # Create temporary directory for audio extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract audio to temporary file
                audio_path = os.path.join(temp_dir, "audio.mp3")  # Using mp3 for Whisper
                
                # Extract audio using FFmpeg
                success = self._extract_audio_from_video(video_path, audio_path)
                
                if not success or not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                    return "Audio extraction failed or video contains no audio"
                
                # Send to OpenAI Whisper API for transcription
                with open(audio_path, "rb") as audio_file:
                    try:
                        print("Sending audio to OpenAI Whisper...")
                        response = self.openai_client.audio.transcriptions.create(
                            model=self.openai_whisper_model,
                            file=audio_file
                        )
                        # Extract transcription
                        if hasattr(response, 'text'):
                            return response.text
                        else:
                            return str(response)
                    except Exception as e:
                        print(f"OpenAI Whisper transcription error: {e}")
                        return f"Transcription error: {str(e)}"
        except Exception as e:
            print(f"Error in Whisper transcription pipeline: {e}")
            return f"Audio transcription failed: {str(e)}"
    
    def _classify_video_with_vision(self, video_path: str) -> str:
        """
        Analyze video content using OpenAI's Vision API with selected frames
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            str: Video content analysis results
        """
        try:
            if not self.use_openai:
                return "Video analysis unavailable (OpenAI API not configured)"
            
            # Extract a sample of frames from the video for analysis
            video = cv2.VideoCapture(video_path)
            
            if not video.isOpened():
                return "Error: Could not open video file."
            
            # Get video properties
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            
            # Calculate optimal frame interval to extract representative frames
            # We'll use fewer frames for the OpenAI API to control costs
            max_vision_frames = min(3, self.max_frames)  # Limit to 3 frames max for API cost reasons
            
            if duration > 60:  # Longer than 1 minute
                # Space frames evenly through video
                sample_points = [int(i * frame_count / max_vision_frames) for i in range(max_vision_frames)]
            else:
                # For shorter videos, take beginning, middle, and end if possible
                sample_points = [0]
                if frame_count > 10:  # Ensure there are enough frames
                    sample_points.append(frame_count // 2)
                if frame_count > 20:
                    sample_points.append(frame_count - 1)
            
            # Extract the frames
            frames = []
            for frame_idx in sample_points:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = video.read()
                if success:
                    # Convert frame to PNG format in memory
                    _, buffer = cv2.imencode('.png', frame)
                    image_bytes = buffer.tobytes()
                    
                    # Convert to base64 for API
                    base64_image = base64.b64encode(image_bytes).decode('utf-8')
                    frames.append({
                        "frame_idx": frame_idx,
                        "time": frame_idx / fps if fps > 0 else 0,
                        "base64": base64_image
                    })
            
            video.release()
            
            if not frames:
                return "Could not extract frames from video."
            
            # Process frames with Vision API
            results = []
            for frame_data in frames:
                try:
                    # Prepare the message content
                    content = [
                        {
                            "type": "text",
                            "text": (
                                f"This is frame {frame_data['frame_idx']} from a video, "
                                f"at approximately {frame_data['time']:.2f} seconds. "
                                f"Analyze this frame in detail and describe what you see: "
                                f"objects, actions, scene type, text visible, people, colors, and atmosphere. "
                                f"If there's any text visible in the image, transcribe it exactly."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{frame_data['base64']}"
                            }
                        }
                    ]
                    
                    # Call OpenAI Vision API
                    response = self.openai_client.chat.completions.create(
                        model=self.openai_vision_model,
                        messages=[
                            {"role": "system", "content": "You are a detailed video frame analyzer. Describe what you see in the frame with precision."},
                            {"role": "user", "content": content}
                        ],
                        max_tokens=1000
                    )
                    
                    # Extract and format the response
                    if hasattr(response, 'choices') and len(response.choices) > 0:
                        frame_analysis = response.choices[0].message.content
                        timestamp = f"{frame_data['time']:.2f}s"
                        results.append(f"Frame at {timestamp}:\n{frame_analysis}")
                    else:
                        results.append(f"Frame at {frame_data['time']:.2f}s: Analysis failed")
                except Exception as e:
                    print(f"Error analyzing frame with OpenAI Vision: {e}")
                    results.append(f"Frame at {frame_data['time']:.2f}s: Analysis error: {str(e)}")
            
            # Combine results into a detailed analysis
            if results:
                summary_prompt = (
                    "Based on the analyzed frames, provide an overall summary of this video. "
                    "What is the main content, style, and purpose of this video? "
                    "Is it professional, educational, entertainment, etc.? "
                    "What subjects or topics does it appear to cover?"
                )
                
                try:
                    # Create a summary using the GPT model
                    summary_response = self.openai_client.chat.completions.create(
                        model="gpt-4-turbo-preview",  # Using GPT-4 for better summarization
                        messages=[
                            {"role": "system", "content": "You summarize video content based on frame analysis."},
                            {"role": "user", "content": summary_prompt + "\n\n" + "\n\n".join(results)}
                        ],
                        max_tokens=500
                    )
                    
                    if hasattr(summary_response, 'choices') and len(summary_response.choices) > 0:
                        summary = summary_response.choices[0].message.content
                        final_result = f"Video Analysis Summary:\n{summary}\n\nDetailed Frame Analysis:\n" + "\n\n".join(results)
                    else:
                        final_result = "Frame Analysis:\n" + "\n\n".join(results)
                except Exception as e:
                    print(f"Error creating video summary: {e}")
                    final_result = "Frame Analysis:\n" + "\n\n".join(results)
                
                return final_result
            else:
                return "No frame analysis results available."
        except Exception as e:
            print(f"Error in OpenAI Vision analysis: {e}")
            return f"Video analysis failed: {str(e)}"
