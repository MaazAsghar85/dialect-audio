"""
Streaming TTS with Piper
Queue-based architecture for real-time speech generation
"""

import queue
import threading
import wave
import io
import os
import logging
from typing import Optional
try:
    from piper import PiperVoice
    import sounddevice as sd
    import numpy as np
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False

logger = logging.getLogger(__name__)


class StreamingTTS:
    """
    Streaming TTS engine with queue-based processing
    Supports chunked text input for low-latency speech
    """
    
    def __init__(self, language="en_US", voice="lessac", quality="medium"):
        """
        Initialize streaming TTS
        
        Args:
            language: Language code (en_US, hi_IN, ta_IN, etc.)
            voice: Voice name
            quality: Voice quality (low, medium, high)
        """
        self.language = language
        self.voice_name = voice
        self.quality = quality
        self.sample_rate = 22050  # Piper default
        
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.worker_thread = None
        self.playback_thread = None
        
        # Try to load Piper voice
        self.piper_voice = None
        if PIPER_AVAILABLE:
            try:
                model_name = f"{language}-{voice}-{quality}"
                model_path = os.path.join("piper_models", model_name, f"{model_name}.onnx")
                config_path = os.path.join("piper_models", model_name, f"{model_name}.onnx.json")
                
                if os.path.exists(model_path) and os.path.exists(config_path):
                    logger.info(f"Loading Piper voice: {model_name}")
                    self.piper_voice = PiperVoice.load(model_path, config_path)
                    self.piper_available = True
                    logger.info(f"Piper voice loaded successfully!")
                else:
                    logger.warning(f"Piper model not found at {model_path}")
                    logger.info("Will use fallback TTS for now")
                    self.piper_available = False
            except Exception as e:
                logger.warning(f"Piper loading failed: {e}")
                logger.info("Will use fallback TTS for now")
                self.piper_available = False
        else:
            logger.warning("Piper not installed. Using fallback TTS.")
            self.piper_available = False
    
    def start(self):
        """Start the streaming TTS workers"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Worker thread: text → audio generation
        self.worker_thread = threading.Thread(target=self._generation_worker, daemon=True)
        self.worker_thread.start()
        
        # Playback thread: audio → speaker
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()
        
        logger.info("Streaming TTS started")
    
    def stop(self):
        """Stop the streaming TTS workers"""
        self.is_running = False
        
        # Clear queues
        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Streaming TTS stopped")
    
    def speak_chunk(self, text: str):
        """
        Add text chunk to processing queue
        
        Args:
            text: Text to speak (sentence or phrase)
        """
        if text.strip():
            self.text_queue.put(text.strip())
    
    def speak_complete(self, text: str):
        """
        Speak complete text, automatically chunked by sentences
        
        Args:
            text: Full text to speak
        """
        # Split by sentence boundaries
        chunks = self._chunk_text(text)
        for chunk in chunks:
            self.speak_chunk(chunk)
    
    def wait_until_done(self, timeout: Optional[float] = None):
        """Wait until all queued speech is completed"""
        # Wait for text queue to empty
        self.text_queue.join()
        
        # Wait for audio queue to empty
        self.audio_queue.join()
    
    def _chunk_text(self, text: str) -> list:
        """
        Split text into speakable chunks
        
        Args:
            text: Full text
            
        Returns:
            List of text chunks
        """
        import re
        
        # Split by sentence endings
        sentences = re.split(r'([.!?]+)', text)
        
        chunks = []
        current_chunk = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i].strip()
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            
            if sentence:
                chunks.append(sentence + punctuation)
        
        return chunks
    
    def _generation_worker(self):
        """Worker thread: Generate audio from text chunks"""
        while self.is_running:
            try:
                # Get text chunk with timeout
                text = self.text_queue.get(timeout=0.1)
                
                # Generate audio
                audio_data = self._generate_audio(text)
                
                # Add to playback queue
                if audio_data is not None:
                    self.audio_queue.put(audio_data)
                
                self.text_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS generation error: {e}")
                self.text_queue.task_done()
    
    def _playback_worker(self):
        """Worker thread: Play generated audio"""
        while self.is_running:
            try:
                # Get audio chunk with timeout
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Play audio
                self._play_audio(audio_data)
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio playback error: {e}")
                self.audio_queue.task_done()
    
    def _generate_audio(self, text: str) -> Optional[np.ndarray]:
        """
        Generate audio from text using Piper (or fallback)
        
        Args:
            text: Text to convert
            
        Returns:
            Audio data as numpy array
        """
        try:
            if self.piper_available and self.piper_voice:
                # Use Piper TTS (fast)
                logger.info(f"Generating audio with Piper: {text[:50]}...")
                
                # Generate audio using Piper (returns generator of AudioChunk objects)
                audio_generator = self.piper_voice.synthesize(text)
                
                # Collect all audio bytes from the chunks
                audio_chunks = []
                for audio_chunk in audio_generator:
                    # Extract audio bytes from Piper's AudioChunk object
                    # AudioChunk has: audio_int16_bytes, audio_int16_array, audio_float_array
                    chunk_data = audio_chunk.audio_int16_bytes
                    audio_chunks.append(chunk_data)
                
                # Combine all chunks into one bytes object
                audio_bytes = b''.join(audio_chunks)
                
                # Convert to numpy array for playback
                import numpy as np
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                
                # Convert to float32 and normalize
                audio_float = audio_array.astype(np.float32) / 32768.0
                
                # Add small silence padding to prevent cut-off
                silence_duration = 0.1  # 100ms silence
                silence_samples = int(self.sample_rate * silence_duration)
                silence = np.zeros(silence_samples, dtype=np.float32)
                audio_with_padding = np.concatenate([audio_float, silence])
                
                logger.info(f"Piper generated {len(audio_float)} samples (+ {len(silence)} padding)")
                return audio_with_padding
                
            else:
                # Fallback to pyttsx3 for now
                return self._generate_audio_fallback(text)
                
        except Exception as e:
            logger.warning(f"Piper generation failed: {e}, using fallback")
            return self._generate_audio_fallback(text)
    
    def _generate_audio_fallback(self, text: str) -> Optional[np.ndarray]:
        """Fallback TTS using pyttsx3 (blocking)"""
        try:
            import pyttsx3
            
            # This is temporary - not truly streaming
            # Just to keep system working while Piper is being set up
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            engine.say(text)
            engine.runAndWait()
            del engine
            
            return None  # pyttsx3 plays directly
            
        except Exception as e:
            logger.error(f"Fallback TTS failed: {e}")
            return None
    
    def _play_audio(self, audio_data: np.ndarray):
        """
        Play audio data through speakers
        
        Args:
            audio_data: Audio samples
        """
        try:
            if audio_data is not None:
                sd.play(audio_data, self.sample_rate)
                sd.wait()  # Wait for playback to complete
                
                # Add small delay to ensure audio buffer is fully flushed
                import time
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"Audio playback error: {e}")


# Quick setup helper
def create_streaming_tts(language="en_US") -> StreamingTTS:
    """
    Create and start streaming TTS instance
    
    Args:
        language: Language code
        
    Returns:
        Initialized StreamingTTS instance
    """
    tts = StreamingTTS(language=language)
    tts.start()
    return tts

