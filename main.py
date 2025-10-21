"""
Hospital Reception Speech-to-Intent System
Optimized: Audio -> Text + Intent (Single Encoder Architecture)
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
import sounddevice as sd
import numpy as np
import os
import logging
import time
import warnings
warnings.filterwarnings('ignore')

# Import streaming TTS
from streaming_tts import StreamingTTS
from typing import Dict, Optional

# Optional LLM (Ollama)
try:
    import ollama  # pip install ollama
    _OLLAMA_AVAILABLE = True
except Exception:
    _OLLAMA_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MultitaskSpeechModel(nn.Module):
    """Audio -> Text + Intent (Optimized: Single encoder)"""
    
    def __init__(self, num_intents):
        super().__init__()
        # Single Wav2Vec2ForCTC model (includes encoder + ASR head)
        self.asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        hidden_size = self.asr_model.config.hidden_size
        
        # Intent classifier head (uses same encoder features)
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_intents)
        )
    
    def forward(self, input_values):
        # Single forward pass through shared encoder
        outputs = self.asr_model.wav2vec2(input_values)
        
        # Intent: pool features and classify
        pooled = outputs.last_hidden_state.mean(dim=1)
        intent_logits = self.intent_classifier(pooled)
        
        # ASR: use existing CTC head
        asr_logits = self.asr_model.lm_head(outputs.last_hidden_state)
        
        return intent_logits, asr_logits


class SpeechDemo:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 16000
        
        logger.info(f"Using device: {self.device}")
        
        # Load processor
        logger.info("Loading Wav2Vec2 processor...")
        start = time.time()
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        logger.info(f"Processor loaded in {time.time() - start:.3f}s")
        
        # Define intents and responses - HOSPITAL RECEPTION SYSTEM
        self.intents = [
            "book_appointment",
            "cancel_appointment",
            "get_test_results",
            "billing_inquiry",
            "emergency_admission",
            "visiting_hours",
            "find_department",
            "general_inquiry"
        ]
        
        self.responses = {
            "book_appointment": "I will help you book an appointment. Please proceed to counter 2.",
            "cancel_appointment": "I can help you cancel your appointment. Please provide your appointment ID.",
            "get_test_results": "Test reports are available at the lab counter. Please show your ID.",
            "billing_inquiry": "For billing information, please proceed to the billing counter on the first floor.",
            "emergency_admission": "Emergency team has been notified. Please proceed to the emergency ward immediately.",
            "visiting_hours": "Visiting hours are from 4 PM to 7 PM daily. Maximum two visitors per patient.",
            "find_department": "I can help you find the department. Please specify which department you need.",
            "general_inquiry": "How can I help you today? Please tell me what information you need."
        }
        
        # Load model
        logger.info("Loading MultitaskSpeechModel...")
        start = time.time()
        self.model = MultitaskSpeechModel(num_intents=len(self.intents)).to(self.device)
        logger.info(f"Model loaded in {time.time() - start:.3f}s")
        
        # Load trained weights if available
        if os.path.exists("trained_intent_model.pt"):
            logger.info("Loading trained model weights...")
            start = time.time()
            checkpoint = torch.load("trained_intent_model.pt", map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            logger.info(f"Trained model loaded in {time.time() - start:.3f}s")
        else:
            logger.warning("No trained model found - using untrained model")
        
        # Initialize Streaming TTS
        logger.info("Initializing Streaming TTS...")
        start = time.time()
        try:
            self.streaming_tts = StreamingTTS(language="en_US")
            self.streaming_tts.start()
            logger.info(f"Streaming TTS initialized in {time.time() - start:.3f}s")
            self.tts_available = True
        except Exception as e:
            logger.error(f"Streaming TTS initialization failed: {e}")
            self.streaming_tts = None
            self.tts_available = False
        
        # LLM usage flag
        self.use_llm = True and _OLLAMA_AVAILABLE
        if not _OLLAMA_AVAILABLE:
            logger.warning("Ollama client not available. Falling back to predefined responses.")
        
        self.model.eval()
        logger.info("System ready!\n")
    
    def record_audio(self, duration=3):
        """Record from microphone"""
        logger.info(f"Recording audio for {duration} seconds...")
        start = time.time()
        audio = sd.rec(int(duration * self.sample_rate), 
                      samplerate=self.sample_rate, 
                      channels=1, 
                      dtype='float32')
        sd.wait()
        elapsed = time.time() - start
        logger.info(f"Recording complete in {elapsed:.3f}s")
        return audio.flatten()
    
    def process_audio(self, audio):
        """Get BOTH text + intent in single forward pass (OPTIMIZED)"""
        start = time.time()
        
        # Preprocess once
        inputs = self.processor(audio, 
                               sampling_rate=self.sample_rate, 
                               return_tensors="pt")
        input_values = inputs.input_values.to(self.device)
        
        # Single forward pass returns both outputs
        with torch.no_grad():
            intent_logits, asr_logits = self.model(input_values)
        
        # Decode intent
        pred_idx = torch.argmax(intent_logits, dim=-1).item()
        intent = self.intents[pred_idx]
        
        # Decode transcription
        predicted_ids = torch.argmax(asr_logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        elapsed = time.time() - start
        logger.info(f"Audio processing (transcription + intent) completed in {elapsed:.3f}s")
        
        return transcription, intent
    
    def speak_response(self, text):
        """Speak text using streaming TTS (chunked for low latency)"""
        try:
            logger.info(f"TTS speaking (streaming): {text}")
            start = time.time()
            
            if self.tts_available and self.streaming_tts:
                # Use streaming TTS with automatic chunking
                self.streaming_tts.speak_complete(text)
                
                # Wait for all chunks to complete
                self.streaming_tts.wait_until_done()
            else:
                logger.warning(f"TTS unavailable. Text response: {text}")
            
            elapsed = time.time() - start
            logger.info(f"TTS completed in {elapsed:.3f}s")
        except Exception as e:
            logger.error(f"TTS Error: {e}")
            logger.warning(f"Fallback text response: {text}")

    def speak_response_llm(self, intent: str, slots: Optional[Dict[str, str]] = None):
        """Stream LLM response phrases to TTS as they are generated."""
        if not (_OLLAMA_AVAILABLE and self.use_llm):
            # Fallback to canned response
            self.speak_response(self.responses.get(intent, self.responses["general_inquiry"]))
            return
        
        system_prompt = (
            "You are a polite, concise hospital reception assistant. "
            "Respond in short phrases suitable for speech. "
            "Use simple English; avoid long sentences. If information is missing, ask one clear follow-up question."
        )
        user_prompt = (
            f"Intent: {intent}. "
            + (f"Slots: {slots}. " if slots else "")
            + "Generate an appropriate receptionist response."
        )
        
        logger.info("Generating response via LLM (streaming)...")
        start = time.time()
        try:
            client = ollama.Client()
            stream = client.chat(model="qwen2.5:3b", stream=True, messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
            
            # Accumulate tokens into phrases and speak per sentence
            buffer = ""
            first_chunk_sent = False
            sentence_delims = {'.', '!', '?'}
            for part in stream:
                delta = part.get("message", {}).get("content", "")
                if not delta:
                    continue
                buffer += delta
                # Emit on sentence boundary
                while any(d in buffer for d in sentence_delims):
                    # split on first delimiter to keep pacing
                    positions = [buffer.find(d) for d in sentence_delims]
                    positions = [p for p in positions if p != -1]
                    if not positions:
                        break
                    cut_idx = min(positions)
                    phrase = buffer[:cut_idx+1].strip()
                    buffer = buffer[cut_idx+1:].lstrip()
                    if phrase:
                        if self.tts_available and self.streaming_tts:
                            self.streaming_tts.speak_chunk(phrase)
                            if not first_chunk_sent:
                                logger.info(f">>> Time to first sound: {time.time() - start:.3f}s")
                                first_chunk_sent = True
                        else:
                            logger.info(f"[TTS disabled] {phrase}")
            # Flush remainder
            tail = buffer.strip()
            if tail:
                if self.tts_available and self.streaming_tts:
                    self.streaming_tts.speak_chunk(tail)
                else:
                    logger.info(f"[TTS disabled] {tail}")
            
            # Wait for audio to finish
            if self.tts_available and self.streaming_tts:
                self.streaming_tts.wait_until_done()
            logger.info(f"LLM streaming completed in {time.time() - start:.3f}s")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fallback
            self.speak_response(self.responses.get(intent, self.responses["general_inquiry"]))
    
    def run(self):
        """Main demo loop"""
        logger.info("="*50)
        logger.info("Hospital Reception Speech-to-Intent System")
        logger.info("Optimized Single-Pass Architecture")
        logger.info("="*50)
        logger.info("\nAvailable intents:")
        for i, intent in enumerate(self.intents, 1):
            logger.info(f"{i}. {intent}")
        
        if not os.path.exists("trained_intent_model.pt"):
            logger.warning("Model is untrained - predictions are random")
            logger.info("Tip: Run train_synthetic.py to train the model")
        logger.info("="*50)
        
        while True:
            choice = input("\nPress ENTER to record (q to quit): ").strip()
            if choice.lower() == 'q':
                break
            
            try:
                total_start = time.time()
                
                # Record audio
                audio = self.record_audio(duration=3)
                response_start = time.time()  # Track when processing starts
                
                # Process audio (OPTIMIZED: single pass for both transcription + intent)
                logger.info("Processing audio...")
                text, intent = self.process_audio(audio)
                logger.info(f"Transcription: {text}")
                logger.info(f"Intent: {intent}")
                
                if self.use_llm and _OLLAMA_AVAILABLE:
                    # Stream LLM to TTS (handles its own time-to-first-sound log)
                    self.speak_response_llm(intent)
                else:
                    # Prepare canned response
                    response = self.responses[intent]
                    logger.info(f"Response: {response}")
                    # Time to first sound (approx, uses processing end)
                    time_to_response = time.time() - response_start
                    logger.info(f">>> Time to first sound: {time_to_response:.3f}s")
                    self.speak_response(response)
                
                total_elapsed = time.time() - total_start
                logger.info(f"=== TOTAL TIME: {total_elapsed:.3f}s ===\n")
                
            except Exception as e:
                logger.error(f"Error: {e}")
        
        # Cleanup streaming TTS
        if hasattr(self, 'streaming_tts') and self.streaming_tts:
            self.streaming_tts.stop()
        
        logger.info("Demo ended")


if __name__ == "__main__":
    demo = SpeechDemo()
    demo.run()