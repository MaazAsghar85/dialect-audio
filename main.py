"""
Simple Speech-to-Intent Demo
Hypothesis 3: Audio -> Text + Intent (Multitask Learning)
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
import sounddevice as sd
import numpy as np
import pyttsx3
import os
import warnings
warnings.filterwarnings('ignore')


class MultitaskSpeechModel(nn.Module):
    """Audio -> Text + Intent"""
    
    def __init__(self, num_intents):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        hidden_size = self.wav2vec2.config.hidden_size
        
        # Intent classifier
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_intents)
        )
    
    def forward(self, input_values):
        # Get features for intent
        outputs = self.wav2vec2(input_values)
        pooled = outputs.last_hidden_state.mean(dim=1)
        intent_logits = self.intent_classifier(pooled)
        
        # Get ASR output
        asr_logits = self.asr_model(input_values).logits
        
        return intent_logits, asr_logits


class SpeechDemo:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 16000
        
        print(f"Using device: {self.device}")
        
        # Load processor
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        
        # Define intents and responses
        self.intents = [
            "call_nurse",
            "need_water",
            "feeling_pain",
            "need_doctor",
            "bathroom_assistance",
            "adjust_bed",
            "turn_off_light",
            "emergency_help"
        ]
        
        self.responses = {
            "call_nurse": "Nurse is coming",
            "need_water": "Someone will get you water",
            "feeling_pain": "Nurse will help with pain",
            "need_doctor": "Doctor is coming",
            "bathroom_assistance": "Someone is coming to help you",
            "adjust_bed": "Someone is coming to adjust your bed",
            "turn_off_light": "Someone is coming to turn off the light",
            "emergency_help": "Emergency help is coming immediately"
        }
        
        # Load model
        self.model = MultitaskSpeechModel(num_intents=len(self.intents)).to(self.device)
        
        # Load trained weights if available
        if os.path.exists("trained_intent_model.pt"):
            print("Loading trained model...")
            checkpoint = torch.load("trained_intent_model.pt", map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Trained model loaded!")
        else:
            print("[Note] No trained model found - using untrained model")
        
        # Initialize offline TTS
        print("Initializing offline TTS...")
        try:
            # Test TTS initialization
            test_engine = pyttsx3.init()
            test_engine.setProperty('rate', 150)
            test_engine.setProperty('volume', 0.9)
            print("Offline TTS ready!")
        except Exception as e:
            print(f"TTS initialization failed: {e}")
            print("TTS will be initialized per-use")
        
        self.model.eval()
        print("System ready!\n")
    
    def record_audio(self, duration=3):
        """Record from microphone"""
        print(f"\nSpeak now... ({duration} seconds)")
        audio = sd.rec(int(duration * self.sample_rate), 
                      samplerate=self.sample_rate, 
                      channels=1, 
                      dtype='float32')
        sd.wait()
        print("Recording complete")
        return audio.flatten()
    
    def transcribe_audio(self, audio):
        """Get text from audio"""
        inputs = self.processor(audio, 
                               sampling_rate=self.sample_rate, 
                               return_tensors="pt")
        input_values = inputs.input_values.to(self.device)
        
        with torch.no_grad():
            logits = self.model.asr_model(input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription
    
    def predict_intent(self, audio):
        """Get intent from audio"""
        inputs = self.processor(audio, 
                               sampling_rate=self.sample_rate, 
                               return_tensors="pt")
        input_values = inputs.input_values.to(self.device)
        
        with torch.no_grad():
            intent_logits, _ = self.model(input_values)
            pred_idx = torch.argmax(intent_logits, dim=-1).item()
        
        return self.intents[pred_idx]
    
    def speak_response(self, text):
        """Speak text using offline TTS"""
        try:
            print(f"[OFFLINE TTS] Speaking: {text}")
            # Reinitialize engine for each use to avoid issues
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            engine.say(text)
            engine.runAndWait()
            print("[OFFLINE TTS] Speech completed")
        except Exception as e:
            print(f"[TTS Error] {e}")
            print(f"[FALLBACK] Text response: {text}")
    
    def run(self):
        """Main demo loop"""
        print("="*50)
        print("Speech-to-Intent Demo")
        print("Hypothesis 3: Audio -> Text + Intent")
        print("="*50)
        print("\nAvailable intents:")
        for i, intent in enumerate(self.intents, 1):
            print(f"{i}. {intent}")
        
        if not os.path.exists("trained_intent_model.pt"):
            print("\n[Note] Model is untrained - predictions are random")
            print("[Tip] Run train_synthetic.py to train the model")
        print("="*50)
        
        while True:
            choice = input("\nPress ENTER to record (q to quit): ").strip()
            if choice.lower() == 'q':
                break
            
            try:
                # Record audio
                audio = self.record_audio(duration=3)
                
                # Get transcription
                print("\nProcessing...")
                text = self.transcribe_audio(audio)
                print(f"\nTranscription: {text}")
                
                # Get intent
                intent = self.predict_intent(audio)
                print(f"Intent: {intent}")
                
                # Speak response
                response = self.responses[intent]
                print(f"Response: {response}")
                print("\nSpeaking response...")
                self.speak_response(response)
                
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nDemo ended")


if __name__ == "__main__":
    demo = SpeechDemo()
    demo.run()