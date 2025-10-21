"""
Generate synthetic training data using TTS and train intent model
OPTIMIZED: Single encoder architecture for hospital reception system
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from gtts import gTTS
import tempfile
import os
from pydub import AudioSegment
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class MultitaskSpeechModel(nn.Module):
    """Audio -> Intent classifier (OPTIMIZED: Single encoder)"""
    
    def __init__(self, num_intents):
        super().__init__()
        # Single Wav2Vec2ForCTC model (includes encoder + ASR head)
        self.asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        hidden_size = self.asr_model.config.hidden_size
        
        # Intent classifier head (uses encoder features)
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_intents)
        )
    
    def forward(self, input_values):
        # Single forward pass through encoder
        outputs = self.asr_model.wav2vec2(input_values)
        pooled = outputs.last_hidden_state.mean(dim=1)
        intent_logits = self.intent_classifier(pooled)
        
        # Also return ASR logits for compatibility with main.py
        asr_logits = self.asr_model.lm_head(outputs.last_hidden_state)
        
        return intent_logits, asr_logits


# Training examples for each intent - HOSPITAL RECEPTION SYSTEM
TRAINING_DATA = {
    "book_appointment": [
        "I want to book an appointment",
        "Can I schedule an appointment",
        "I need to see a doctor",
        "I would like to make an appointment",
        "Book an appointment please",
        "Schedule a doctor visit",
        "I need an appointment",
        "Can you fix an appointment for me"
    ],
    "cancel_appointment": [
        "I want to cancel my appointment",
        "Cancel my appointment please",
        "I need to cancel my booking",
        "Can you cancel my appointment",
        "I cannot come for my appointment",
        "Please cancel my scheduled visit",
        "I want to cancel my doctor visit",
        "Cancel my scheduled appointment"
    ],
    "get_test_results": [
        "I want to collect my reports",
        "Are my test results ready",
        "I need to get my test reports",
        "Can I collect my lab results",
        "I want my medical reports",
        "Are my reports ready",
        "I came to pick up my test results",
        "I need to get my blood test report"
    ],
    "billing_inquiry": [
        "I have a question about my bill",
        "How much do I need to pay",
        "I want to know about billing",
        "Can you tell me the charges",
        "What is my total bill",
        "I need billing information",
        "I want to pay my bill",
        "How much does the treatment cost"
    ],
    "emergency_admission": [
        "This is an emergency",
        "I need immediate admission",
        "Emergency patient needs help",
        "I need urgent medical attention",
        "This is a critical emergency",
        "Patient needs immediate care",
        "Emergency admission required",
        "I need help this is urgent"
    ],
    "visiting_hours": [
        "What are the visiting hours",
        "When can I visit the patient",
        "What time can I see my relative",
        "Tell me about visiting times",
        "When are visitors allowed",
        "Can I visit the patient now",
        "What are the visiting timings",
        "When can family members visit"
    ],
    "find_department": [
        "Where is the cardiology department",
        "I need directions to radiology",
        "How do I get to the X-ray department",
        "Where is the emergency ward",
        "I am looking for the pharmacy",
        "Can you tell me where the lab is",
        "I need to find the neurology department",
        "Where is the reception area"
    ],
    "general_inquiry": [
        "I have a general question",
        "Can you help me with information",
        "I need some information",
        "I want to know about hospital services",
        "Can you tell me about the facilities",
        "I need some help",
        "I have a question",
        "Can you provide me information"
    ]
}


def text_to_audio(text, output_path):
    """Convert text to audio using TTS"""
    tts = gTTS(text=text, lang='en', slow=False)
    temp_mp3 = output_path + ".mp3"
    tts.save(temp_mp3)
    
    # Convert to wav
    audio = AudioSegment.from_mp3(temp_mp3)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_path, format="wav")
    
    os.unlink(temp_mp3)
    return output_path


def load_audio(file_path):
    """Load audio file as numpy array"""
    audio = AudioSegment.from_wav(file_path)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples = samples / np.max(np.abs(samples))  # Normalize
    return samples


def generate_dataset():
    """Generate synthetic dataset"""
    print("Generating synthetic training data...")
    
    dataset = []
    intent_list = list(TRAINING_DATA.keys())
    
    for intent_idx, (intent, examples) in enumerate(TRAINING_DATA.items()):
        print(f"Generating {intent}... ({len(examples)} samples)")
        
        for i, text in enumerate(examples):
            try:
                # Generate audio
                temp_dir = tempfile.mkdtemp()
                audio_path = os.path.join(temp_dir, f"audio_{intent}_{i}.wav")
                text_to_audio(text, audio_path)
                
                # Load audio
                audio_array = load_audio(audio_path)
                
                dataset.append({
                    'audio': audio_array,
                    'intent': intent_idx,
                    'text': text
                })
                
                # Cleanup
                os.unlink(audio_path)
                os.rmdir(temp_dir)
                
            except Exception as e:
                print(f"Error generating {text}: {e}")
    
    print(f"Generated {len(dataset)} samples")
    return dataset, intent_list


def train_model(dataset, intent_list, epochs=50):
    """Train model on synthetic data"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining on {device}...")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Initialize
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    model = MultitaskSpeechModel(num_intents=len(intent_list))
    model = model.to(device)
    print(f"Model moved to {device}")
    
    # Unfreeze last few layers of encoder for better adaptation
    for param in model.asr_model.wav2vec2.parameters():
        param.requires_grad = False
    
    # Unfreeze last transformer layer
    for param in model.asr_model.wav2vec2.encoder.layers[-1].parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.Adam([
        {'params': model.intent_classifier.parameters(), 'lr': 0.001},
        {'params': model.asr_model.wav2vec2.encoder.layers[-1].parameters(), 'lr': 0.0001}
    ])
    criterion = nn.CrossEntropyLoss()
    
    # Training loop with better batching
    model.train()
    batch_size = 8
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        
        # Shuffle data
        import random
        random.shuffle(dataset)
        
        # Batch training
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            
            # Prepare batch
            batch_audio = [item['audio'] for item in batch]
            batch_labels = torch.tensor([item['intent'] for item in batch]).to(device)
            
            # Preprocess batch
            inputs = processor(batch_audio, 
                             sampling_rate=16000, 
                             return_tensors="pt", 
                             padding=True)
            input_values = inputs.input_values.to(device)
            
            # Forward (model returns intent_logits, asr_logits)
            intent_logits, _ = model(input_values)
            loss = criterion(intent_logits, batch_labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = intent_logits.argmax(dim=-1)
            correct += (pred == batch_labels).sum().item()
        
        accuracy = correct / len(dataset) * 100
        avg_loss = total_loss / (len(dataset) // batch_size)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Early stopping if accuracy is good
        if accuracy > 80:
            print(f"Good accuracy reached! Stopping early at epoch {epoch+1}")
            break
    
    return model, processor, intent_list


def save_model(model, intent_list, filename="trained_intent_model.pt"):
    """Save trained model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'intents': intent_list
    }, filename)
    print(f"\nModel saved to {filename}")


if __name__ == "__main__":
    print("="*50)
    print("Synthetic Data Training")
    print("="*50)
    
    # Generate dataset
    dataset, intent_list = generate_dataset()
    
    # Train (increased epochs for better accuracy)
    model, processor, intent_list = train_model(dataset, intent_list, epochs=50)
    
    # Save
    save_model(model, intent_list)
    
    print("\nTraining complete!")
    print(f"Trained on {len(dataset)} samples")
    print(f"Intents: {intent_list}")

