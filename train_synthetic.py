"""
Generate synthetic training data using TTS and train intent model
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from gtts import gTTS
import tempfile
import os
from pydub import AudioSegment
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class MultitaskSpeechModel(nn.Module):
    """Audio -> Intent classifier"""
    
    def __init__(self, num_intents):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        hidden_size = self.wav2vec2.config.hidden_size
        
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_intents)
        )
    
    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        pooled = outputs.last_hidden_state.mean(dim=1)
        intent_logits = self.intent_classifier(pooled)
        return intent_logits


# Training examples for each intent
TRAINING_DATA = {
    "call_nurse": [
        "I need the nurse",
        "Can you call the nurse",
        "Please get the nurse",
        "I need help from a nurse",
        "Nurse please come here",
        "I want to see the nurse",
        "Get me a nurse",
        "I need nursing assistance"
    ],
    "need_water": [
        "I want water",
        "Can I have some water",
        "I need water please",
        "I am thirsty",
        "Get me water",
        "I would like to drink water",
        "Please bring water",
        "I need something to drink"
    ],
    "feeling_pain": [
        "I am in pain",
        "I feel pain",
        "It hurts",
        "I am feeling pain",
        "This is painful",
        "I need pain relief",
        "My pain is getting worse",
        "I am suffering from pain"
    ],
    "need_doctor": [
        "I need the doctor",
        "Can I see the doctor",
        "Please call the doctor",
        "I want to see a doctor",
        "Get the doctor please",
        "I need medical attention",
        "Doctor please come",
        "I need to speak with the doctor"
    ],
    "bathroom_assistance": [
        "I need to use the bathroom",
        "I want to go to the washroom",
        "Can you help me to the toilet",
        "I need bathroom help",
        "Take me to the bathroom please",
        "I need to use the restroom",
        "Help me go to the toilet",
        "I need toilet assistance"
    ],
    "adjust_bed": [
        "Adjust my bed",
        "Can you fix the bed",
        "I want the bed adjusted",
        "Please adjust my bed position",
        "Change the bed angle",
        "Make the bed more comfortable",
        "Raise the bed",
        "Lower the bed please"
    ],
    "turn_off_light": [
        "Turn off the light",
        "Switch off the lights",
        "I want the lights off",
        "Please turn the light off",
        "Lights off please",
        "Can you turn off the light",
        "Switch the lights off",
        "Make it dark please"
    ],
    "emergency_help": [
        "Emergency",
        "I need help now",
        "This is an emergency",
        "Help me please urgent",
        "Emergency assistance needed",
        "I need immediate help",
        "Urgent help required",
        "Emergency call for help"
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
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = MultitaskSpeechModel(num_intents=len(intent_list))
    model = model.to(device)
    print(f"Model moved to {device}")
    
    # Unfreeze last few layers of wav2vec2 for better adaptation
    for param in model.wav2vec2.parameters():
        param.requires_grad = False
    
    # Unfreeze last transformer layer
    for param in model.wav2vec2.encoder.layers[-1].parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.Adam([
        {'params': model.intent_classifier.parameters(), 'lr': 0.001},
        {'params': model.wav2vec2.encoder.layers[-1].parameters(), 'lr': 0.0001}
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
            
            # Forward
            logits = model(input_values)
            loss = criterion(logits, batch_labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=-1)
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
    
    # Train
    model, processor, intent_list = train_model(dataset, intent_list, epochs=20)
    
    # Save
    save_model(model, intent_list)
    
    print("\nTraining complete!")
    print(f"Trained on {len(dataset)} samples")
    print(f"Intents: {intent_list}")

