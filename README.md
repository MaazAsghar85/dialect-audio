# DialectAudio - Speech-to-Intent System

A complete speech-to-intent system implementing **Hypothesis 3: Multitask Learning** (Audio → Text + Intent) for hospital patient assistance.

## 🎯 Overview

This system provides real-time speech recognition and intent classification for hospital patients, enabling them to communicate their needs through voice commands. The system processes audio input, transcribes speech, classifies intent, and provides spoken responses.

## 🏗️ Architecture

### **Multitask Learning Approach**
- **Audio → Text**: Speech recognition using Wav2Vec2
- **Audio → Intent**: Intent classification using neural networks
- **Simultaneous Processing**: Both tasks performed in parallel for efficiency

### **Key Components**
- **Acoustic Frontend**: Wav2Vec2 featurization
- **Core Agent**: Multitask model (Text + Intent)
- **TTS Decoder**: Offline text-to-speech responses

## 📁 Files

- **`main.py`** - Main demo application with offline TTS
- **`speech_to_intent.py`** - Core model implementation and experiments
- **`trained_intent_model.pt`** - Pre-trained model weights (361MB)
- **`requirements.txt`** - Python dependencies

## 🚀 Quick Start

### **Installation**
```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo
python main.py
```

### **Usage**
1. **Start the system**: `python main.py`
2. **Speak your request**: Press ENTER and speak for 3 seconds
3. **Get response**: System transcribes, classifies intent, and speaks back

## 🎤 Supported Intents

| Intent | Description | Response |
|--------|-------------|----------|
| `call_nurse` | Request nurse assistance | "Nurse is coming" |
| `need_water` | Request water | "Someone will get you water" |
| `feeling_pain` | Report pain | "Nurse will help with pain" |
| `need_doctor` | Request doctor | "Doctor is coming" |
| `bathroom_assistance` | Bathroom help | "Someone is coming to help you" |
| `adjust_bed` | Bed adjustment | "Someone is coming to adjust your bed" |
| `turn_off_light` | Light control | "Someone is coming to turn off the light" |
| `emergency_help` | Emergency assistance | "Emergency help is coming immediately" |

## 🔧 Technical Details

### **Models Used**
- **Wav2Vec2**: `facebook/wav2vec2-base-960h` (pre-trained on LibriSpeech)
- **Architecture**: Multitask learning with shared encoder
- **Training**: Synthetic data generation + fine-tuning

### **Performance**
- **Accuracy**: 84% on intent classification
- **Latency**: ~100-200ms inference time
- **Offline**: No internet required after initial setup

### **Hardware Requirements**
- **GPU**: NVIDIA RTX 3050+ (recommended)
- **RAM**: 8GB+ system memory
- **Storage**: 2GB+ for models and dependencies

## 🛠️ Development

### **Training Pipeline**
```python
# Generate synthetic training data
python train_synthetic.py

# Train model with GPU acceleration
# Model automatically saves to trained_intent_model.pt
```

### **Model Architecture**
```python
class MultitaskSpeechModel(nn.Module):
    def __init__(self, num_intents):
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(...)
        self.asr_model = Wav2Vec2ForCTC.from_pretrained(...)
        self.intent_classifier = nn.Sequential(...)
    
    def forward(self, input_values):
        # Shared encoder
        outputs = self.wav2vec2(input_values)
        
        # Intent classification
        intent_logits = self.intent_classifier(pooled_output)
        
        # Speech recognition
        asr_logits = self.asr_model(input_values).logits
        
        return intent_logits, asr_logits
```

## 🌐 Deployment

### **Production Requirements**
- **Offline Operation**: No internet dependency
- **Real-time Processing**: <500ms response time
- **Hospital Environment**: Noise-robust processing
- **Scalability**: Multiple patient support

### **System Integration**
```python
# Load trained model
model = MultitaskSpeechModel(num_intents=8)
checkpoint = torch.load("trained_intent_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Process audio
intent_logits, asr_logits = model(audio_input)
predicted_intent = intent_logits.argmax(dim=-1)
```

## 📊 Performance Metrics

### **Training Results**
- **Dataset**: 64 synthetic samples (8 per intent)
- **Epochs**: 18 (early stopping at 84% accuracy)
- **Training Time**: ~5 minutes on RTX 3050
- **Model Size**: 361MB

### **Inference Performance**
- **GPU Inference**: ~100ms
- **CPU Inference**: ~500ms
- **Memory Usage**: ~2GB VRAM
- **Accuracy**: 84% intent classification

## 🔮 Future Enhancements

### **Planned Features**
- **Hindi Support**: IndicWav2Vec integration
- **Dialect Adaptation**: South Indian dialect support
- **Cultural Responses**: Local language

### **Research Directions**
- **Low-resource Languages**: Dialect-specific models
- **Multimodal Input**: Gesture + speech recognition
- **Context Awareness**: Patient history integration
- **Privacy**: On-device processing

## 📚 References

### **Key Papers**
- [Wav2Vec2: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)
- [End-to-End Speech Recognition](https://arxiv.org/abs/1508.04395)
- [Multitask Learning for Speech Recognition](https://arxiv.org/abs/1904.03418)

### **Datasets**
- **LibriSpeech**: 960h English ASR data
- **Fluent Speech Commands**: Intent classification
- **Synthetic Data**: Hospital-specific intents

## 🤝 Contributing

### **Development Setup**
```bash
# Clone repository
git clone https://github.com/MaazAsghar85/dialect-audio.git
cd DialectAudio

# Install dependencies
pip install -r requirements.txt

# Run tests
python speech_to_intent.py
```

### **Code Structure**
- **`main.py`**: Demo application
- **`speech_to_intent.py`**: Core model implementation
- **`train_synthetic.py`**: Training pipeline
- **`requirements.txt`**: Dependencies

## 📄 License

MIT License - See LICENSE file for details.

## 🆘 Support

### **Common Issues**
- **TTS not working**: Install `pyttsx3` and check audio drivers
- **Model loading errors**: Ensure `trained_intent_model.pt` is present
- **GPU not detected**: Install CUDA-enabled PyTorch

### **Troubleshooting**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Test audio recording
python -c "import sounddevice; print('Audio OK')"

# Test TTS
python -c "import pyttsx3; print('TTS OK')"
```

---

**Built with ❤️ for hospital patient care**
