"""
Download Piper TTS Voice Models
Automatically downloads and extracts voice models for streaming TTS
"""

import os
import requests
import tarfile
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_piper_voice(language="en_US", voice="lessac", quality="medium"):
    """
    Download and extract Piper voice model
    
    Args:
        language: Language code (en_US, hi_IN, ta_IN, etc.)
        voice: Voice name (lessac, ljspeech, etc.)
        quality: Quality level (low, medium, high)
    """
    
    # Voice model URLs (these are the actual working URLs)
    voice_models = {
        "en_US-lessac-medium": {
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
            "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
        },
        "en_US-lessac-high": {
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/high/en_US-lessac-high.onnx",
            "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/high/en_US-lessac-high.onnx.json"
        }
    }
    
    model_name = f"{language}-{voice}-{quality}"
    
    if model_name not in voice_models:
        logger.error(f"Voice model {model_name} not available")
        logger.info("Available models:")
        for model in voice_models.keys():
            logger.info(f"  - {model}")
        return False
    
    # Create models directory
    models_dir = "piper_models"
    os.makedirs(models_dir, exist_ok=True)
    
    model_dir = os.path.join(models_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # Download model file
        logger.info(f"Downloading {model_name} model...")
        model_url = voice_models[model_name]["url"]
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        model_path = os.path.join(model_dir, f"{model_name}.onnx")
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Model downloaded: {model_path}")
        
        # Download config file
        logger.info(f"Downloading {model_name} config...")
        config_url = voice_models[model_name]["config_url"]
        response = requests.get(config_url)
        response.raise_for_status()
        
        config_path = os.path.join(model_dir, f"{model_name}.onnx.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        logger.info(f"Config downloaded: {config_path}")
        
        logger.info(f"✅ Voice model {model_name} ready!")
        logger.info(f"Model directory: {os.path.abspath(model_dir)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False

def download_english_voice():
    """Download English voice model (recommended for testing)"""
    return download_piper_voice("en_US", "lessac", "medium")

def download_hindi_voice():
    """Download Hindi voice model (for future use)"""
    # Note: Hindi models may have different naming
    logger.info("Hindi voice models may need different URLs")
    return False

if __name__ == "__main__":
    print("="*50)
    print("Piper TTS Voice Model Downloader")
    print("="*50)
    
    # Download English voice
    success = download_english_voice()
    
    if success:
        print("\n✅ Download complete!")
        print("You can now use Piper TTS in your streaming system.")
        print("\nTo use in main.py:")
        print("1. Update streaming_tts.py with model path")
        print("2. Test with: python main.py")
    else:
        print("\n❌ Download failed!")
        print("Check your internet connection and try again.")
