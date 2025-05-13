from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import shutil
import gdown
import subprocess
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer
import torch
import librosa
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Whisper Transcription API")

# Global variables
model = None
feature_extractor = None
tokenizer = None
model_loaded = False
model_error = None
MODEL_DIR = "./whisper-tiny_Akan_non_standardspeech"
MODEL_DOWNLOAD_FLAG = os.path.join(MODEL_DIR, ".download_complete")


def download_model():
    """Download the model from Google Drive if it doesn't exist locally"""
    try:
        # If directory exists but is incomplete, remove it
        if os.path.exists(MODEL_DIR) and not os.path.exists(MODEL_DOWNLOAD_FLAG):
            logger.info("Incomplete model directory found. Removing it...")
            shutil.rmtree(MODEL_DIR)

        # Check if model already exists and is complete
        if os.path.exists(MODEL_DIR) and os.path.exists(MODEL_DOWNLOAD_FLAG):
            logger.info("Model already downloaded and verified. Skipping download.")
            return True

        logger.info("Model not found locally or incomplete. Downloading from Google Drive...")
        
        # Create output directory
        os.makedirs("temp_download", exist_ok=True)
        
        # Google Drive folder ID
        folder_id = "19qVaP1cOrextWlVq9eyLkwx3b35o3gbh"
        
        # Download the entire folder
        gdown.download_folder(id=folder_id, output="temp_download", quiet=False)
        
        # Move files to the correct location
        if os.path.exists("temp_download/whisper-tiny_Akan_non_standardspeech"):
            # Create target directory if it doesn't exist
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            # Move all files from the downloaded directory to MODEL_DIR
            src_dir = "temp_download/whisper-tiny_Akan_non_standardspeech"
            for item in os.listdir(src_dir):
                src_path = os.path.join(src_dir, item)
                dst_path = os.path.join(MODEL_DIR, item)
                if os.path.isdir(src_path):
                    if os.path.exists(dst_path):
                        shutil.rmtree(dst_path)
                    shutil.copytree(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
        else:
            # If the folder structure is different, try to find and move the model files
            model_files_found = False
            for root, dirs, files in os.walk("temp_download"):
                if "model.safetensors" in files or "pytorch_model.bin" in files:
                    os.makedirs(MODEL_DIR, exist_ok=True)
                    for file in files:
                        shutil.copy2(os.path.join(root, file), os.path.join(MODEL_DIR, file))
                    model_files_found = True
                    break
            
            if not model_files_found:
                logger.error("Could not find model files in the downloaded content")
                return False
        
        # Clean up temporary directory
        shutil.rmtree("temp_download", ignore_errors=True)
        
        # Create a flag file to indicate successful download
        with open(MODEL_DOWNLOAD_FLAG, 'w') as f:
            f.write("Model download completed successfully")
        
        logger.info("Model download completed and verified.")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return False


@app.on_event("startup")
async def load_model():
    global model, feature_extractor, tokenizer, model_loaded, model_error

    try:
        # First, ensure the model is downloaded
        if not download_model():
            model_error = "Failed to download model"
            return
        
        # Verify that required model files exist
        if not os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")) and \
           not os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")):
            model_error = "Model files not found after download"
            return

        # Load the model
        logger.info(f"Loading model from {MODEL_DIR}...")
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)
        feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
        model.config.forced_decoder_ids = None
        model_loaded = True
        logger.info("Model loaded successfully")

    except Exception as e:
        model_error = str(e)
        logger.error(f"Error loading model: {model_error}")


@app.get("/")
async def root():
    if model_loaded:
        return {"status": "ready", "message": "Whisper transcription API is running"}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": f"Model loading failed: {model_error}"}
        )


@app.get("/health")
async def health_check():
    if model_loaded:
        return {"status": "healthy"}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": model_error}
        )


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not model_loaded:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {model_error}")

    if not file.filename.endswith((".wav", ".mp3")):
        raise HTTPException(status_code=400, detail="Only .wav and .mp3 files are supported.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        audio, sr = librosa.load(tmp_path, sr=16000)
        os.unlink(tmp_path)  # Clean up the temp file

        inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
        inputs["attention_mask"] = torch.ones(inputs["input_features"].shape[:-1], dtype=torch.long)

        with torch.no_grad():
            predicted_ids = model.generate(**inputs)

        transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Remove consecutive duplicate words
        words = transcription.strip().split()
        deduped_words = [words[0]] if words else []
        for word in words[1:]:
            if word != deduped_words[-1]:
                deduped_words.append(word)
        cleaned_transcription = " ".join(deduped_words)

        return {"transcription": cleaned_transcription}

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9000))
    uvicorn.run("whisper_fastapi_app:app", host="0.0.0.0", port=port, reload=True)
