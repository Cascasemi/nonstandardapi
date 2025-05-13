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

app = FastAPI(title="Whisper Transcription API")

# Global variables
model = None
feature_extractor = None
tokenizer = None
model_loaded = False
model_error = None


@app.on_event("startup")
async def load_model():
    global model, feature_extractor, tokenizer, model_loaded, model_error

    try:
        model_dir = "./whisper-tiny_Akan_non_standardspeech"

        # Check if model directory already exists
        if not os.path.exists(model_dir) or not os.path.exists(os.path.join(model_dir, "model.safetensors")):
            print("Model not found locally. Downloading from Google Drive...")

            # If directory exists but is incomplete, remove it
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)

            # Google Drive folder ID - extract this from your Google Drive link
            # If your link is https://drive.google.com/drive/folders/1AbCdEfG-HIjkLmNoPqRsTuVwXyZ
            # Then the folder_id is "1AbCdEfG-HIjkLmNoPqRsTuVwXyZ"
            folder_id = "19qVaP1cOrextWlVq9eyLkwx3b35o3gbh"

            # Create output directory
            os.makedirs("temp_download", exist_ok=True)

            # Download the entire folder
            gdown.download_folder(id=folder_id, output="temp_download", quiet=False)

            # Move files to the correct location
            if os.path.exists("temp_download/whisper-tiny_Akan_non_standardspeech"):
                shutil.move("temp_download/whisper-tiny_Akan_non_standardspeech", "./")
            else:
                # If the folder structure is different, try to find and move the model files
                for root, dirs, files in os.walk("temp_download"):
                    if "model.safetensors" in files or "pytorch_model.bin" in files:
                        os.makedirs(model_dir, exist_ok=True)
                        for file in files:
                            shutil.move(os.path.join(root, file), os.path.join(model_dir, file))
                        break

            # Clean up temporary directory
            shutil.rmtree("temp_download", ignore_errors=True)

            print("Model download completed.")

        # Load the model
        print(f"Loading model from {model_dir}...")
        model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
        model.config.forced_decoder_ids = None
        model_loaded = True
        print("Model loaded successfully")

    except Exception as e:
        model_error = str(e)
        print(f"Error loading model: {model_error}")


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

    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
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
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9000))
    uvicorn.run("whisper_fastapi_app:app", host="0.0.0.0", port=port, reload=True)