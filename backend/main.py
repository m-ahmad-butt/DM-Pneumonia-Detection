import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pathlib import Path
from pydantic import BaseModel

app = FastAPI(
    title="Pneumonia Detection API",
    description="API for detecting pneumonia from chest X-ray images using a trained MobileNetV2 model."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(SCRIPT_DIR.parent / 'model' / 'xray_mobilenet_model.keras')
IMG_SIZE = (128, 128)
DECISION_THRESHOLD = 0.70

print(f"[*] Loading model from {MODEL_PATH}...")
try:
    if os.path.exists(MODEL_PATH):
        MODEL = keras.models.load_model(MODEL_PATH)
        print("[+] Model loaded successfully!")
    else:
        print(f"[-] Model file not found at {MODEL_PATH}")
        MODEL = None
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[-] Error loading model: {e}")
    MODEL = None

class PredictionResult(BaseModel):
    filename: str
    prediction: str
    probability: float

def preprocess_image_bytes(image_bytes: bytes, target_size=IMG_SIZE):
    """
    Decode and preprocess the image bytes.
    Steps: Grayscale -> Bicubic Resize -> Median Blur -> CLAHE -> Z-score
    """
    # Decode from bytes to grayscale
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image (invalid format or corrupted)")

    # Cubic Resize  ->  128x128
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)

    # Median Blur     ->  kernel=3 (noise removal)
    img = cv2.medianBlur(img, 3)

    # CLAHE           ->  clipLimit=2.0, tileGridSize=(8,8) (contrast)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Z-score         ->  per-image mean=0, std=1
    img = img.astype(np.float32)
    mean = np.mean(img)
    std = np.std(img)
    if std == 0:
        std = 1
    img = (img - mean) / std

    # Adding channel dimension: (128, 128) -> (128, 128, 1)
    img = img[..., np.newaxis]
    return img

@app.post("/predict", response_model=List[PredictionResult])
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Takes a batch of images and predicts whether each is NORMAL or PNEUMONIA.
    """
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    results = []
    batch_images = []
    valid_filenames = []
    
    for file in files:
        try:
            image_bytes = await file.read()
            processed_img = preprocess_image_bytes(image_bytes)
            batch_images.append(processed_img)
            valid_filenames.append(file.filename)
        except Exception as e:
            results.append(PredictionResult(
                filename=file.filename,
                prediction=f"ERROR: {str(e)}",
                probability=0.0
            ))
    
    if batch_images:
        batch_np = np.array(batch_images) # Shape: (N, 128, 128, 1)
        
        try:
            # Output shape: (N, 1) containing probabilities
            predictions = MODEL.predict(batch_np, verbose=0)
            
            for i, prob_array in enumerate(predictions):
                prob = float(prob_array[0])
                label = 'PNEUMONIA' if prob > DECISION_THRESHOLD else 'NORMAL'
                
                results.append(PredictionResult(
                    filename=valid_filenames[i],
                    prediction=label,
                    probability=prob
                ))
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
            
    return results

@app.get("/health")
def health_check():
    """
    Check the API health status.
    """
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "threshold": DECISION_THRESHOLD
    }
