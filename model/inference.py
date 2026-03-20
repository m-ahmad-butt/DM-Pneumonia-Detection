import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import matplotlib.pyplot as plt

# CONFIGURATION & MODEL LOADING
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(SCRIPT_DIR / 'xray_mobilenet_model.keras')
IMG_SIZE = (128, 128)

DECISION_THRESHOLD = 0.70  # matches backend/main.py and README-documented production threshold

print(f"[*] Loading model from {MODEL_PATH}...")
try:
    MODEL = keras.models.load_model(MODEL_PATH)
    print("[+] Model loaded successfully!")
except Exception as e:
    print(f"[-] Error loading model: {e}")
    MODEL = None

class_names = ['NORMAL', 'PNEUMONIA']

def preprocess_single_image(img_path, target_size=IMG_SIZE):
    """
    Standard preprocessing pipeline used during training:
    1. Read Grayscale
    2. Bicubic Resize
    3. Median Blur
    4. CLAHE Contrast Enhancement
    5. Z-score Normalization
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at: {img_path}")

    # Resize
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)

    # Median blur
    img = cv2.medianBlur(img, 3)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Z-score normalization
    img = img.astype(np.float32)
    mean = np.mean(img)
    std = np.std(img)
    if std == 0:
        std = 1
    img = (img - mean) / std

    # Reshape for model (1, 128, 128, 1)
    img = img[np.newaxis, ..., np.newaxis]
    return img

def predict_pneumonia(image_path):
    """
    Takes an image path (or .npy path) and returns the prediction and probability.
    """
    if MODEL is None:
        return "Model not loaded", 0.0

    try:
        # Check if the file is already a preprocessed .npy file
        if str(image_path).lower().endswith('.npy'):
            print(f"[*] Loading preprocessed numpy file: {image_path}")
            processed_img = np.load(image_path)
            # Ensure correct shape (batch_dim, h, w, channels)
            if len(processed_img.shape) == 2:
                processed_img = processed_img[np.newaxis, ..., np.newaxis]
            elif len(processed_img.shape) == 3:
                processed_img = processed_img[np.newaxis, ...]
        else:
            raw_preprocessed = preprocess_single_image(image_path)
            processed_img = np.array(raw_preprocessed, dtype=np.float32)
        p_mean = np.mean(processed_img)
        p_std = np.std(processed_img)
        print(f"[*] Preprocessing Stats -> Mean: {p_mean:.4f}, Std: {p_std:.4f}")
        
        # Predict
        # We use [0][0] because output is [[prob]]
        prob_array = MODEL.predict(processed_img, verbose=0)
        prob = float(prob_array[0][0])
        
        # Apply threshold
        prediction = 'PNEUMONIA' if prob > DECISION_THRESHOLD else 'NORMAL'
        
        return prediction, prob
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 0.0

def visualize_prediction(image_path, prediction, probability):
    """
    Displays the image with its prediction.
    """
    try:
        if str(image_path).lower().endswith('.npy'):
            img_data = np.load(image_path)
            img_vis = ((img_data - img_data.min()) / (img_data.max() - img_data.min()) * 255).astype(np.uint8)
        else:
            img_vis = cv2.imread(str(image_path))
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img_vis, cmap='gray' if len(img_vis.shape) == 2 else None)
        
        color = 'red' if prediction == 'PNEUMONIA' else 'green'
        title = f"PREDICTION: {prediction}\nProbability: {probability:.4f}\nThreshold: {DECISION_THRESHOLD}"
        
        plt.title(title, fontsize=15, color=color, fontweight='bold')
        plt.axis('off')
        
        save_dir = SCRIPT_DIR / 'training_logs'
        if save_dir.exists():
            save_name = f"prediction_{Path(image_path).stem}.png"
            plt.savefig(save_dir / save_name, bbox_inches='tight')
            print(f"[+] Visualization saved to: training_logs/{save_name}")
        
        plt.show()
    except Exception as e:
        print(f"[-] Visualization error: {e}")

if __name__ == "__main__":
    import sys
    import platform
    
    print("\n" + "-"*40)
    print(" MEGA PNEUMONIA DETECTION INFERENCE ")
    print("-"*40)
    
    while True:
        print("\n" + "-"*40)
        print("[INPUT] Enter image path (or drag here).")
        print("[QUIT] Type '1' to exit.")
        
        user_input = input("\nPath: ").strip().replace('"', '').replace("'", "")
        
        if user_input in ['1', 'q', 'exit', 'quit']:
            print("[*] Goodbye!")
            break
        
        if not user_input:
            continue

        test_image_path = user_input
        
        if platform.system() == "Linux" and ":" in test_image_path[:3]:
            drive = test_image_path[0].lower()
            test_image_path = f"/mnt/{drive}/" + test_image_path[3:].replace("\\", "/")
            print(f"[*] Converted Path: {test_image_path}")
        
        if os.path.exists(test_image_path):
            result, confidence = predict_pneumonia(test_image_path)
            
            print("\n" + "-"*30)
            print(f" FINAL RESULT : {result}")
            print(f" CONFIDENCE   : {confidence:.4f}")
            print(f" THRESHOLD    : {DECISION_THRESHOLD}")
            print("-"*30)
            
            visualize_prediction(test_image_path, result, confidence)
        else:
            relative_fallback = str((SCRIPT_DIR / "../../../chest_xray/test/PNEUMONIA/person1685_virus_2903.jpeg").resolve())
            if "person1685" in test_image_path and os.path.exists(relative_fallback):
                test_image_path = relative_fallback
                print(f"[*] Found via fallback: {test_image_path}")
                result, confidence = predict_pneumonia(test_image_path)
                visualize_prediction(test_image_path, result, confidence)
            else:
                print(f"[-] File not found: {test_image_path}")
