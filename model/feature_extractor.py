import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

# GPU SETUP (WSL2)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU found: {gpus}")
else:
    print("No GPU — running on CPU (will be slower)")

# CONFIGURATION
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(SCRIPT_DIR / 'xray_mobilenet_model.keras')

# Data paths
TRAIN_DIR = str((SCRIPT_DIR / '../chest_xray/train_processed/').resolve())
TEST_DIR = str((SCRIPT_DIR / '../chest_xray/test_processed/').resolve())

# Output paths
FEATURES_DIR = str(SCRIPT_DIR / 'extracted_features')
os.makedirs(FEATURES_DIR, exist_ok=True)

print(f"\n{'-'*60}")
print("CNN FEATURE EXTRACTION FOR DECISION TREE & RULE-BASED")
print(f"{'-'*60}")

# Load trained model
print(f"\nLoading trained CNN model...")
try:
    # Try loading with compile=False to avoid optimizer issues
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print(f"  Model loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"  Error loading model: {e}")
    print(f"  Attempting to load with custom objects...")
    
    # Custom object scope to handle quantization_config
    import tensorflow as tf
    from tensorflow.keras import layers
    
    # Create custom Dense layer that ignores quantization_config
    class CompatibleDense(layers.Dense):
        def __init__(self, *args, quantization_config=None, **kwargs):
            super().__init__(*args, **kwargs)
    
    custom_objects = {'Dense': CompatibleDense}
    model = keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
    print(f"  Model loaded with compatibility layer")

feature_layer = model.get_layer('global_average_pooling2d')
feature_extractor = keras.Model(
    inputs=model.input,
    outputs=feature_layer.output
)

print(f"  Feature extractor created")
print(f"  Feature dimension: {feature_extractor.output_shape[1]}")


def load_npy_dataset(base_dir, subset_name):
    """Load preprocessed .npy files and extract labels"""
    X, y, filenames = [], [], []
    
    for label, class_name in enumerate(['NORMAL', 'PNEUMONIA']):
        folder = os.path.join(base_dir, class_name)
        if not os.path.exists(folder):
            print(f"  WARNING: {folder} not found")
            continue
        
        files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
        print(f"  {subset_name} {class_name}: {len(files)} files")
        
        for fname in files:
            arr = np.load(os.path.join(folder, fname))
            X.append(arr)
            y.append(label)
            filenames.append(fname)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    # Add channel dimension: (N, 128, 128) -> (N, 128, 128, 1)
    if len(X.shape) == 3:
        X = X[..., np.newaxis]
    
    return X, y, filenames


# Load datasets
print(f"\nLoading preprocessed datasets...")
X_train, y_train, train_files = load_npy_dataset(TRAIN_DIR, "TRAIN")
X_test, y_test, test_files = load_npy_dataset(TEST_DIR, "TEST")

print(f"\n  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")


# Extract features
print(f"\nExtracting CNN features...")

print("  Extracting TRAIN features...")
train_features = feature_extractor.predict(X_train, batch_size=32, verbose=0)

print("  Extracting TEST features...")
test_features = feature_extractor.predict(X_test, batch_size=32, verbose=0)

print(f"  Train features shape: {train_features.shape}")
print(f"  Test features shape: {test_features.shape}")


# Save extracted features
print(f"\nSaving extracted features...")

np.save(os.path.join(FEATURES_DIR, 'X_train_features.npy'), train_features)
np.save(os.path.join(FEATURES_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(FEATURES_DIR, 'X_test_features.npy'), test_features)
np.save(os.path.join(FEATURES_DIR, 'y_test.npy'), y_test)

print(f"  Saved to: {FEATURES_DIR}/")
print(f"    - X_train_features.npy ({train_features.shape})")
print(f"    - y_train.npy ({y_train.shape})")
print(f"    - X_test_features.npy ({test_features.shape})")
print(f"    - y_test.npy ({y_test.shape})")

print(f"\n{'-'*60}")
print("FEATURE EXTRACTION COMPLETE")
