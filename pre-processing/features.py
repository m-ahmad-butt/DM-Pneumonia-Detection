import numpy as np
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MODEL_PATH = '../model/xray_mobilenet_model.keras'
NORMAL_DIR = Path('../chest_xray/test_processed/NORMAL')

print("-"*70)
print("SIMPLE FEATURE COMPARISON")
print("-"*70)

# Load one sample image
normal_files = sorted(list(NORMAL_DIR.glob('*.npy')))
sample_img = np.load(normal_files[0])

print(f"\n1. RAW IMAGE")
print(f"   Shape: {sample_img.shape}")
print(f"   Total pixel values: {sample_img.shape[0]} x {sample_img.shape[1]} = {sample_img.size:,}")

# Load model and extract features
print(f"\n2. LOADING CNN MODEL...")

# Create custom Dense layer that properly handles quantization_config
class CompatibleDense(layers.Dense):
    def __init__(self, *args, quantization_config=None, **kwargs):
        # Remove quantization_config before passing to parent
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_config(cls, config):
        # Remove quantization_config from config
        config.pop('quantization_config', None)
        return cls(**config)

custom_objects = {'Dense': CompatibleDense}

try:
    model = keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
    print(f"   Model loaded")
except Exception as e:
    print(f"   Error loading model: {e}")
    print(f"   This is a Keras version compatibility issue.")
    print(f"   The model was saved with a different Keras version.")
    exit(1)

# Extract features
feature_layer = model.get_layer('global_average_pooling2d')
feature_extractor = keras.Model(inputs=model.input, outputs=feature_layer.output)

sample_img_cnn = np.expand_dims(sample_img, axis=(0, -1))
cnn_features = feature_extractor.predict(sample_img_cnn, verbose=0)[0]

print(f"\n3. CNN FEATURES (after extraction)")
print(f"   Shape: {cnn_features.shape}")
print(f"   Total features: {cnn_features.size:,}")
print(f"   Reduction: {sample_img.size:,} -> {cnn_features.size:,}")
print(f"   Percentage: {(1 - cnn_features.size/sample_img.size)*100:.1f}% reduction!")
