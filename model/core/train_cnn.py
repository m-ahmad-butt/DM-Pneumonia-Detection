import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pathlib import Path

print("-" * 80)
print("PURE CNN TRAINING - MobileNetV2")
print("-" * 80)

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / 'chest_xray'
TRAIN_DIR = DATA_DIR / 'train_processed'
TEST_DIR = DATA_DIR / 'test_processed'
MODEL_PATH = Path(__file__).parent.parent / 'xray_mobilenet_model.keras'

# Load preprocessed data
print("\nLoading preprocessed data...")
X_train, y_train = [], []
X_test, y_test = [], []

# Load training data
for label, class_name in enumerate(['NORMAL', 'PNEUMONIA']):
    class_dir = TRAIN_DIR / class_name
    for file in class_dir.glob('*.npy'):
        img = np.load(file)
        X_train.append(img)
        y_train.append(label)

# Load test data
for label, class_name in enumerate(['NORMAL', 'PNEUMONIA']):
    class_dir = TEST_DIR / class_name
    for file in class_dir.glob('*.npy'):
        img = np.load(file)
        X_test.append(img)
        y_test.append(label)

X_train = np.array(X_train).reshape(-1, 128, 128, 1)
X_test = np.array(X_test).reshape(-1, 128, 128, 1)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Convert grayscale to RGB (MobileNetV2 requires 3 channels)
X_train_rgb = np.repeat(X_train, 3, axis=-1)
X_test_rgb = np.repeat(X_test, 3, axis=-1)

# Build model
print("\nBuilding MobileNetV2 model...")
# pretained model trained on imagenet dataset, inlcudetop means remove pooling and desne (we have our own classes not 1000)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

# conv layer output
x = base_model.output

# onverts feature maps to single vector
x = GlobalAveragePooling2D()(x)

# dense layer
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# final binary class
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='binary_crossentropy', 
              metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)

# Train
print("\nTraining model...")
history = model.fit(
    X_train_rgb, y_train,
    validation_data=(X_test_rgb, y_test),
    epochs=20,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Save model
model.save(MODEL_PATH)
print(f"\nModel saved: {MODEL_PATH}")

# Evaluate
print("\nEvaluating on test set...")
test_loss, test_acc, test_prec, test_rec = model.evaluate(X_test_rgb, y_test, verbose=0)

print("\n" + "-" * 80)
print("PURE CNN RESULTS")
print("-" * 80)
print(f"Accuracy:  {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall:    {test_rec:.4f}")
print(f"F1-Score:  {2 * (test_prec * test_rec) / (test_prec + test_rec):.4f}")
print("-" * 80)
