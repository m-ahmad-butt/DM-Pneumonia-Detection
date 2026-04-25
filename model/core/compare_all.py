"""
Compare All Models: Pure CNN vs CNN + Decision Tree
Clean implementation without visualization code
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

print("=" * 80)
print("MODEL COMPARISON: Pure CNN vs CNN + Decision Tree")
print("=" * 80)

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / 'chest_xray'
TEST_DIR = DATA_DIR / 'test_processed'
CNN_MODEL_PATH = Path(__file__).parent.parent / 'xray_mobilenet_model.keras'
DT_MODEL_PATH = Path(__file__).parent.parent / 'decision_tree_model.pkl'
FEATURES_DIR = Path(__file__).parent.parent / 'extracted_features'

# Load test data
print("\nLoading test data...")
X_test, y_test = [], []
for label, class_name in enumerate(['NORMAL', 'PNEUMONIA']):
    class_dir = TEST_DIR / class_name
    for file in class_dir.glob('*.npy'):
        img = np.load(file)
        X_test.append(img)
        y_test.append(label)

X_test = np.array(X_test).reshape(-1, 128, 128, 1)
y_test = np.array(y_test)
X_test_rgb = np.repeat(X_test, 3, axis=-1)

print(f"Test samples: {len(X_test)}")

# ============================================================================
# 1. PURE CNN EVALUATION
# ============================================================================
print("\n" + "-" * 80)
print("EVALUATING PURE CNN")
print("-" * 80)

cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
cnn_probs = cnn_model.predict(X_test_rgb, verbose=0)
cnn_preds = (cnn_probs > 0.5).astype(int).flatten()

cnn_accuracy = accuracy_score(y_test, cnn_preds)
cnn_precision = precision_score(y_test, cnn_preds)
cnn_recall = recall_score(y_test, cnn_preds)
cnn_f1 = f1_score(y_test, cnn_preds)

# PNEUMONIA recall
pneumonia_mask = y_test == 1
cnn_pneumonia_recall = recall_score(y_test[pneumonia_mask], cnn_preds[pneumonia_mask])

print(f"Accuracy:         {cnn_accuracy:.4f}")
print(f"Precision:        {cnn_precision:.4f}")
print(f"Recall:           {cnn_recall:.4f}")
print(f"F1-Score:         {cnn_f1:.4f}")
print(f"PNEUMONIA Recall: {cnn_pneumonia_recall:.4f}")

# ============================================================================
# 2. CNN + DECISION TREE EVALUATION
# ============================================================================
print("\n" + "-" * 80)
print("EVALUATING CNN + DECISION TREE")
print("-" * 80)

# Load features and DT model
X_test_features = np.load(FEATURES_DIR / 'X_test_features.npy')
with open(DT_MODEL_PATH, 'rb') as f:
    dt_model = pickle.load(f)

dt_preds = dt_model.predict(X_test_features)

dt_accuracy = accuracy_score(y_test, dt_preds)
dt_precision = precision_score(y_test, dt_preds)
dt_recall = recall_score(y_test, dt_preds)
dt_f1 = f1_score(y_test, dt_preds)
dt_pneumonia_recall = recall_score(y_test[pneumonia_mask], dt_preds[pneumonia_mask])

print(f"Accuracy:         {dt_accuracy:.4f}")
print(f"Precision:        {dt_precision:.4f}")
print(f"Recall:           {dt_recall:.4f}")
print(f"F1-Score:         {dt_f1:.4f}")
print(f"PNEUMONIA Recall: {dt_pneumonia_recall:.4f}")

# ============================================================================
# 3. COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

print("\n{:<25} {:<15} {:<15}".format("Metric", "Pure CNN", "CNN + DT"))
print("-" * 55)
print("{:<25} {:<15.4f} {:<15.4f}".format("Accuracy", cnn_accuracy, dt_accuracy))
print("{:<25} {:<15.4f} {:<15.4f}".format("Precision", cnn_precision, dt_precision))
print("{:<25} {:<15.4f} {:<15.4f}".format("Recall", cnn_recall, dt_recall))
print("{:<25} {:<15.4f} {:<15.4f}".format("F1-Score", cnn_f1, dt_f1))
print("{:<25} {:<15.4f} {:<15.4f}".format("PNEUMONIA Recall", cnn_pneumonia_recall, dt_pneumonia_recall))

# Calculate overall score
cnn_score = cnn_f1 * 0.5 + cnn_pneumonia_recall * 0.4 + cnn_recall * 0.1
dt_score = dt_f1 * 0.5 + dt_pneumonia_recall * 0.4 + dt_recall * 0.1

print("\n" + "-" * 55)
print("{:<25} {:<15.4f} {:<15.4f}".format("Overall Score", cnn_score, dt_score))
print("-" * 55)

# Winner
if cnn_score > dt_score:
    winner = "Pure CNN"
    improvement = ((cnn_score - dt_score) / dt_score) * 100
else:
    winner = "CNN + Decision Tree"
    improvement = ((dt_score - cnn_score) / cnn_score) * 100

print(f"\nWINNER: {winner}")
print(f"Improvement: {improvement:.2f}%")

print("\n" + "=" * 80)
