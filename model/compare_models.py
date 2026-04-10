import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
import joblib
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_tree_classifier import DecisionTreeClassifier

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
FEATURES_DIR = str(SCRIPT_DIR / 'extracted_features')
MODELS_DIR = str(SCRIPT_DIR / 'hybrid_models')
RESULTS_DIR = str(SCRIPT_DIR / 'comparison_results')

os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
DECISION_THRESHOLD = 0.70

print(f"\n{'-'*70}")
print("MODEL COMPARISON: Pure CNN vs CNN + Decision Tree")
print(f"{'-'*70}")

# Load test data
print(f"\nLoading test data...")
X_test_features = np.load(os.path.join(FEATURES_DIR, 'X_test_features.npy'))
y_test = np.load(os.path.join(FEATURES_DIR, 'y_test.npy'))

# Load original images for CNN
TEST_DIR = str((SCRIPT_DIR / '../chest_xray/test_processed/').resolve())
X_test_images = []
for label, class_name in enumerate(['NORMAL', 'PNEUMONIA']):
    folder = os.path.join(TEST_DIR, class_name)
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
    for fname in files:
        arr = np.load(os.path.join(folder, fname))
        X_test_images.append(arr)

X_test_images = np.array(X_test_images, dtype=np.float32)
if len(X_test_images.shape) == 3:
    X_test_images = X_test_images[..., np.newaxis]

print(f"  Test images: {X_test_images.shape}")
print(f"  Test features: {X_test_features.shape}")
print(f"  Test labels: {y_test.shape}")


# Load models
print(f"\nLoading trained models...")

# Model 1: Pure CNN
try:
    cnn_model = keras.models.load_model(MODEL_PATH, compile=False)
    print(f"  CNN model loaded")
except Exception as e:
    print(f"  Error loading model: {e}")
    print(f"  Attempting to load with custom objects...")
    
    # Custom object scope to handle quantization_config
    from tensorflow.keras import layers
    
    class CompatibleDense(layers.Dense):
        def __init__(self, *args, quantization_config=None, **kwargs):
            super().__init__(*args, **kwargs)
    
    custom_objects = {'Dense': CompatibleDense}
    cnn_model = keras.models.load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)
    print(f"  CNN model loaded with compatibility layer")

# Model 2: Decision Tree
dt_model = joblib.load(os.path.join(MODELS_DIR, 'decision_tree_classifier.pkl'))
print(f"  Decision Tree loaded")


# Make predictions
print(f"\nMaking predictions...")

# CNN predictions
cnn_proba = cnn_model.predict(X_test_images, batch_size=32, verbose=0).flatten()
cnn_pred = (cnn_proba > DECISION_THRESHOLD).astype(int)

# Decision Tree predictions
dt_pred = dt_model.predict(X_test_features)
dt_proba = dt_model.predict_proba(X_test_features)[:, 1]

print(f"  All predictions complete")


# Calculate metrics for all models
print(f"\nComparing performance...")

def calculate_metrics(y_true, y_pred, y_proba, model_name):
    """Calculate all metrics for a model"""
    return {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'NORMAL Recall': recall_score(y_true, y_pred, pos_label=0),
        'PNEUMONIA Recall': recall_score(y_true, y_pred, pos_label=1)
    }

# Calculate metrics
cnn_metrics = calculate_metrics(y_test, cnn_pred, cnn_proba, 'Pure CNN')
dt_metrics = calculate_metrics(y_test, dt_pred, dt_proba, 'CNN + Decision Tree')

# Create comparison DataFrame
comparison_df = pd.DataFrame([cnn_metrics, dt_metrics])

print(f"\n{'-'*70}")
print("PERFORMANCE COMPARISON")
print(f"{'-'*70}\n")
print(comparison_df.to_string(index=False))

# Find best model for each metric
print(f"\n{'-'*70}")
print("BEST MODEL PER METRIC")
print(f"{'-'*70}")
for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'PNEUMONIA Recall']:
    best_idx = comparison_df[metric].idxmax()
    best_model = comparison_df.loc[best_idx, 'Model']
    best_value = comparison_df.loc[best_idx, metric]
    print(f"  {metric:<20}: {best_model:<25} ({best_value:.4f})")

# Overall best model (based on F1-Score and PNEUMONIA Recall)
print(f"\n{'-'*70}")
print("RECOMMENDED MODEL")
print(f"{'-'*70}")

# Weighted scoring: F1 (50%) + PNEUMONIA Recall (40%) + Recall (10%)
comparison_df['Score'] = (
    0.5 * comparison_df['F1-Score'] +
    0.4 * comparison_df['PNEUMONIA Recall'] +
    0.1 * comparison_df['Recall']
)

best_overall_idx = comparison_df['Score'].idxmax()
best_model_name = comparison_df.loc[best_overall_idx, 'Model']
best_score = comparison_df.loc[best_overall_idx, 'Score']

print(f"\n  WINNER: {best_model_name}")
print(f"  Overall Score: {best_score:.4f}")
print(f"\n  Reasoning:")
print(f"    - Weighted by F1-Score (50%), PNEUMONIA Recall (40%), Recall (10%)")
print(f"    - F1-Score balances precision and recall")
print(f"    - PNEUMONIA Recall is critical for detecting pneumonia cases")


# Save comparison results
comparison_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)
print(f"\n  Comparison saved: {RESULTS_DIR}/model_comparison.csv")


# Visualizations
print(f"\nCreating comparison visualizations...")

# 1. Metrics Bar Chart
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Model Comparison - All Metrics', fontsize=16, fontweight='bold')

metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'NORMAL Recall', 'PNEUMONIA Recall']
colors = ['#1f77b4', '#ff7f0e']

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 3, idx % 3]
    values = comparison_df[metric].values
    bars = ax.bar(comparison_df['Model'], values, color=colors)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    
    # Highlight best
    best_idx = values.argmax()
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
    
    # Add value labels
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

plt.tight_layout()
metrics_path = os.path.join(RESULTS_DIR, 'comparison_metrics.png')
plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
print(f"  Metrics comparison: {metrics_path}")
plt.close()

# 2. Confusion Matrices Side-by-Side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')

models_data = [
    ('Pure CNN', cnn_pred, 'Blues'),
    ('CNN + Decision Tree', dt_pred, 'Oranges')
]

for ax, (name, pred, cmap) in zip(axes, models_data):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label' if ax == axes[0] else '', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)

plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, 'comparison_confusion_matrices.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
print(f"  Confusion matrices: {cm_path}")
plt.close()

# 3. Radar Chart
fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection='polar'))

categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'PNEUMONIA Recall']
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

for idx, row in comparison_df.iterrows():
    values = [row[cat] for cat in categories]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[idx])
    ax.fill(angles, values, alpha=0.15, color=colors[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 1)
ax.set_title('Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.grid(True)

plt.tight_layout()
radar_path = os.path.join(RESULTS_DIR, 'comparison_radar.png')
plt.savefig(radar_path, dpi=150, bbox_inches='tight')
print(f"  Radar chart: {radar_path}")
plt.close()

print(f"\n{'-'*70}")
print("COMPARISON COMPLETE")
print(f"{'-'*70}")
print(f"\nAll results saved to: {RESULTS_DIR}/")
print(f"\nRecommendation: Use {best_model_name} for production")