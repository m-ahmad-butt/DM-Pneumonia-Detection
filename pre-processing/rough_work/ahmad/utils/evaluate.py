import os
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

TEST_DIR        = str((SCRIPT_DIR / '../../chest_xray/test_processed/').resolve())
MODEL_SAVE_PATH = str(SCRIPT_DIR / 'xray_mobilenet_model.keras')
LOG_DIR         = str(SCRIPT_DIR / 'training_logs/')

IMG_SIZE           = (128, 128)
BATCH_SIZE         = 32
DECISION_THRESHOLD = 0.40

os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================
# GPU SETUP
# ============================================================

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU found: {gpus}")
else:
    print("No GPU — running on CPU")

# ============================================================
# LOAD TEST DATA
# ============================================================

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    color_mode  = 'grayscale',
    class_mode  = 'binary',
    shuffle     = False
)

print(f"\nTest samples : {test_generator.samples}")
print(f"Class indices: {test_generator.class_indices}")

# ============================================================
# LOAD MODEL
# ============================================================

print(f"\nLoading model from: {MODEL_SAVE_PATH}")
model = tf.keras.models.load_model(MODEL_SAVE_PATH)
model.summary()

# ============================================================
# EVALUATE
# ============================================================

print("\n" + "-" * 55)
print("EVALUATING ON TEST SET")
print("-" * 55)

test_results = model.evaluate(test_generator, verbose=1)

print("\nTest Results:")
for name, value in zip(model.metrics_names, test_results):
    print(f"  {name:<20} : {value:.4f}")

# ============================================================
# PROBABILITY DISTRIBUTION CHECK
# shows how confident model is for each class
# ============================================================

test_generator.reset()
y_pred_prob = model.predict(test_generator, verbose=1).flatten()
y_true      = test_generator.classes
class_names = list(test_generator.class_indices.keys())

normal_probs    = y_pred_prob[y_true == 0]
pneumonia_probs = y_pred_prob[y_true == 1]

print("\nProbability Distribution:")
print(f"  NORMAL    — mean: {normal_probs.mean():.3f}  "
      f"min: {normal_probs.min():.3f}  max: {normal_probs.max():.3f}")
print(f"  PNEUMONIA — mean: {pneumonia_probs.mean():.3f}  "
      f"min: {pneumonia_probs.min():.3f}  max: {pneumonia_probs.max():.3f}")

# ============================================================
# THRESHOLD SWEEP
# find the best threshold for NORMAL recall > 0.70
# ============================================================

print("\nThreshold | NORMAL recall | PNEUMONIA recall | Accuracy")
print("-" * 58)
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    pred  = (y_pred_prob > t).astype(int)
    nr    = (pred[y_true == 0] == 0).mean()
    pr    = (pred[y_true == 1] == 1).mean()
    acc   = (pred == y_true).mean()
    flag  = " <-- " if nr >= 0.70 and pr >= 0.90 else ""
    print(f"  {t}     |     {nr:.2f}      |       {pr:.2f}       |   {acc:.2f} {flag}")

# ============================================================
# THRESHOLD COMPARISON — 0.50 vs DECISION_THRESHOLD
# ============================================================

print("\n" + "-" * 55)
print(f"THRESHOLD COMPARISON: 0.50 vs {DECISION_THRESHOLD}")
print("-" * 55)

for thresh in [0.50, DECISION_THRESHOLD]:
    y_pred = (y_pred_prob > thresh).astype(int).flatten()
    print(f"\nThreshold = {thresh}")
    print(classification_report(y_true, y_pred, target_names=class_names))

y_pred_final = (y_pred_prob > DECISION_THRESHOLD).astype(int).flatten()

# ============================================================
# CONFUSION MATRIX
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Confusion Matrix Comparison', fontsize=13, fontweight='bold')

for ax, thresh, title in zip(
    axes,
    [0.50, DECISION_THRESHOLD],
    ['Default Threshold (0.50)', f'Tuned Threshold ({DECISION_THRESHOLD})']
):
    y_p  = (y_pred_prob > thresh).astype(int).flatten()
    cm_t = confusion_matrix(y_true, y_p)
    sns.heatmap(cm_t, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

plt.tight_layout()
cm_path = os.path.join(LOG_DIR, 'confusion_matrix_eval.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
print(f"\nConfusion matrix saved: {cm_path}")

# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

test_generator.reset()
all_images, all_true, all_pred = [], [], []

for batch_imgs, batch_labels in test_generator:
    preds = (model.predict(batch_imgs, verbose=0) > DECISION_THRESHOLD).astype(int).flatten()
    for i in range(len(batch_imgs)):
        all_images.append(batch_imgs[i])
        all_true.append(int(batch_labels[i]))
        all_pred.append(int(preds[i]))
        if len(all_images) >= 12:
            break
    if len(all_images) >= 12:
        break

cols = 4
rows = 3
fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
fig.suptitle(f'Sample Predictions — Threshold={DECISION_THRESHOLD} '
             f'(Green=Correct, Red=Wrong)', fontsize=11)
axes = axes.flatten()

for idx in range(12):
    img   = all_images[idx].squeeze()
    true  = class_names[all_true[idx]]
    pred  = class_names[all_pred[idx]]
    color = 'green' if all_true[idx] == all_pred[idx] else 'red'
    axes[idx].imshow(img, cmap='gray')
    axes[idx].set_title(f"True: {true}\nPred: {pred}", fontsize=9, color=color)
    axes[idx].axis('off')
    for spine in axes[idx].spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)

plt.tight_layout()
pred_path = os.path.join(LOG_DIR, 'sample_predictions_eval.png')
plt.savefig(pred_path, dpi=150, bbox_inches='tight')
print(f"Sample predictions saved: {pred_path}")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "-" * 55)
print("EVALUATION COMPLETE")
print("-" * 55)
print(f"  Model              : {MODEL_SAVE_PATH}")
print(f"  Test samples       : {test_generator.samples}")
print(f"  DECISION_THRESHOLD : {DECISION_THRESHOLD}")
print("\n  Target metrics:")
print("    PNEUMONIA recall  > 0.95")
print("    NORMAL recall     > 0.70")
print("    AUC               > 0.95")
print("-" * 55)