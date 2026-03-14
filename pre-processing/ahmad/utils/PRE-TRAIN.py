import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
# GLOBAL CONFIGURATION
# change any hyperparameter here — don't touch the rest of the code
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# paths
PROCESSED_DIR   = str((SCRIPT_DIR / '../../chest_xray/train_processed/').resolve())
TEST_DIR        = str((SCRIPT_DIR / '../../chest_xray/test_processed/').resolve())
MODEL_SAVE_PATH = str(SCRIPT_DIR / 'xray_mobilenet_model.keras')
LOG_DIR         = str(SCRIPT_DIR / 'training_logs/')

# image settings
IMG_SIZE     = (128, 128)
IMG_CHANNELS = 1           # grayscale

# training — phase 1 (frozen base, train head only)
BATCH_SIZE    = 32
EPOCHS        = 20         # reduced from 30 — head learns fast, stop early
LEARNING_RATE = 1e-4

# fine-tuning — phase 2 (unfreeze last few layers only)
FINE_TUNE_EPOCHS   = 10    # reduced from 20 — less fine-tuning = less overfit
FINE_TUNE_LR       = 5e-6  # reduced from 1e-5 — very careful weight updates
FINE_TUNE_AT_LAYER = 140   # only unfreeze last 14 layers (154 total - 140 = 14)
                            # was 100 (54 layers) — too many caused memorization

# validation split
VAL_SPLIT = 0.2

# regularization — stronger than before to fight overfitting
L2_LAMBDA     = 1e-3       # was 1e-4 — 10x stronger weight penalty
DROPOUT_DENSE = 0.60       # was 0.50 — more dropout in head

# dense head — smaller to reduce overfit
DENSE_UNITS = 64           # was 128 — smaller head = less memorization

# callbacks
EARLY_STOP_PATIENCE = 5    # was 8 — stop faster when val_auc plateaus
REDUCE_LR_PATIENCE  = 3
REDUCE_LR_FACTOR    = 0.5
REDUCE_LR_MIN       = 1e-7

# decision threshold
DECISION_THRESHOLD = 0.40

# ============================================================

os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================
# DATA LOADERS
# grayscale images — 1x1 conv inside model converts to 3ch for MobileNetV2
# ============================================================

datagen = ImageDataGenerator(
    rescale          = 1.0 / 255.0,
    validation_split = VAL_SPLIT
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = datagen.flow_from_directory(
    PROCESSED_DIR,
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    color_mode  = 'grayscale',
    class_mode  = 'binary',
    shuffle     = True,
    seed        = 42,
    subset      = 'training'
)

val_generator = datagen.flow_from_directory(
    PROCESSED_DIR,
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    color_mode  = 'grayscale',
    class_mode  = 'binary',
    shuffle     = False,
    seed        = 42,
    subset      = 'validation'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    color_mode  = 'grayscale',
    class_mode  = 'binary',
    shuffle     = False
)

print("\nClass indices      :", train_generator.class_indices)
print(f"Training  samples  : {train_generator.samples}")
print(f"Val       samples  : {val_generator.samples}")
print(f"Test      samples  : {test_generator.samples}")


# ============================================================
# CLASS WEIGHTS
# NORMAL=1341, PNEUMONIA=3875 — imbalanced
# balanced weights make model care more about NORMAL errors
# formula: weight = total / (n_classes * class_count)
# ============================================================

class_weights_array = compute_class_weight(
    class_weight = 'balanced',
    classes      = np.array([0, 1]),
    y            = train_generator.classes
)
class_weight_dict = {
    0: class_weights_array[0],   # NORMAL — higher weight (~1.94)
    1: class_weights_array[1]    # PNEUMONIA — lower weight (~0.67)
}
print(f"\nClass weights:")
print(f"  NORMAL    (0) : {class_weight_dict[0]:.4f}")
print(f"  PNEUMONIA (1) : {class_weight_dict[1]:.4f}")


# ============================================================
# MODEL — MobileNetV2 TRANSFER LEARNING
#
# what is transfer learning?
#   MobileNetV2 was pretrained on 1.4 million ImageNet images
#   it already knows edges, shapes, textures — we reuse this knowledge
#   we only teach it the difference between NORMAL and PNEUMONIA X-rays
#   much better than training from scratch on 5000 images
#
# why previous model overfit?
#   - fine-tuned too many layers (54) — memorized training data
#   - head was too large (Dense 128) — too many parameters
#   - L2 and dropout were too weak
#
# fixes applied:
#   - only unfreeze last 14 layers in phase 2 (was 54)
#   - smaller head: Dense(64) instead of Dense(128)
#   - L2 10x stronger: 1e-3 instead of 1e-4
#   - dropout increased: 0.60 instead of 0.50
#   - fewer fine-tune epochs: 10 instead of 20
#   - lower fine-tune LR: 5e-6 instead of 1e-5
# ============================================================

def build_mobilenet(input_shape=(128, 128, 1)):

    inputs = layers.Input(shape=input_shape)

    # convert grayscale (1ch) -> RGB (3ch)
    # MobileNetV2 needs 3 channel input
    # 1x1 conv learns optimal grayscale to RGB mapping
    x = layers.Conv2D(3, (1, 1), padding='same', name='grayscale_to_rgb')(inputs)

    # MobileNetV2 pretrained on ImageNet
    # include_top=False: remove original 1000-class head, keep feature extractor
    base_model = keras.applications.MobileNetV2(
        input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top = False,
        weights     = 'imagenet'
    )
    # phase 1: freeze all base layers
    # only our custom head trains in phase 1
    base_model.trainable = False

    # training=False: keep BatchNorm layers in inference mode
    # important when base is frozen — prevents running stats from updating
    x = base_model(x, training=False)

    # global average pooling: converts feature maps to single vector
    # e.g. 4x4x1280 -> 1280 values
    x = layers.GlobalAveragePooling2D()(x)

    # small dense head — intentionally small to prevent memorization
    x = layers.Dense(DENSE_UNITS, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(L2_LAMBDA))(x)
    x = layers.Dropout(DROPOUT_DENSE)(x)

    # sigmoid output: probability of PNEUMONIA (1=PNEUMONIA, 0=NORMAL)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model, base_model


model, base_model = build_mobilenet(input_shape=(IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS))
model.summary()

print(f"\nMobileNetV2 total layers  : {len(base_model.layers)}")
print(f"Layers frozen in phase 1  : all ({len(base_model.layers)})")
print(f"Layers unfrozen in phase 2: {len(base_model.layers) - FINE_TUNE_AT_LAYER}")


# ============================================================
# COMPILE — PHASE 1
# ============================================================

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss      = 'binary_crossentropy',
    metrics   = [
        'accuracy',
        keras.metrics.AUC(name='auc'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')
    ]
)


# ============================================================
# CALLBACKS
# reused for both phases — suffix differentiates csv logs
# ============================================================

def get_callbacks(suffix=''):
    return [
        # save model only when val_auc improves
        callbacks.ModelCheckpoint(
            filepath             = MODEL_SAVE_PATH,
            monitor              = 'val_auc',
            save_best_only       = True,
            mode                 = 'max',
            verbose              = 1
        ),
        # stop training early if val_auc stops improving
        callbacks.EarlyStopping(
            monitor              = 'val_auc',
            patience             = EARLY_STOP_PATIENCE,
            restore_best_weights = True,
            mode                 = 'max',
            verbose              = 1
        ),
        # halve LR when val_loss plateaus
        callbacks.ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = REDUCE_LR_FACTOR,
            patience = REDUCE_LR_PATIENCE,
            min_lr   = REDUCE_LR_MIN,
            verbose  = 1
        ),
        callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=0),
        callbacks.CSVLogger(
            os.path.join(LOG_DIR, f'training_history{suffix}.csv'),
            append=False
        )
    ]


# ============================================================
# PHASE 1 — TRAIN HEAD ONLY (base frozen)
# only our Dense(64) + Dropout + sigmoid trains here
# fast, stable — learns basic NORMAL vs PNEUMONIA distinction
# ============================================================

print("\n" + "=" * 55)
print("PHASE 1 — TRAINING HEAD (base frozen)")
print("=" * 55)

history_phase1 = model.fit(
    train_generator,
    epochs          = EPOCHS,
    validation_data = val_generator,
    callbacks       = get_callbacks(suffix='_phase1'),
    class_weight    = class_weight_dict,
    verbose         = 1
)

print(f"\nPhase 1 complete. Best model saved to: {MODEL_SAVE_PATH}")


# ============================================================
# PHASE 2 — FINE-TUNING (unfreeze last 14 layers only)
# last 14 layers learn X-ray specific high-level features
# very low LR to avoid destroying pretrained ImageNet weights
# ============================================================

print("\n" + "=" * 55)
print(f"PHASE 2 — FINE-TUNING (last {len(base_model.layers) - FINE_TUNE_AT_LAYER} layers)")
print("=" * 55)

# unfreeze base model
base_model.trainable = True

# re-freeze everything before FINE_TUNE_AT_LAYER
# layers before this index learned general features — don't touch them
for layer in base_model.layers[:FINE_TUNE_AT_LAYER]:
    layer.trainable = False

trainable = sum(1 for l in base_model.layers if l.trainable)
print(f"MobileNetV2 trainable layers: {trainable} / {len(base_model.layers)}")

# recompile with very low LR
# high LR in fine-tuning destroys pretrained weights — leads to overfitting
model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
    loss      = 'binary_crossentropy',
    metrics   = [
        'accuracy',
        keras.metrics.AUC(name='auc'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')
    ]
)

history_phase2 = model.fit(
    train_generator,
    epochs          = FINE_TUNE_EPOCHS,
    validation_data = val_generator,
    callbacks       = get_callbacks(suffix='_phase2'),
    class_weight    = class_weight_dict,
    verbose         = 1
)

print(f"\nPhase 2 complete. Best model saved to: {MODEL_SAVE_PATH}")


# ============================================================
# TRAINING CURVES — phase 1 + phase 2 combined
# green dashed line shows where fine-tuning started
# ============================================================

def plot_training_history(h1, h2, save_dir=LOG_DIR):
    combined   = {k: h1.history[k] + h2.history[k] for k in h1.history}
    phase1_end = len(h1.history['loss'])

    metrics = [
        ('accuracy',  'val_accuracy',  'Accuracy'),
        ('loss',      'val_loss',      'Loss'),
        ('auc',       'val_auc',       'AUC'),
        ('precision', 'val_precision', 'Precision'),
        ('recall',    'val_recall',    'Recall'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History — Phase 1 + Phase 2', fontsize=15, fontweight='bold')
    axes = axes.flatten()

    for idx, (tk, vk, title) in enumerate(metrics):
        if tk in combined:
            axes[idx].plot(combined[tk], label='Train', color='steelblue')
            axes[idx].plot(combined[vk], label='Val',   color='tomato')
            axes[idx].axvline(x=phase1_end, color='green',
                              linestyle='--', linewidth=1.5, label='Fine-tune start')
            axes[idx].set_title(title)
            axes[idx].set_xlabel('Epoch')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

    axes[-1].axis('off')
    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved: {path}")
    plt.show()


plot_training_history(history_phase1, history_phase2)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "-" * 55)
print("EVALUATING ON TEST SET")
print("-" * 55)

best_model   = keras.models.load_model(MODEL_SAVE_PATH)
test_results = best_model.evaluate(test_generator, verbose=1)

print("\nTest Results:")
for name, value in zip(best_model.metrics_names, test_results):
    print(f"  {name:<20} : {value:.4f}")


# ============================================================
# THRESHOLD SWEEP + COMPARISON
# find best threshold — marked with <-- in output
# ============================================================

test_generator.reset()
y_pred_prob = best_model.predict(test_generator, verbose=1).flatten()
y_true      = test_generator.classes
class_names = list(test_generator.class_indices.keys())

print("\nProbability Distribution:")
print(f"  NORMAL    — mean: {y_pred_prob[y_true==0].mean():.3f}")
print(f"  PNEUMONIA — mean: {y_pred_prob[y_true==1].mean():.3f}")

print("\nThreshold | NORMAL recall | PNEUMONIA recall | Accuracy")
print("-" * 58)
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    pred = (y_pred_prob > t).astype(int)
    nr   = (pred[y_true == 0] == 0).mean()
    pr   = (pred[y_true == 1] == 1).mean()
    acc  = (pred == y_true).mean()
    flag = " <--" if nr >= 0.70 and pr >= 0.90 else ""
    print(f"  {t}     |     {nr:.2f}      |       {pr:.2f}       |  {acc:.2f}{flag}")

print("\n" + "-" * 55)
print(f"THRESHOLD COMPARISON: 0.50 vs {DECISION_THRESHOLD}")
print("-" * 55)

for thresh in [0.50, DECISION_THRESHOLD]:
    y_pred = (y_pred_prob > thresh).astype(int)
    print(f"\nThreshold = {thresh}")
    print(classification_report(y_true, y_pred, target_names=class_names))

y_pred_final = (y_pred_prob > DECISION_THRESHOLD).astype(int)


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
    y_p  = (y_pred_prob > thresh).astype(int)
    cm_t = confusion_matrix(y_true, y_p)
    sns.heatmap(cm_t, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

plt.tight_layout()
cm_path = os.path.join(LOG_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
print(f"\nConfusion matrix saved: {cm_path}")
plt.show()


# ============================================================
# SAMPLE PREDICTIONS — green=correct, red=wrong
# ============================================================

def visualize_predictions(generator, model, n_samples=12,
                           threshold=DECISION_THRESHOLD, save_dir=LOG_DIR):
    generator.reset()
    all_images, all_true, all_pred = [], [], []

    for batch_imgs, batch_labels in generator:
        preds = (model.predict(batch_imgs, verbose=0) > threshold).astype(int).flatten()
        for i in range(len(batch_imgs)):
            all_images.append(batch_imgs[i])
            all_true.append(int(batch_labels[i]))
            all_pred.append(int(preds[i]))
            if len(all_images) >= n_samples:
                break
        if len(all_images) >= n_samples:
            break

    cols = 4
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle(f'Sample Predictions — Threshold={threshold} '
                 f'(Green=Correct, Red=Wrong)', fontsize=11)
    axes = axes.flatten()

    for idx in range(n_samples):
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

    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    path = os.path.join(save_dir, 'sample_predictions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Sample predictions saved: {path}")
    plt.show()


visualize_predictions(test_generator, best_model, n_samples=12)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "-" * 55)
print("FINAL SUMMARY")
print("-" * 55)
print(f"  Model              : MobileNetV2 + custom head")
print(f"  IMG_SIZE           : {IMG_SIZE}")
print(f"  BATCH_SIZE         : {BATCH_SIZE}")
print(f"  Phase 1 LR         : {LEARNING_RATE}")
print(f"  Phase 2 LR         : {FINE_TUNE_LR}")
print(f"  Fine-tune layers   : last {len(base_model.layers) - FINE_TUNE_AT_LAYER}")
print(f"  L2_LAMBDA          : {L2_LAMBDA}")
print(f"  DROPOUT_DENSE      : {DROPOUT_DENSE}")
print(f"  DENSE_UNITS        : {DENSE_UNITS}")
print(f"  DECISION_THRESHOLD : {DECISION_THRESHOLD}")
print(f"  NORMAL weight      : {class_weight_dict[0]:.4f}")
print(f"  PNEUMONIA weight   : {class_weight_dict[1]:.4f}")
print(f"  Model saved        : {MODEL_SAVE_PATH}")
print(f"  Logs               : {LOG_DIR}")
print("\n  Target metrics:")
print("    PNEUMONIA recall  > 0.95  (missing pneumonia is dangerous)")
print("    NORMAL recall     > 0.70  (false alarms are costly)")
print("    AUC               > 0.95  (overall discrimination)")
print("-" * 55)