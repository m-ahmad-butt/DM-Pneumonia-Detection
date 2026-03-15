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

# GPU SETUP
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU found: {gpus}")
else:
    print("No GPU — running on CPU")

# GLOBAL CONFIGURATION
SCRIPT_DIR = Path(__file__).resolve().parent

# paths
PROCESSED_DIR   = str((SCRIPT_DIR / '../../chest_xray/train_processed/').resolve())
TEST_DIR        = str((SCRIPT_DIR / '../../chest_xray/test_processed/').resolve())
MODEL_SAVE_PATH = str(SCRIPT_DIR / 'xray_mobilenet_model.keras')
LOG_DIR         = str(SCRIPT_DIR / 'training_logs/')

# image settings
IMG_SIZE     = (128, 128)
IMG_CHANNELS = 1

# training — phase 1
BATCH_SIZE    = 32
EPOCHS        = 20
LEARNING_RATE = 1e-4

# fine-tuning — phase 2
FINE_TUNE_EPOCHS   = 10
FINE_TUNE_LR       = 5e-6
FINE_TUNE_AT_LAYER = 140   # unfreeze last 14 of 154 MobileNetV2 layers

# validation split
VAL_SPLIT = 0.2

# regularization
L2_LAMBDA     = 1e-3
DROPOUT_DENSE = 0.60
DENSE_UNITS   = 64

# callbacks
EARLY_STOP_PATIENCE = 5
REDUCE_LR_PATIENCE  = 3
REDUCE_LR_FACTOR    = 0.5
REDUCE_LR_MIN       = 1e-7

# decision threshold
DECISION_THRESHOLD = 0.40

os.makedirs(LOG_DIR, exist_ok=True)

def load_npy_dataset(base_dir, val_split=0.0, subset='all', seed=42):
    """
    Load all .npy files from base_dir/NORMAL and base_dir/PNEUMONIA.
    Returns numpy arrays (X, y) — float32 images and binary labels.
    subset: 'all', 'training', 'validation'
    """
    X, y = [], []

    for label, class_name in enumerate(['NORMAL', 'PNEUMONIA']):
        folder = os.path.join(base_dir, class_name)
        if not os.path.exists(folder):
            print(f"  WARNING: {folder} not found")
            continue
        files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
        print(f"  {class_name}: {len(files)} files")
        for fname in files:
            arr = np.load(os.path.join(folder, fname))   # float32, (128,128)
            X.append(arr)
            y.append(label)

    X = np.array(X, dtype=np.float32)   # (N, 128, 128)
    y = np.array(y, dtype=np.float32)   # (N,)

    # added channel dimension for model: (N, 128, 128) -> (N, 128, 128, 1)
    X = X[..., np.newaxis]

    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    if val_split > 0 and subset in ('training', 'validation'):
        split_at = int(len(X) * (1 - val_split))
        if subset == 'training':
            return X[:split_at], y[:split_at]
        else:
            return X[split_at:], y[split_at:]

    return X, y


# LOAD DATA
print("\nLoading TRAIN data...")
X_train, y_train = load_npy_dataset(PROCESSED_DIR, val_split=VAL_SPLIT, subset='training')
X_val,   y_val   = load_npy_dataset(PROCESSED_DIR, val_split=VAL_SPLIT, subset='validation')

print("\nLoading TEST data...")
X_test, y_test = load_npy_dataset(TEST_DIR)

print(f"\nTraining   : {X_train.shape}  labels: {y_train.shape}")
print(f"Validation : {X_val.shape}  labels: {y_val.shape}")
print(f"Test       : {X_test.shape}  labels: {y_test.shape}")

print(f"\nData stats (z-score check):")
print(f"  Train mean: {X_train.mean():.3f}  std: {X_train.std():.3f}  (should be ≈0, ≈1)")
print(f"  Val   mean: {X_val.mean():.3f}  std: {X_val.std():.3f}")
print(f"  Test  mean: {X_test.mean():.3f}  std: {X_test.std():.3f}")
train_normal_mean = X_train[y_train == 0].mean()
test_normal_mean  = X_test[y_test   == 0].mean()
print(f"\nDomain shift check:")
print(f"  Train NORMAL mean : {train_normal_mean:.3f}")
print(f"  Test  NORMAL mean : {test_normal_mean:.3f}")
print(f"  Difference        : {abs(train_normal_mean - test_normal_mean):.3f}  (should be < 0.1)")


# CLASS WEIGHTS
class_weights_array = compute_class_weight(
    class_weight = 'balanced',
    classes      = np.array([0, 1]),
    y            = y_train
)
class_weight_dict = {
    0: class_weights_array[0],   # NORMAL — higher weight
    1: class_weights_array[1]    # PNEUMONIA — lower weight
}
print(f"\nClass weights:")
print(f"  NORMAL    (0) : {class_weight_dict[0]:.4f}")
print(f"  PNEUMONIA (1) : {class_weight_dict[1]:.4f}")

def build_mobilenet(input_shape=(128, 128, 1)):

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(3, (1, 1), padding='same', name='grayscale_to_rgb')(inputs)

    base_model = keras.applications.MobileNetV2(
        input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top = False,
        weights     = 'imagenet'
    )
    base_model.trainable = False   # phase 1: freeze all

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(DENSE_UNITS, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(L2_LAMBDA))(x)
    x = layers.Dropout(DROPOUT_DENSE)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model, base_model


model, base_model = build_mobilenet(input_shape=(IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS))
model.summary()

print(f"\nMobileNetV2 layers        : {len(base_model.layers)}")
print(f"Frozen in phase 1         : all")
print(f"Unfrozen in phase 2       : {len(base_model.layers) - FINE_TUNE_AT_LAYER}")


# COMPILE — PHASE 1
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


# CALLBACKS
def get_callbacks(suffix=''):
    return [
        callbacks.ModelCheckpoint(
            filepath             = MODEL_SAVE_PATH,
            monitor              = 'val_auc',
            save_best_only       = True,
            mode                 = 'max',
            verbose              = 1
        ),
        callbacks.EarlyStopping(
            monitor              = 'val_auc',
            patience             = EARLY_STOP_PATIENCE,
            restore_best_weights = True,
            mode                 = 'max',
            verbose              = 1
        ),
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


# PHASE 1 — TRAIN HEAD ONLY (base frozen)
print("\n" + "=" * 55)
print("PHASE 1 — TRAINING HEAD (base frozen)")
print("=" * 55)

history_phase1 = model.fit(
    X_train, y_train,
    epochs          = EPOCHS,
    batch_size      = BATCH_SIZE,
    validation_data = (X_val, y_val),
    callbacks       = get_callbacks(suffix='_phase1'),
    class_weight    = class_weight_dict,
    verbose         = 1
)

print(f"\nPhase 1 complete. Best model: {MODEL_SAVE_PATH}")


# PHASE 2 — FINE-TUNING (last 14 layers only)
print("\n" + "=" * 55)
print(f"PHASE 2 — FINE-TUNING (last {len(base_model.layers) - FINE_TUNE_AT_LAYER} layers)")
print("=" * 55)

base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT_LAYER]:
    layer.trainable = False

trainable = sum(1 for l in base_model.layers if l.trainable)
print(f"MobileNetV2 trainable layers: {trainable} / {len(base_model.layers)}")

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
    X_train, y_train,
    epochs          = FINE_TUNE_EPOCHS,
    batch_size      = BATCH_SIZE,
    validation_data = (X_val, y_val),
    callbacks       = get_callbacks(suffix='_phase2'),
    class_weight    = class_weight_dict,
    verbose         = 1
)

print(f"\nPhase 2 complete. Best model: {MODEL_SAVE_PATH}")


# TRAINING CURVES — phase 1 + phase 2 combined
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


# TEST EVALUATION
print("\n" + "-" * 55)
print("EVALUATING ON TEST SET")
print("-" * 55)

best_model   = keras.models.load_model(MODEL_SAVE_PATH)
test_results = best_model.evaluate(X_test, y_test, verbose=1)

print("\nTest Results:")
for name, value in zip(best_model.metrics_names, test_results):
    print(f"  {name:<20} : {value:.4f}")


# THRESHOLD SWEEP + COMPARISON
y_pred_prob = best_model.predict(X_test, verbose=1).flatten()
class_names = ['NORMAL', 'PNEUMONIA']

print("\nProbability Distribution:")
print(f"  NORMAL    — mean: {y_pred_prob[y_test==0].mean():.3f}")
print(f"  PNEUMONIA — mean: {y_pred_prob[y_test==1].mean():.3f}")

print("\nThreshold | NORMAL recall | PNEUMONIA recall | Accuracy")
print("-" * 58)
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    pred = (y_pred_prob > t).astype(int)
    nr   = (pred[y_test == 0] == 0).mean()
    pr   = (pred[y_test == 1] == 1).mean()
    acc  = (pred == y_test).mean()
    flag = " <--" if nr >= 0.70 and pr >= 0.90 else ""
    print(f"  {t}     |     {nr:.2f}      |       {pr:.2f}       |  {acc:.2f}{flag}")

print("\n" + "-" * 55)
print(f"THRESHOLD COMPARISON: 0.50 vs {DECISION_THRESHOLD}")
print("-" * 55)

for thresh in [0.50, DECISION_THRESHOLD]:
    y_pred = (y_pred_prob > thresh).astype(int)
    print(f"\nThreshold = {thresh}")
    print(classification_report(y_test, y_pred, target_names=class_names))


# CONFUSION MATRIX
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Confusion Matrix Comparison', fontsize=13, fontweight='bold')

for ax, thresh, title in zip(
    axes,
    [0.50, DECISION_THRESHOLD],
    ['Default Threshold (0.50)', f'Tuned Threshold ({DECISION_THRESHOLD})']
):
    y_p  = (y_pred_prob > thresh).astype(int)
    cm_t = confusion_matrix(y_test, y_p)
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


# SAMPLE PREDICTIONS — green=correct, red=wrong
n_samples = 12
indices   = np.random.choice(len(X_test), n_samples, replace=False)
preds     = (y_pred_prob[indices] > DECISION_THRESHOLD).astype(int)

cols = 4
rows = (n_samples + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
fig.suptitle(f'Sample Predictions — Threshold={DECISION_THRESHOLD} '
             f'(Green=Correct, Red=Wrong)', fontsize=11)
axes = axes.flatten()

for idx, (i, pred) in enumerate(zip(indices, preds)):
    img   = X_test[i].squeeze()
    true  = class_names[int(y_test[i])]
    p     = class_names[int(pred)]
    color = 'green' if int(y_test[i]) == int(pred) else 'red'
    axes[idx].imshow(img, cmap='gray')
    axes[idx].set_title(f"True: {true}\nPred: {p}", fontsize=9, color=color)
    axes[idx].axis('off')
    for spine in axes[idx].spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)

for idx in range(n_samples, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
pred_path = os.path.join(LOG_DIR, 'sample_predictions.png')
plt.savefig(pred_path, dpi=150, bbox_inches='tight')
print(f"Sample predictions saved: {pred_path}")
plt.show()


# FINAL SUMMARY
print("\n" + "-" * 55)
print("FINAL SUMMARY")
print("-" * 55)
print(f"  Model              : MobileNetV2 + custom head")
print(f"  Data format        : float32 .npy (z-score preserved)")
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