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

# GPU SETUP
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU found: {gpus}")
else:
    print("No GPU — running on CPU")


# GLOBAL CONFIGURATION
SCRIPT_DIR    = Path(__file__).resolve().parent

# paths
PROCESSED_DIR = str((SCRIPT_DIR / '../../chest_xray/train_processed/').resolve())
TEST_DIR      = str((SCRIPT_DIR / '../../chest_xray/test_processed/').resolve())
MODEL_SAVE_PATH = str(SCRIPT_DIR / 'xray_cnn_model.keras')
LOG_DIR         = str(SCRIPT_DIR / 'training_logs/')

# image settings
IMG_SIZE      = (128, 128)   # width x height — must match preprocessing TARGET_SIZE
IMG_CHANNELS  = 1            # 1 = grayscale, 3 = RGB

# training hyperparameters
BATCH_SIZE    = 32           # number of images per gradient update
EPOCHS        = 50           # max epochs — early stopping will cut this short
LEARNING_RATE = 1e-4         # adam optimizer learning rate
VAL_SPLIT     = 0.2          # fraction of train data used for validation (20%)

# model regularization
L2_LAMBDA     = 1e-4         # L2 weight decay — penalizes large weights to reduce overfit
DROPOUT_CONV  = 0.25         # dropout after conv blocks 1 and 2
DROPOUT_DEEP  = 0.40         # dropout after conv block 3 (deeper = more regularization)
DROPOUT_DENSE = 0.50         # dropout after dense layer

# model architecture
FILTERS       = [32, 64, 128]   # filters per conv block — doubles each block
DENSE_UNITS   = 128              # units in fully connected layer

# callbacks
EARLY_STOP_PATIENCE    = 10   # stop if val_auc doesn't improve for this many epochs
REDUCE_LR_PATIENCE     = 4    # reduce LR if val_loss doesn't improve for this many epochs
REDUCE_LR_FACTOR       = 0.5  # new_lr = lr * factor
REDUCE_LR_MIN          = 1e-7 # minimum learning rate floor

# decision threshold
# default 0.5 means predict PNEUMONIA if probability > 50%
# lower value = more conservative = fewer false positives on NORMAL
# higher value = more aggressive = catches more PNEUMONIA but misses more NORMAL
DECISION_THRESHOLD = 0.40

os.makedirs(LOG_DIR, exist_ok=True)

# DATA LOADERS
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
    subset      = 'training'     # 80% of data
)

val_generator = datagen.flow_from_directory(
    PROCESSED_DIR,
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    color_mode  = 'grayscale',
    class_mode  = 'binary',
    shuffle     = False,        
    seed        = 42,
    subset      = 'validation'   # 20% of data
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    color_mode  = 'grayscale',
    class_mode  = 'binary',
    shuffle     = False
)

print("\nClass indices     :", train_generator.class_indices)
print(f"Training  samples : {train_generator.samples}")
print(f"Val       samples : {val_generator.samples}")
print(f"Test      samples : {test_generator.samples}")


# CLASS WEIGHTS
# dataset is imbalanced: NORMAL=1341, PNEUMONIA=3875
# class_weight='balanced' computes:
#   weight = total_samples / (n_classes * class_samples)
# NORMAL gets higher weight so model penalized more for missing it
# this fixes low NORMAL recall without changing the data

class_weights_array = compute_class_weight(
    class_weight = 'balanced',
    classes      = np.array([0, 1]),
    y            = train_generator.classes
)
class_weight_dict = {
    0: class_weights_array[0],   # NORMAL weight — will be ~2.88
    1: class_weights_array[1]    # PNEUMONIA weight — will be ~1.0
}
print(f"\nClass weights computed:")
print(f"  NORMAL    (0) : {class_weight_dict[0]:.4f}")
print(f"  PNEUMONIA (1) : {class_weight_dict[1]:.4f}")

def build_cnn(input_shape=(128, 128, 1)):

    # L2 regularizer applied to all Conv and Dense layers
    reg = keras.regularizers.l2(L2_LAMBDA)

    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(FILTERS[0], (3, 3), padding='same', activation='relu',
                      kernel_regularizer=reg),
        layers.Conv2D(FILTERS[0], (3, 3), padding='same', activation='relu',
                      kernel_regularizer=reg),
        layers.MaxPooling2D((2, 2)),       # 128x128 -> 64x64
        layers.Dropout(DROPOUT_CONV),

        # Block 2
        layers.Conv2D(FILTERS[1], (3, 3), padding='same', activation='relu',
                      kernel_regularizer=reg),
        layers.Conv2D(FILTERS[1], (3, 3), padding='same', activation='relu',
                      kernel_regularizer=reg),
        layers.MaxPooling2D((2, 2)),       # 64x64 -> 32x32
        layers.Dropout(DROPOUT_CONV),

        # Block 3
        layers.Conv2D(FILTERS[2], (3, 3), padding='same', activation='relu',
                      kernel_regularizer=reg),
        layers.Conv2D(FILTERS[2], (3, 3), padding='same', activation='relu',
                      kernel_regularizer=reg),
        layers.MaxPooling2D((2, 2)),       # 32x32 -> 16x16
        layers.Dropout(DROPOUT_DEEP),

        layers.GlobalAveragePooling2D(),

        # fully connected layer 
        layers.Dense(DENSE_UNITS, activation='relu', kernel_regularizer=reg),
        layers.Dropout(DROPOUT_DENSE),

        # output LAYER
        layers.Dense(1, activation='sigmoid')
    ])

    return model


model = build_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS))
model.summary()


# COMPILE
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


# CALLBACKS [DEBUG]
callback_list = [

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
        os.path.join(LOG_DIR, 'training_history.csv'),
        append=False
    )
]


# TRAINING
print("\n" + "-" * 55)
print("STARTING TRAINING")
print("-" * 55)

history = model.fit(
    train_generator,
    epochs          = EPOCHS,
    validation_data = val_generator,
    callbacks       = callback_list,
    class_weight    = class_weight_dict,
    verbose         = 1
)

print(f"\nBest model saved to: {MODEL_SAVE_PATH}")


# TRAINING CURVES
def plot_training_history(history, save_dir=LOG_DIR):
    metrics = [
        ('accuracy',  'val_accuracy',  'Accuracy'),
        ('loss',      'val_loss',      'Loss'),
        ('auc',       'val_auc',       'AUC'),
        ('precision', 'val_precision', 'Precision'),
        ('recall',    'val_recall',    'Recall'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History', fontsize=15, fontweight='bold')
    axes = axes.flatten()

    for idx, (tk, vk, title) in enumerate(metrics):
        if tk in history.history:
            axes[idx].plot(history.history[tk], label='Train', color='steelblue')
            axes[idx].plot(history.history[vk], label='Val',   color='tomato')
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


plot_training_history(history)


# TEST EVALUATION
print("\n" + "-" * 55)
print("EVALUATING ON TEST SET")
print("-" * 55)

best_model   = keras.models.load_model(MODEL_SAVE_PATH)
test_results = best_model.evaluate(test_generator, verbose=1)

print("\nTest Results:")
for name, value in zip(best_model.metrics_names, test_results):
    print(f"  {name:<20} : {value:.4f}")


# THRESHOLD COMPARISON
test_generator.reset()
y_pred_prob = best_model.predict(test_generator, verbose=1)
y_true      = test_generator.classes
class_names = list(test_generator.class_indices.keys())

print("\n" + "-" * 55)
print(f"THRESHOLD COMPARISON: 0.50 vs {DECISION_THRESHOLD}")
print("-" * 55)

for thresh in [0.50, DECISION_THRESHOLD]:
    y_pred = (y_pred_prob > thresh).astype(int).flatten()
    print(f"\nThreshold = {thresh}")
    print(classification_report(y_true, y_pred, target_names=class_names))
y_pred_final = (y_pred_prob > DECISION_THRESHOLD).astype(int).flatten()


# CONFUSION MATRIX
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
cm_path = os.path.join(LOG_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
print(f"\nConfusion matrix saved: {cm_path}")
plt.show()


# SAMPLE PREDICTIONS
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


# FINAL SUMMARY
print("\n" + "-" * 55)
print("FINAL SUMMARY")
print("-" * 55)
print(f"  IMG_SIZE           : {IMG_SIZE}")
print(f"  BATCH_SIZE         : {BATCH_SIZE}")
print(f"  LEARNING_RATE      : {LEARNING_RATE}")
print(f"  L2_LAMBDA          : {L2_LAMBDA}")
print(f"  DECISION_THRESHOLD : {DECISION_THRESHOLD}")
print(f"  NORMAL weight      : {class_weight_dict[0]:.4f}")
print(f"  PNEUMONIA weight   : {class_weight_dict[1]:.4f}")
print(f"  Model saved        : {MODEL_SAVE_PATH}")
print(f"  Logs               : {LOG_DIR}")
print("\n  Target metrics for medical X-ray:")
print("    PNEUMONIA recall  > 0.95  (missing pneumonia is dangerous)")
print("    NORMAL recall     > 0.70  (false alarms are costly)")
print("    AUC               > 0.95  (overall discrimination)")
print("-" * 55)