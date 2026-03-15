import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from tqdm import tqdm


# CONFIGURATION

DATASET_DIR   = '../../chest_xray/train/'
VAL_DIR       = '../../chest_xray/val/'
TEST_DIR      = '../../chest_xray/test/'

OUTPUT_DIR        = '../../chest_xray/train_processed/'
VAL_OUTPUT_DIR    = '../../chest_xray/val_processed/'
TEST_OUTPUT_DIR   = '../../chest_xray/test_processed/'

NORMAL_DIR    = os.path.join(DATASET_DIR, 'NORMAL')
PNEUMONIA_DIR = os.path.join(DATASET_DIR, 'PNEUMONIA')

OUT_NORMAL    = os.path.join(OUTPUT_DIR, 'NORMAL')
OUT_PNEUMONIA = os.path.join(OUTPUT_DIR, 'PNEUMONIA')

TARGET_SIZE   = (128, 128) 
SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.bmp')


# COUNT OF IMAGES AND DETECT IMBALANCE
def count_images(folder):
    return [
        f for f in os.listdir(folder)
        if f.lower().endswith(SUPPORTED_EXT)
    ]

normal_files    = count_images(NORMAL_DIR)
pneumonia_files = count_images(PNEUMONIA_DIR)

normal_count    = len(normal_files)
pneumonia_count = len(pneumonia_files)

print("-" * 55)
print("DATASET ANALYSIS")
print("-" * 55)
print(f"  NORMAL    images : {normal_count}")
print(f"  PNEUMONIA images : {pneumonia_count}")
print(f"  Difference       : {abs(normal_count - pneumonia_count)}")

if normal_count == pneumonia_count:
    print("  Status: Balanced — no augmentation needed")
    NEEDS_AUGMENTATION = False
else:
    minority = 'NORMAL' if normal_count < pneumonia_count else 'PNEUMONIA'
    majority_count = max(normal_count, pneumonia_count)
    minority_count = min(normal_count, pneumonia_count)
    needed = majority_count - minority_count
    print(f"  Status: IMBALANCED — {minority} is minority class")
    print(f"  Images needed to balance: {needed}")
    NEEDS_AUGMENTATION = True

print("-" * 55)


# PREPROCESSING PIPELINE
# Same pipeline applied to train, val, and test
# Resize: Bicubic | Denoise: Median | Contrast: CLAHE | Norm: Z-score

def preprocess_image(img_path, target_size=TARGET_SIZE):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    # Bicubic resize
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)

    # Median blur — edge-preserving noise removal
    img = cv2.medianBlur(img, 3)

    # CLAHE — local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Z-score normalization
    img = img.astype(np.float32)
    mean = np.mean(img)
    std  = np.std(img)
    if std == 0:
        std = 1
    img = (img - mean) / std

    return img


def save_preprocessed(img_array, save_path):
    img_min = img_array.min()
    img_max = img_array.max()
    if img_max - img_min == 0:
        img_uint8 = np.zeros_like(img_array, dtype=np.uint8)
    else:
        img_uint8 = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    cv2.imwrite(str(save_path), img_uint8)


# Data Augmentation
def aug_horizontal_flip(img):
    return cv2.flip(img, 1)


def aug_rotation(img, angle_range=(-10, 10)):
    rows, cols = img.shape[:2]
    angle = random.uniform(*angle_range)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (cols, rows),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REFLECT)


def aug_brightness_contrast(img, alpha_range=(0.85, 1.15)):
    alpha = random.uniform(*alpha_range)
    img_float = np.clip(img.astype(np.float32) * alpha, 0, 255)
    return img_float.astype(img.dtype)


def aug_zoom_crop(img, zoom_range=(0.85, 0.95)):
    rows, cols = img.shape[:2]
    scale = random.uniform(*zoom_range)
    new_rows, new_cols = int(rows * scale), int(cols * scale)
    start_r = (rows - new_rows) // 2
    start_c = (cols - new_cols) // 2
    cropped = img[start_r:start_r + new_rows, start_c:start_c + new_cols]
    return cv2.resize(cropped, (cols, rows), interpolation=cv2.INTER_CUBIC)


def aug_shear(img, shear_range=(-0.05, 0.05)):
    rows, cols = img.shape[:2]
    shear = random.uniform(*shear_range)
    M = np.float32([[1, shear, -shear * rows / 2], [0, 1, 0]])
    return cv2.warpAffine(img, M, (cols, rows),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REFLECT)


def aug_gaussian_noise(img, std_range=(2, 6)):
    std = random.uniform(*std_range)
    noisy = img.astype(np.float32) + np.random.normal(0, std, img.shape)
    return np.clip(noisy, 0, 255).astype(img.dtype)


AUGMENTATION_PIPELINE = [
    ('flip',       aug_horizontal_flip),
    ('rotate',     aug_rotation),
    ('brightness', aug_brightness_contrast),
    ('zoom',       aug_zoom_crop),
    ('shear',      aug_shear),
    ('noise',      aug_gaussian_noise),
]


def augment_image(img):
    augmented = img.copy()
    applied   = []
    pipeline  = AUGMENTATION_PIPELINE.copy()
    random.shuffle(pipeline)

    for name, func in pipeline:
        if random.random() < 0.6:
            augmented = func(augmented)
            applied.append(name)

    if not applied:
        name, func = random.choice(AUGMENTATION_PIPELINE)
        augmented  = func(augmented)
        applied.append(name)

    return augmented, applied



# PROCESS FUNCTIONS
def process_class(src_dir, out_dir, class_name):
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(SUPPORTED_EXT)]

    print(f"\n  Processing {class_name} ({len(files)} images)...")
    processed_paths = []

    for fname in tqdm(files, desc=f"    {class_name}"):
        src_path  = os.path.join(src_dir, fname)
        save_name = f"{Path(fname).stem}_preprocessed.png"
        save_path = os.path.join(out_dir, save_name)
        try:
            save_preprocessed(preprocess_image(src_path), save_path)
            processed_paths.append(save_path)
        except Exception as e:
            print(f"  ERROR: {fname} — {e}")

    print(f"  Done — {len(processed_paths)} saved to {out_dir}")
    return processed_paths


def process_split(split_dir, output_dir, split_name):
    """
    Process an entire split (val or test) — both NORMAL and PNEUMONIA.
    No augmentation applied — val/test must be clean original images only.
    """
    print(f"\n--- Processing {split_name} split ---")

    for class_name in ['NORMAL', 'PNEUMONIA']:
        src  = os.path.join(split_dir,  class_name)
        dest = os.path.join(output_dir, class_name)

        if not os.path.exists(src):
            print(f"  WARNING: {src} not found — skipping")
            continue

        files = [f for f in os.listdir(src) if f.lower().endswith(SUPPORTED_EXT)]
        os.makedirs(dest, exist_ok=True)
        print(f"  {class_name}: {len(files)} images")

        for fname in tqdm(files, desc=f"    {class_name}"):
            src_path  = os.path.join(src,  fname)
            save_path = os.path.join(dest, f"{Path(fname).stem}_p.png")
            try:
                save_preprocessed(preprocess_image(src_path), save_path)
            except Exception as e:
                print(f"  ERROR: {fname} — {e}")

    for class_name in ['NORMAL', 'PNEUMONIA']:
        dest  = os.path.join(output_dir, class_name)
        count = len([f for f in os.listdir(dest) if f.endswith('.png')]) if os.path.exists(dest) else 0
        print(f"  {split_name} {class_name} processed: {count}")


def augment_minority_class(src_dir, out_dir, src_files, needed_count):
    """Generate augmented images for the minority class until balanced."""
    print(f"\nAugmenting minority class — need {needed_count} more images...")

    existing = [f for f in os.listdir(out_dir) if f.endswith('.png')]
    if not existing:
        print("  ERROR: No preprocessed images found in output dir.")
        return

    generated = 0
    batch     = 0

    with tqdm(total=needed_count, desc="  Generating augmented") as pbar:
        while generated < needed_count:
            src_fname = random.choice(src_files)
            src_path  = os.path.join(src_dir, src_fname)
            try:
                img_raw = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
                if img_raw is None:
                    continue

                img_aug, transforms = augment_image(img_raw)

                temp_path = '/tmp/xray_aug_temp.png'
                cv2.imwrite(temp_path, img_aug)
                processed = preprocess_image(temp_path)

                stem      = Path(src_fname).stem
                aug_label = '_'.join(transforms[:2])
                save_name = f"{stem}_aug_{batch:05d}_{aug_label}.png"
                save_preprocessed(processed, os.path.join(out_dir, save_name))

                generated += 1
                batch     += 1
                pbar.update(1)

            except Exception as e:
                print(f"\n  WARNING: Skipped — {e}")
                continue

    print(f"  Done — {generated} augmented images generated")

# VISUALIZATION
def visualize_preprocessing_steps(img_path, save_fig=True):
    img_orig    = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img_orig, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
    img_median  = cv2.medianBlur(img_resized, 3)
    clahe       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe   = clahe.apply(img_median)
    img_zscore  = (img_clahe.astype(np.float32) - img_clahe.mean()) / (img_clahe.std() + 1e-7)

    steps = [
        ('1. Original',           img_orig),
        ('2. Resized (Bicubic)',   img_resized),
        ('3. Median Blur (k=3)',   img_median),
        ('4. CLAHE',              img_clahe),
        ('5. Z-score Normalized', img_zscore),
    ]

    fig, axes = plt.subplots(2, len(steps), figsize=(20, 7))
    fig.suptitle('Preprocessing Pipeline', fontsize=14, fontweight='bold')

    for col, (title, image) in enumerate(steps):
        axes[0, col].imshow(image, cmap='gray')
        axes[0, col].set_title(title, fontsize=9)
        axes[0, col].axis('off')
        axes[1, col].hist(image.ravel(), bins=100, color='black', alpha=0.75)
        axes[1, col].set_xlabel('Pixel Value')

    plt.tight_layout()
    if save_fig:
        path = os.path.join(OUTPUT_DIR, 'preprocessing_pipeline_visualization.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved: {path}")
    plt.show()


def visualize_class_balance(before_normal, before_pneumonia,
                             after_normal, after_pneumonia, save_fig=True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Class Balance — Before vs After', fontsize=13, fontweight='bold')
    categories = ['NORMAL', 'PNEUMONIA']

    for ax, counts, title in zip(axes,
                                  [[before_normal, before_pneumonia],
                                   [after_normal,  after_pneumonia]],
                                  ['Before Augmentation', 'After Augmentation']):
        ax.bar(categories, counts, color=['steelblue', 'tomato'], width=0.5)
        ax.set_title(title)
        ax.set_ylabel('Image Count')
        for i, v in enumerate(counts):
            ax.text(i, v + 20, str(v), ha='center', fontweight='bold')

    plt.tight_layout()
    if save_fig:
        path = os.path.join(OUTPUT_DIR, 'class_balance_chart.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Class balance chart saved: {path}")
    plt.show()


# MAIN PIPELINE
if __name__ == '__main__':

    print("\n" + "=" * 55)
    print("X-RAY PREPROCESSING AND AUGMENTATION PIPELINE")
    print("=" * 55)

    os.makedirs(OUT_NORMAL,    exist_ok=True)
    os.makedirs(OUT_PNEUMONIA, exist_ok=True)

    # --- Visualize one sample ---
    sample_normal = os.path.join(NORMAL_DIR, normal_files[0])
    print("\n[1/6] Visualizing preprocessing pipeline...")
    visualize_preprocessing_steps(sample_normal)

    # --- Preprocess TRAIN ---
    print("\n[2/6] Preprocessing TRAIN images...")
    process_class(NORMAL_DIR,    OUT_NORMAL,    'NORMAL')
    process_class(PNEUMONIA_DIR, OUT_PNEUMONIA, 'PNEUMONIA')

    # --- Augment minority class ---
    print("\n[3/6] Augmenting minority class if needed...")
    if NEEDS_AUGMENTATION:
        after_normal_count    = len([f for f in os.listdir(OUT_NORMAL)    if f.endswith('.png')])
        after_pneumonia_count = len([f for f in os.listdir(OUT_PNEUMONIA) if f.endswith('.png')])

        if normal_count < pneumonia_count:
            augment_minority_class(
                src_dir      = NORMAL_DIR,
                out_dir      = OUT_NORMAL,
                src_files    = normal_files,
                needed_count = after_pneumonia_count - after_normal_count
            )
        else:
            augment_minority_class(
                src_dir      = PNEUMONIA_DIR,
                out_dir      = OUT_PNEUMONIA,
                src_files    = pneumonia_files,
                needed_count = after_normal_count - after_pneumonia_count
            )
    else:
        print("  Already balanced — skipping augmentation")

    # --- Preprocess VAL ---
    print("\n[4/6] Preprocessing VAL images...")
    print("  NOTE: No augmentation on val — must stay original for honest evaluation")
    process_split(VAL_DIR, VAL_OUTPUT_DIR, 'VAL')

    # --- Preprocess TEST ---
    print("\n[5/6] Preprocessing TEST images...")
    print("  NOTE: No augmentation on test — must stay original for honest evaluation")
    process_split(TEST_DIR, TEST_OUTPUT_DIR, 'TEST')

    # --- Final summary ---
    print("\n[6/6] Final dataset summary...")

    final_normal    = len([f for f in os.listdir(OUT_NORMAL)    if f.endswith('.png')])
    final_pneumonia = len([f for f in os.listdir(OUT_PNEUMONIA) if f.endswith('.png')])

    visualize_class_balance(
        before_normal    = normal_count,
        before_pneumonia = pneumonia_count,
        after_normal     = final_normal,
        after_pneumonia  = final_pneumonia
    )

    print("\n" + "-" * 55)
    print("PIPELINE COMPLETE")
    print("-" * 55)
    print(f"  TRAIN NORMAL      : {final_normal}")
    print(f"  TRAIN PNEUMONIA   : {final_pneumonia}")
    print(f"  Balance ratio     : {min(final_normal, final_pneumonia) / max(final_normal, final_pneumonia):.3f}")

    for split_name, split_out in [('VAL', VAL_OUTPUT_DIR), ('TEST', TEST_OUTPUT_DIR)]:
        for cls in ['NORMAL', 'PNEUMONIA']:
            d = os.path.join(split_out, cls)
            c = len([f for f in os.listdir(d) if f.endswith('.png')]) if os.path.exists(d) else 0
            print(f"  {split_name} {cls:<12}: {c}")

    print("-" * 55)