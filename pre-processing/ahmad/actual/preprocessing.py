import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from tqdm import tqdm


# CONFIGURATION
DATASET_DIR = '../../../chest_xray/train/'
VAL_DIR     = '../../../chest_xray/val/'
TEST_DIR    = '../../../chest_xray/test/'

# output dirs — saving .npy float32 files
# previous version saved uint8 png which UNDID z-score normalization
# this version saves raw float32 so z-score is fully preserved
OUTPUT_DIR      = '../../chest_xray/train_processed/'
VAL_OUTPUT_DIR  = '../../chest_xray/val_processed/'
TEST_OUTPUT_DIR = '../../chest_xray/test_processed/'

NORMAL_DIR    = os.path.join(DATASET_DIR, 'NORMAL')
PNEUMONIA_DIR = os.path.join(DATASET_DIR, 'PNEUMONIA')

OUT_NORMAL    = os.path.join(OUTPUT_DIR, 'NORMAL')
OUT_PNEUMONIA = os.path.join(OUTPUT_DIR, 'PNEUMONIA')

TARGET_SIZE   = (128, 128)
SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.bmp')


# COUNT IMAGES AND DETECT IMBALANCE
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
    minority      = 'NORMAL' if normal_count < pneumonia_count else 'PNEUMONIA'
    majority_count = max(normal_count, pneumonia_count)
    minority_count = min(normal_count, pneumonia_count)
    needed        = majority_count - minority_count
    print(f"  Status: IMBALANCED — {minority} is minority class")
    print(f"  Images needed to balance: {needed}")
    NEEDS_AUGMENTATION = True

print("-" * 55)


# PREPROCESSING PIPELINE
# Resize: Bicubic | Denoise: Median | Contrast: CLAHE | Norm: Z-score
#
# KEY CHANGE from previous version:
#   before: z-score applied then saved as uint8 png
#           uint8 conversion rescales min->0 max->255
#           this DESTROYS z-score — brightness difference returns
#           result: test NORMAL mean=151 vs train NORMAL mean=124 (domain shift)
#
#   now:    z-score applied then saved as float32 .npy
#           no conversion — exact float values preserved on disk
#           result: all images have mean=0, std=1 regardless of scanner

def preprocess_image(img_path, target_size=TARGET_SIZE):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    # bicubic resize — preserves subtle intensity gradations
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)

    # median blur — edge-preserving noise removal
    img = cv2.medianBlur(img, 3)

    # CLAHE — local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)

    # z-score normalization — mean=0, std=1 per image
    # this makes every image have the same brightness distribution
    img  = img.astype(np.float32)
    mean = np.mean(img)
    std  = np.std(img)
    if std == 0:
        std = 1
    img = (img - mean) / std

    return img  # float32, shape (128, 128), mean≈0, std≈1


def save_as_npy(img_array, save_path):
    np.save(str(save_path), img_array.astype(np.float32))


# AUGMENTATION FUNCTIONS
# def aug_horizontal_flip(img):
#     # mirror left-right — safe, lungs are bilaterally symmetric
#     return cv2.flip(img, 1)
#
#
# def aug_rotation(img, angle_range=(-10, 10)):
#     # small rotation — simulates patient positioning difference
#     rows, cols = img.shape[:2]
#     angle = random.uniform(*angle_range)
#     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
#     return cv2.warpAffine(img, M, (cols, rows),
#                           flags=cv2.INTER_CUBIC,
#                           borderMode=cv2.BORDER_REFLECT)
#
#
# def aug_brightness_contrast(img, alpha_range=(0.85, 1.15)):
#     # scale pixel values — no additive constant (avoids clipping)
#     alpha     = random.uniform(*alpha_range)
#     img_float = np.clip(img.astype(np.float32) * alpha, 0, 255)
#     return img_float.astype(img.dtype)
#
#
# def aug_zoom_crop(img, zoom_range=(0.85, 0.95)):
#     # zoom in and resize back — simulates field of view difference
#     rows, cols           = img.shape[:2]
#     scale                = random.uniform(*zoom_range)
#     new_rows, new_cols   = int(rows * scale), int(cols * scale)
#     start_r              = (rows - new_rows) // 2
#     start_c              = (cols - new_cols) // 2
#     cropped              = img[start_r:start_r + new_rows, start_c:start_c + new_cols]
#     return cv2.resize(cropped, (cols, rows), interpolation=cv2.INTER_CUBIC)
#
#
# def aug_shear(img, shear_range=(-0.05, 0.05)):
#     # mild horizontal shear — simulates slight patient tilt
#     rows, cols = img.shape[:2]
#     shear      = random.uniform(*shear_range)
#     M          = np.float32([[1, shear, -shear * rows / 2], [0, 1, 0]])
#     return cv2.warpAffine(img, M, (cols, rows),
#                           flags=cv2.INTER_CUBIC,
#                           borderMode=cv2.BORDER_REFLECT)
#
#
# def aug_gaussian_noise(img, std_range=(2, 6)):
#     # very mild noise — simulates detector variation
#     std   = random.uniform(*std_range)
#     noisy = img.astype(np.float32) + np.random.normal(0, std, img.shape)
#     return np.clip(noisy, 0, 255).astype(img.dtype)
#
#
# AUGMENTATION_PIPELINE = [
#     ('flip',       aug_horizontal_flip),
#     ('rotate',     aug_rotation),
#     ('brightness', aug_brightness_contrast),
#     ('zoom',       aug_zoom_crop),
#     ('shear',      aug_shear),
#     ('noise',      aug_gaussian_noise),
# ]
#
#
# def augment_image(img):
#     augmented = img.copy()
#     applied   = []
#     pipeline  = AUGMENTATION_PIPELINE.copy()
#     random.shuffle(pipeline)
#     for name, func in pipeline:
#         if random.random() < 0.6:
#             augmented = func(augmented)
#             applied.append(name)
#     if not applied:
#         name, func = random.choice(AUGMENTATION_PIPELINE)
#         augmented  = func(augmented)
#         applied.append(name)
#     return augmented, applied
#
#
# def augment_minority_class(src_dir, out_dir, src_files, needed_count):
#     print(f"\nAugmenting minority class — need {needed_count} more images...")
#     existing = [f for f in os.listdir(out_dir) if f.endswith('.npy')]
#     if not existing:
#         print("  ERROR: No preprocessed files found.")
#         return
#     generated = 0
#     batch     = 0
#     with tqdm(total=needed_count, desc="  Generating augmented") as pbar:
#         while generated < needed_count:
#             src_fname = random.choice(src_files)
#             src_path  = os.path.join(src_dir, src_fname)
#             try:
#                 img_raw = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
#                 if img_raw is None:
#                     continue
#                 img_aug, transforms = augment_image(img_raw)
#                 temp_path = '/tmp/xray_aug_temp.png'
#                 cv2.imwrite(temp_path, img_aug)
#                 processed = preprocess_image(temp_path)
#                 stem      = Path(src_fname).stem
#                 aug_label = '_'.join(transforms[:2])
#                 save_name = f"{stem}_aug_{batch:05d}_{aug_label}.npy"
#                 save_as_npy(processed, os.path.join(out_dir, save_name))
#                 generated += 1
#                 batch     += 1
#                 pbar.update(1)
#             except Exception as e:
#                 print(f"\n  WARNING: Skipped — {e}")
#                 continue
#     print(f"  Done — {generated} augmented files generated")


# PROCESS FUNCTIONS
def process_class(src_dir, out_dir, class_name):
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(SUPPORTED_EXT)]

    print(f"\n  Processing {class_name} ({len(files)} images)...")
    saved = []

    for fname in tqdm(files, desc=f"    {class_name}"):
        src_path  = os.path.join(src_dir, fname)
        save_name = f"{Path(fname).stem}_preprocessed.npy"
        save_path = os.path.join(out_dir, save_name)
        try:
            save_as_npy(preprocess_image(src_path), save_path)
            saved.append(save_path)
        except Exception as e:
            print(f"  ERROR: {fname} — {e}")

    print(f"  Done — {len(saved)} .npy files saved to {out_dir}")
    return saved


def process_split(split_dir, output_dir, split_name):
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
            src_path  = os.path.join(src, fname)
            save_path = os.path.join(dest, f"{Path(fname).stem}_p.npy")
            try:
                save_as_npy(preprocess_image(src_path), save_path)
            except Exception as e:
                print(f"  ERROR: {fname} — {e}")

    # summary
    for class_name in ['NORMAL', 'PNEUMONIA']:
        dest  = os.path.join(output_dir, class_name)
        count = len([f for f in os.listdir(dest) if f.endswith('.npy')]) if os.path.exists(dest) else 0
        print(f"  {split_name} {class_name}: {count} files")


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
        ('5. Z-score (saved)',    img_zscore),
    ]

    fig, axes = plt.subplots(2, len(steps), figsize=(20, 7))
    fig.suptitle('Preprocessing Pipeline — Z-score preserved as float32',
                 fontsize=13, fontweight='bold')

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
    fig.suptitle('Class Balance', fontsize=13, fontweight='bold')
    categories = ['NORMAL', 'PNEUMONIA']

    for ax, counts, title in zip(
        axes,
        [[before_normal, before_pneumonia], [after_normal, after_pneumonia]],
        ['Before Processing', 'After Processing']
    ):
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


def verify_npy_stats(output_dir, split_name, n=10):
    # verify saved npy files have correct mean≈0 std≈1
    for class_name in ['NORMAL', 'PNEUMONIA']:
        folder = os.path.join(output_dir, class_name)
        if not os.path.exists(folder):
            continue
        files  = [f for f in os.listdir(folder) if f.endswith('.npy')][:n]
        means  = [np.load(os.path.join(folder, f)).mean() for f in files]
        stds   = [np.load(os.path.join(folder, f)).std()  for f in files]
        print(f"  {split_name} {class_name} — "
              f"mean: {np.mean(means):.3f}  std: {np.mean(stds):.3f}  "
              f"(should be ≈0 and ≈1)")


# MAIN PIPELINE
if __name__ == '__main__':

    import shutil

    print("\n" + "=" * 55)
    print("X-RAY PREPROCESSING PIPELINE — float32 npy")
    print("=" * 55)

    # delete old processed folders — old png files are wrong
    print("\nCleaning old processed folders...")
    for d in [OUTPUT_DIR, VAL_OUTPUT_DIR, TEST_OUTPUT_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"  Deleted: {d}")

    os.makedirs(OUT_NORMAL,    exist_ok=True)
    os.makedirs(OUT_PNEUMONIA, exist_ok=True)

    # step 1 — visualize preprocessing on one sample
    sample_normal = os.path.join(NORMAL_DIR, normal_files[0])
    print("\n[1/5] Visualizing preprocessing pipeline...")
    visualize_preprocessing_steps(sample_normal)

    # step 2 — preprocess TRAIN
    print("\n[2/5] Preprocessing TRAIN images...")
    process_class(NORMAL_DIR,    OUT_NORMAL,    'NORMAL')
    process_class(PNEUMONIA_DIR, OUT_PNEUMONIA, 'PNEUMONIA')

    # step 3 — augmentation disabled
    print("\n[3/5] Augmentation skipped — using class_weight in training")

    # step 4 — preprocess VAL (no augmentation)
    print("\n[4/5] Preprocessing VAL images...")
    process_split(VAL_DIR, VAL_OUTPUT_DIR, 'VAL')

    # step 5 — preprocess TEST (no augmentation)
    print("\n[5/5] Preprocessing TEST images...")
    process_split(TEST_DIR, TEST_OUTPUT_DIR, 'TEST')

    # verify z-score preserved correctly
    print("\nVerifying z-score preservation:")
    verify_npy_stats(OUTPUT_DIR,      'TRAIN')
    verify_npy_stats(VAL_OUTPUT_DIR,  'VAL')
    verify_npy_stats(TEST_OUTPUT_DIR, 'TEST')

    # final counts
    final_normal    = len([f for f in os.listdir(OUT_NORMAL)    if f.endswith('.npy')])
    final_pneumonia = len([f for f in os.listdir(OUT_PNEUMONIA) if f.endswith('.npy')])

    visualize_class_balance(
        before_normal    = normal_count,
        before_pneumonia = pneumonia_count,
        after_normal     = final_normal,
        after_pneumonia  = final_pneumonia
    )

    print("\n" + "-" * 55)
    print("PIPELINE COMPLETE")
    print("-" * 55)
    print(f"  Format            : float32 .npy (z-score preserved)")
    print(f"  TRAIN NORMAL      : {final_normal}")
    print(f"  TRAIN PNEUMONIA   : {final_pneumonia}")
    print(f"  Imbalance ratio   : {min(final_normal, final_pneumonia) / max(final_normal, final_pneumonia):.3f}")
    print(f"  Imbalance handled : class_weight in train.py")

    for split_name, split_out in [('VAL', VAL_OUTPUT_DIR), ('TEST', TEST_OUTPUT_DIR)]:
        for cls in ['NORMAL', 'PNEUMONIA']:
            d = os.path.join(split_out, cls)
            c = len([f for f in os.listdir(d) if f.endswith('.npy')]) if os.path.exists(d) else 0
            print(f"  {split_name} {cls:<12}: {c}")

    print("-" * 55)