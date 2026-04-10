"""
Preprocessing Evaluation
Testing all combinations of techniques to find optimal pipeline
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product

# Configuration
DATASET_DIR = '../chest_xray/train/NORMAL'
NUM_SAMPLES = 50
TARGET_SIZE = (128, 128)

print("-"*80)
print("PREPROCESSING EVALUATION")
print("-"*80)

# Loading sample images
print(f"\nLoading {NUM_SAMPLES} sample images...")
image_files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
np.random.seed(42)
sample_files = np.random.choice(image_files, min(NUM_SAMPLES, len(image_files)), replace=False)

original_images = []
for fname in sample_files:
    img = cv2.imread(os.path.join(DATASET_DIR, fname), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        original_images.append(img)

print(f"Loaded {len(original_images)} images!!!")


def calculate_metrics(image):
    """Calculating quantitative metrics"""
    # uint8 type
    img_uint8 = image.astype(np.uint8) if image.dtype != np.uint8 else image
    
    # std dev
    contrast_metric = np.std(image)
    
    # Sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(img_uint8, cv2.CV_64F)
    sharpness = laplacian.var()
    
    # Edge Strength (sobel)
    sobelx = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.mean(np.sqrt(sobelx**2 + sobely**2))
    
    # Entropy (info)
    histogram = cv2.calcHist([img_uint8], [0], None, [256], [0, 256])
    histogram = histogram / (histogram.sum() + 1e-10)
    entropy = -np.sum(histogram * np.log2(histogram + 1e-10))
    
    # SNR (SIGNAL-TO-NOISE RATIO)
    mean_signal = np.mean(image)
    std_noise = np.std(image)
    snr = mean_signal / (std_noise + 1e-10)
    
    # 6. Dynamic Range
    dynamic_range = np.max(image) - np.min(image)
    
    return {
        'contrast_metric': contrast_metric,
        'sharpness': sharpness,
        'edge_strength': edge_strength,
        'entropy': entropy,
        'snr': snr,
        'dynamic_range': dynamic_range
    }


# Defining all technique options
RESIZE_METHODS = {
    'cubic': cv2.INTER_CUBIC,
    'linear': cv2.INTER_LINEAR,
    'area': cv2.INTER_AREA,
    'nearest': cv2.INTER_NEAREST
}

DENOISE_METHODS = {
    'none': None,
    'median_3': lambda img: cv2.medianBlur(img, 3),
    'median_5': lambda img: cv2.medianBlur(img, 5),
    'gaussian_3': lambda img: cv2.GaussianBlur(img, (3, 3), 0),
    'gaussian_5': lambda img: cv2.GaussianBlur(img, (5, 5), 0),
    'bilateral': lambda img: cv2.bilateralFilter(img, 9, 75, 75),
}

CONTRAST_METHODS = {
    'none': None,
    'clahe': lambda img: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img),
    'hist_eq': lambda img: cv2.equalizeHist(img),
}


def apply_pipeline(image, resize_method, denoise_method, contrast_method):
    """Applying the complete preprocessing pipeline"""
    
    # Resize
    img = cv2.resize(image, TARGET_SIZE, interpolation=RESIZE_METHODS[resize_method])
    
    # Denoise
    if denoise_method != 'none' and DENOISE_METHODS[denoise_method] is not None:
        img = DENOISE_METHODS[denoise_method](img)
    
    # Contrast Enhancement
    if contrast_method != 'none' and CONTRAST_METHODS[contrast_method] is not None:
        img = CONTRAST_METHODS[contrast_method](img)
    
    return img


# Generating all combinations
combinations = list(product(RESIZE_METHODS.keys(), DENOISE_METHODS.keys(), CONTRAST_METHODS.keys()))

print(f"\nEvaluating {len(combinations)} combinations...")
print(f"  Resize methods: {len(RESIZE_METHODS)}")
print(f"  Denoise methods: {len(DENOISE_METHODS)}")
print(f"  Contrast methods: {len(CONTRAST_METHODS)}")
print(f"  Total: {len(combinations)} pipelines\n")

# all combinations
results = []

for i, (resize, denoise, contrast) in enumerate(combinations, 1):
    pipeline_name = f"{resize}+{denoise}+{contrast}"
    
    if i % 10 == 0:
        print(f"  Progress: {i}/{len(combinations)} ({i/len(combinations)*100:.1f}%)")
    
    all_metrics = []
    for img in original_images:
        try:
            processed = apply_pipeline(img, resize, denoise, contrast)
            metrics = calculate_metrics(processed)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"    Error with {pipeline_name}: {e}")
            continue
    
    if all_metrics:
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        avg_metrics['pipeline'] = pipeline_name
        avg_metrics['resize'] = resize
        avg_metrics['denoise'] = denoise
        avg_metrics['contrast'] = contrast
        results.append(avg_metrics)

print(f"\n Evaluated {len(results)} pipelines successfully")


# visuals
df = pd.DataFrame(results)

df = df.set_index('pipeline')

cat_data = df[['resize', 'denoise', 'contrast']].copy()

print("\nCalculating scores...")
numeric_cols = ['contrast_metric', 'sharpness', 'edge_strength', 'entropy', 'snr', 'dynamic_range']
df_norm = df[numeric_cols].copy()

for col in df_norm.columns:
    col_min = float(df_norm[col].min())
    col_max = float(df_norm[col].max())
    if col_max - col_min > 1e-10:
        df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
    else:
        df_norm[col] = 0.5

# Weighted scoring
weights = {
    'contrast_metric': 0.15,
    'sharpness': 0.25,
    'edge_strength': 0.25,
    'entropy': 0.20,
    'snr': 0.05,
    'dynamic_range': 0.10
}

df_norm['TOTAL_SCORE'] = sum(df_norm[col] * weights[col] for col in weights.keys())

df_norm['resize'] = cat_data['resize']
df_norm['denoise'] = cat_data['denoise']
df_norm['contrast'] = cat_data['contrast']

df_norm = df_norm.sort_values('TOTAL_SCORE', ascending=False)

print("\n" + "-"*80)
print("TOP 10 PREPROCESSING PIPELINES")
print("-"*80)
print(df_norm[['resize', 'denoise', 'contrast', 'TOTAL_SCORE']].head(10).to_string())

current_pipeline = 'cubic+median_3+clahe'
if current_pipeline in df_norm.index:
    current_rank = df_norm.index.get_loc(current_pipeline) + 1
    current_score = df_norm.loc[current_pipeline, 'TOTAL_SCORE']
    print(f"\n" + "="*80)
    print("CURRENT TECHNIQUE VALIDATION")
    print("-"*80)
    print(f"Pipeline: cubic + median_3 + clahe")
    print(f"Rank: #{current_rank} out of {len(df_norm)}")
    print(f"Score: {current_score:.4f}")
    print(f"Top Score: {df_norm.iloc[0]['TOTAL_SCORE']:.4f}")
    print(f"Difference: {abs(df_norm.iloc[0]['TOTAL_SCORE'] - current_score):.4f}")

print(f"\n" + "-"*80)
print("BEST TECHNIQUE PER CATEGORY")
print("-"*80)

print("\nBest Resize Method:")
for resize in RESIZE_METHODS.keys():
    best = df_norm[df_norm['resize'] == resize].iloc[0]
    print(f"  {resize}: {best['TOTAL_SCORE']:.4f} ({best.name})")

print("\nBest Denoise Method:")
for denoise in DENOISE_METHODS.keys():
    best = df_norm[df_norm['denoise'] == denoise].iloc[0]
    print(f"  {denoise}: {best['TOTAL_SCORE']:.4f} ({best.name})")

print("\nBest Contrast Method:")
for contrast in CONTRAST_METHODS.keys():
    best = df_norm[df_norm['contrast'] == contrast].iloc[0]
    print(f"  {contrast}: {best['TOTAL_SCORE']:.4f} ({best.name})")

# Recommendations
print(f"\n" + "-"*80)
print("FINAL RECOMMENDATION")
print("-"*80)

best_pipeline = df_norm.iloc[0]
print(f"\nOptimal Pipeline:")
print(f"  Resize: {best_pipeline['resize']}")
print(f"  Denoise: {best_pipeline['denoise']}")
print(f"  Contrast: {best_pipeline['contrast']}")
print(f"  Score: {best_pipeline['TOTAL_SCORE']:.4f}")


os.makedirs('../pre-processing/results', exist_ok=True)
df.to_csv('../pre-processing/results/comprehensive_metrics.csv')
df_norm.to_csv('../pre-processing/results/comprehensive_scores.csv')
print(f"\n Results saved to results/comprehensive_*.csv")

fig = plt.figure(figsize=(16, 8))
fig.suptitle('Comprehensive Preprocessing Evaluation', fontsize=16, fontweight='bold')

# Plot 1: Top 15 Pipelines
ax1 = plt.subplot(2, 2, 1)
top_15 = df_norm.head(15)
colors = ['green' if idx == current_pipeline else 'steelblue' for idx in top_15.index]
ax1.barh(range(len(top_15)), top_15['TOTAL_SCORE'].values, color=colors)
ax1.set_yticks(range(len(top_15)))
ax1.set_yticklabels([name[:30] for name in top_15.index], fontsize=8)
ax1.set_xlabel('Score', fontsize=10)
ax1.set_title('Top 15 Pipelines', fontsize=12, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Resize Method Comparison
ax2 = plt.subplot(2, 2, 2)
resize_scores = df_norm.groupby('resize')['TOTAL_SCORE'].max().sort_values(ascending=False)
ax2.bar(range(len(resize_scores)), resize_scores.values, color='coral', alpha=0.7, edgecolor='black')
ax2.set_xticks(range(len(resize_scores)))
ax2.set_xticklabels(resize_scores.index, rotation=45, fontsize=9)
ax2.set_ylabel('Best Score', fontsize=10)
ax2.set_title('Best Score by Resize Method', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Denoise Method Comparison
ax3 = plt.subplot(2, 2, 3)
denoise_scores = df_norm.groupby('denoise')['TOTAL_SCORE'].max().sort_values(ascending=False)
ax3.bar(range(len(denoise_scores)), denoise_scores.values, color='lightgreen', alpha=0.7, edgecolor='black')
ax3.set_xticks(range(len(denoise_scores)))
ax3.set_xticklabels(denoise_scores.index, rotation=45, ha='right', fontsize=9)
ax3.set_ylabel('Best Score', fontsize=10)
ax3.set_title('Best Score by Denoise Method', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Contrast Method Comparison
ax4 = plt.subplot(2, 2, 4)
contrast_scores = df_norm.groupby('contrast')['TOTAL_SCORE'].max().sort_values(ascending=False)
ax4.bar(range(len(contrast_scores)), contrast_scores.values, color='lightyellow', alpha=0.7, edgecolor='black')
ax4.set_xticks(range(len(contrast_scores)))
ax4.set_xticklabels(contrast_scores.index, rotation=45, fontsize=9)
ax4.set_ylabel('Best Score', fontsize=10)
ax4.set_title('Best Score by Contrast Method', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
os.makedirs('../pre-processing/results', exist_ok=True)
plt.savefig('../pre-processing/results/comprehensive_evaluation.png', dpi=150, bbox_inches='tight')
print(f" Visualization saved: pre-processing/results/comprehensive_evaluation.png")

print("\n" + "-"*80)
print("EVALUATION COMPLETE")
print("-"*80)
