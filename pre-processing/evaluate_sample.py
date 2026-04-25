import cv2
import numpy as np
import os
from pathlib import Path
from tabulate import tabulate

# CONFIGURATION
SCRIPT_DIR = Path(__file__).resolve().parent
SAMPLE_IMAGE_PATH = str((SCRIPT_DIR / '../chest_xray/test/NORMAL/IM-0001-0001.jpeg').resolve())
OUTPUT_DIR = str(SCRIPT_DIR / 'results')
OUTPUT_TABLE_FILE = str(SCRIPT_DIR / 'results/sample_evaluation_table.txt')

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n{'-'*70}")
print("PREPROCESSING EVALUATION - SINGLE SAMPLE IMAGE")
print(f"{'-'*70}")

# Load sample image
print(f"\nLoading sample image...")
original_img = cv2.imread(SAMPLE_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if original_img is None:
    print(f"ERROR: Could not load image from {SAMPLE_IMAGE_PATH}")
    exit(1)

print(f"Image loaded: {original_img.shape}")


# EVALUATION METRICS
def calculate_contrast(image):
    """Standard deviation of pixel intensities"""
    return np.std(image)


def calculate_sharpness(image):
    """Laplacian variance - measures edge sharpness"""
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return laplacian.var()


def calculate_edge_strength(image):
    """Sobel edge detection strength"""
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return edge_magnitude.mean()


def calculate_entropy(image):
    """Shannon entropy - information content"""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def calculate_snr(image):
    """Signal-to-Noise Ratio"""
    signal = np.mean(image)
    noise = np.std(image)
    return signal / noise if noise > 0 else 0


def calculate_dynamic_range(image):
    """Range of pixel intensities"""
    return np.max(image) - np.min(image)


def evaluate_image(image, name):
    """Calculate all 6 metrics for an image"""
    return {
        'Technique': name,
        'Contrast': calculate_contrast(image),
        'Sharpness': calculate_sharpness(image),
        'Edge Strength': calculate_edge_strength(image),
        'Entropy': calculate_entropy(image),
        'SNR': calculate_snr(image),
        'Dynamic Range': calculate_dynamic_range(image)
    }


# PREPROCESSING TECHNIQUES

# 1. BLUR/SMOOTHING FILTERS
def apply_gaussian_blur(image):
    """Gaussian blur to reduce noise"""
    return cv2.GaussianBlur(image, (5, 5), 0)

def apply_median_blur(image):
    """Median blur to remove salt-and-pepper noise"""
    return cv2.medianBlur(image, 5)

def apply_mean_blur(image):
    """Mean/Average blur for smoothing"""
    return cv2.blur(image, (5, 5))

def apply_bilateral_filter(image):
    """Bilateral filter - edge-preserving smoothing"""
    return cv2.bilateralFilter(image, 9, 75, 75)


# 2. CONTRAST ENHANCEMENT
def apply_histogram_equalization(image):
    """Histogram equalization to improve contrast"""
    return cv2.equalizeHist(image)

def apply_clahe(image):
    """CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


# 3. NORMALIZATION TECHNIQUES
def apply_zscore_normalization(image):
    """Z-score normalization (standardization)"""
    mean = np.mean(image)
    std = np.std(image)
    if std > 0:
        normalized = (image - mean) / std
        # Scale back to 0-255 range
        normalized = ((normalized - normalized.min()) / (normalized.max() - normalized.min()) * 255)
        return normalized.astype(np.uint8)
    return image

def apply_minmax_normalization(image):
    """Min-Max normalization (0-255 range)"""
    normalized = ((image - image.min()) / (image.max() - image.min()) * 255)
    return normalized.astype(np.uint8)


# 4. INTERPOLATION METHODS (for resizing)
def apply_area_interpolation(image):
    """Area interpolation - best for downsampling"""
    # Resize up then down to show effect
    temp = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    return cv2.resize(temp, (128, 128), interpolation=cv2.INTER_AREA)

def apply_cubic_interpolation(image):
    """Cubic interpolation - smooth results"""
    temp = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(temp, (128, 128), interpolation=cv2.INTER_CUBIC)

def apply_nearest_interpolation(image):
    """Nearest neighbor interpolation - fastest but blocky"""
    temp = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(temp, (128, 128), interpolation=cv2.INTER_NEAREST)

def apply_linear_interpolation(image):
    """Linear interpolation - balanced speed and quality"""
    temp = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (128, 128), interpolation=cv2.INTER_LINEAR)


# Resize to standard size
TARGET_SIZE = (128, 128)
original_resized = cv2.resize(original_img, TARGET_SIZE)

print(f"\nEvaluating preprocessing techniques...")

# Evaluate original (before preprocessing)
original_metrics = evaluate_image(original_resized, 'Original')

# Apply and evaluate each preprocessing technique
techniques = [
    # BLUR/SMOOTHING FILTERS
    ('Gaussian Blur', apply_gaussian_blur),
    ('Median Blur', apply_median_blur),
    ('Mean Blur', apply_mean_blur),
    ('Bilateral Filter', apply_bilateral_filter),
    
    # CONTRAST ENHANCEMENT
    ('Histogram Equalization', apply_histogram_equalization),
    ('CLAHE', apply_clahe),
    
    # NORMALIZATION
    ('Z-Score Normalization', apply_zscore_normalization),
    ('Min-Max Normalization', apply_minmax_normalization),
    
    # INTERPOLATION METHODS
    ('Area Interpolation', apply_area_interpolation),
    ('Cubic Interpolation', apply_cubic_interpolation),
    ('Nearest Interpolation', apply_nearest_interpolation),
    ('Linear Interpolation', apply_linear_interpolation)
]

# Store all results for table
all_results = []

# Evaluate each technique
for name, func in techniques:
    # Apply preprocessing
    processed = func(original_resized)
    processed_metrics = evaluate_image(processed, name)
    
    # Create before/after comparison
    result = {
        'Technique': name,
        'Metric': [],
        'Before': [],
        'After': [],
        'Change': [],
        'Change %': []
    }
    
    for metric in ['Contrast', 'Sharpness', 'Edge Strength', 'Entropy', 'SNR', 'Dynamic Range']:
        before = original_metrics[metric]
        after = processed_metrics[metric]
        change = after - before
        percent = (change / before * 100) if before > 0 else 0
        
        result['Metric'].append(metric)
        result['Before'].append(f"{before:.2f}")
        result['After'].append(f"{after:.2f}")
        result['Change'].append(f"{change:+.2f}")
        result['Change %'].append(f"{percent:+.1f}%")
    
    all_results.append(result)

# Print and save results
output_text = []
output_text.append("="*90)
output_text.append("PREPROCESSING EVALUATION - BEFORE vs AFTER")
output_text.append("="*90)
output_text.append("")

print(f"\n{'='*90}")
print("PREPROCESSING EVALUATION - BEFORE vs AFTER")
print(f"{'='*90}\n")

for result in all_results:
    technique_name = result['Technique']
    
    # Create table data
    table_data = []
    for i in range(len(result['Metric'])):
        table_data.append([
            result['Metric'][i],
            result['Before'][i],
            result['After'][i],
            result['Change'][i],
            result['Change %'][i]
        ])
    
    # Print table
    print(f"\nTECHNIQUE: {technique_name}")
    print("-" * 90)
    table_str = tabulate(
        table_data,
        headers=['Metric', 'Before', 'After', 'Change', 'Change %'],
        tablefmt='grid',
        stralign='left',
        numalign='right'
    )
    print(table_str)
    
    # Save to output text
    output_text.append(f"\nTECHNIQUE: {technique_name}")
    output_text.append("-" * 90)
    output_text.append(table_str)
    output_text.append("")

# Create summary table comparing all techniques
print(f"\n{'='*90}")
print("SUMMARY - ALL TECHNIQUES COMPARISON")
print(f"{'='*90}\n")

summary_data = []
for result in all_results:
    row = [result['Technique']]
    # Add "After" values for each metric
    for i in range(len(result['Metric'])):
        row.append(result['After'][i])
    summary_data.append(row)

# Add original as first row
original_row = ['Original (Before)']
for metric in ['Contrast', 'Sharpness', 'Edge Strength', 'Entropy', 'SNR', 'Dynamic Range']:
    original_row.append(f"{original_metrics[metric]:.2f}")
summary_data.insert(0, original_row)

summary_table = tabulate(
    summary_data,
    headers=['Technique', 'Contrast', 'Sharpness', 'Edge Str.', 'Entropy', 'SNR', 'Dyn. Range'],
    tablefmt='grid',
    stralign='left',
    numalign='right'
)
print(summary_table)

output_text.append("\n" + "="*90)
output_text.append("SUMMARY - ALL TECHNIQUES COMPARISON")
output_text.append("="*90)
output_text.append(summary_table)

# Save to file
with open(OUTPUT_TABLE_FILE, 'w') as f:
    f.write('\n'.join(output_text))

print(f"\n{'='*90}")
print("EVALUATION COMPLETE")
print(f"{'='*90}")
print(f"\nResults saved to: {OUTPUT_TABLE_FILE}")
print(f"You can copy the tables to your slides!")
