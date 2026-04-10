# Preprocessing Evaluation Metrics
---

## 1. CONTRAST (Standard Deviation)

### What it measures:
The spread of pixel intensity values in the image.

### Formula:
```
Contrast = Standard Deviation of pixel values
         = sqrt(mean((pixel - mean)²))
```

### Why we use it:
- **Higher contrast = Better feature distinction**
- In X-rays, we need to distinguish between:
  - Lung tissue (darker)
  - Bones (brighter)
  - Pneumonia infiltrates (medium gray)
- Low contrast = Everything looks similar (bad!)
- High contrast = Clear differences between structures (good!)

### Example:
```
Low Contrast Image:  All pixels between 100-120 → std = 5.7 (BAD)
High Contrast Image: Pixels range 0-255 → std = 73.5 (GOOD)
```

### What we want:
**HIGHER standard deviation = BETTER contrast**

---

## 2. SHARPNESS (Laplacian Variance)

### What it measures:
How well-defined the edges and details are in the image.

### Formula:
```
Laplacian = Second derivative of image
          = Detects rapid intensity changes
Sharpness = Variance of Laplacian values
```

### How it works:
The Laplacian operator detects edges by finding where pixel values change rapidly:
```
Laplacian kernel:
[ 0  1  0]
[ 1 -4  1]
[ 0  1  0]
```

### Why we use it:
- **Higher variance = Sharper edges**
- Sharp images show:
  - Clear lung boundaries
  - Distinct rib edges
  - Well-defined infiltrates
- Blurry images have low Laplacian variance
- Sharp images have high Laplacian variance

### Example:
```
Blurry Image:  Laplacian variance = 150 (BAD)
Sharp Image:   Laplacian variance = 2500 (GOOD)
```

### What we want:
**HIGHER Laplacian variance = SHARPER image**

---

## 3. EDGE STRENGTH (Sobel Gradient)

### What it measures:
The magnitude of edges in the image (how strong the boundaries are).

### Formula:
```
Sobel X = Horizontal edges (left-right changes)
Sobel Y = Vertical edges (top-bottom changes)
Edge Strength = mean(sqrt(Sobel_X² + Sobel_Y²))
```

### How it works:
Sobel operators detect edges in X and Y directions:
```
Sobel X:           Sobel Y:
[-1  0  1]         [-1 -2 -1]
[-2  0  2]         [ 0  0  0]
[-1  0  1]         [ 1  2  1]
```

### Why we use it:
- **Higher edge strength = Better structure visibility**
- Strong edges show:
  - Lung boundaries
  - Rib cage structure
  - Heart outline
  - Pneumonia regions
- Weak edges = Hard to see anatomical structures

### Example:
```
Weak Edges:    Mean edge strength = 45 (BAD)
Strong Edges:  Mean edge strength = 138 (GOOD)
```

### What we want:
**HIGHER edge strength = BETTER structure visibility**

---

## 4. ENTROPY (Information Content)

### What it measures:
How much information/randomness is in the image.

### Formula:
```
Entropy = -Σ(p(i) × log₂(p(i)))

where:
- p(i) = probability of pixel value i
- Σ = sum over all 256 possible values (0-255)
```

### How it works:
- If all pixels are the same value → Entropy = 0 (no information)
- If pixels are uniformly distributed → Entropy = 8 (maximum information)

### Why we use it:
- **Higher entropy = More information preserved**
- High entropy means:
  - Many different pixel values
  - Rich texture details
  - Preserved anatomical information
- Low entropy means:
  - Few distinct values
  - Lost details
  - Over-processed image

### Example:
```
Flat image (all pixels = 128):     Entropy = 0.0 (BAD)
Uniform distribution (0-255):      Entropy = 8.0 (PERFECT)
Good X-ray with details:           Entropy = 6.5-7.5 (GOOD)
Over-processed (only 10 values):   Entropy = 3.3 (BAD)
```

### What we want:
**HIGHER entropy = MORE information preserved**

---

## 5. SIGNAL-TO-NOISE RATIO (SNR)

### What it measures:
The ratio of meaningful signal (image content) to noise (random variations).

### Formula:
```
SNR = Mean Signal / Standard Deviation (Noise)
    = mean(pixel values) / std(pixel values)
```

### Why we use it:
- **Higher SNR = Less noise, clearer image**
- High SNR means:
  - Clean image
  - Less random noise
  - Better for CNN to learn patterns
- Low SNR means:
  - Noisy image
  - Random variations
  - Harder to detect pneumonia

### Example:
```
Noisy Image:  Mean=120, Std=45 → SNR = 2.67 (BAD)
Clean Image:  Mean=120, Std=15 → SNR = 8.00 (GOOD)
```

### Important Note:
SNR can be misleading! A flat image (no details) has high SNR but is useless.
That's why we use it WITH other metrics.

### What we want:
**HIGHER SNR = LESS noise** (but balance with other metrics)

---

## 6. DYNAMIC RANGE

### What it measures:
The range of intensity values from darkest to brightest pixel. Measures how much of the available intensity scale (0-255) your image is using.
### Formula:
```
Dynamic Range = max(pixel values) - min(pixel values)
```

#### Visual Example:

```
Image A (Poor Dynamic Range):
Pixel values: [80, 85, 90, 95, 100, 105, 110, 115, 120]
Min = 80, Max = 120
Dynamic Range = 120 - 80 = 40

Only using 40 out of 255 possible values (15.7% of scale)
Wasted potential!

Histogram:
0                                                           255
|                    ████████                                |
                     80    120
                     
Image looks FLAT and GRAY - hard to see details!
```

```
Image B (Good Dynamic Range):
Pixel values: [0, 30, 60, 90, 120, 150, 180, 210, 240, 255]
Min = 0, Max = 255
Dynamic Range = 255 - 0 = 255

Using FULL 255 values (100% of scale)
Maximum potential!

Histogram:
0                                                           255
|████████████████████████████████████████████████████████████|
0                                                           255

Image has FULL range - easy to see details!
```

#### Why This Matters for X-rays:

```
Poor Dynamic Range X-ray:
- Lungs: gray (100)
- Ribs: slightly lighter gray (110)
- Pneumonia: medium gray (105)
→ HARD TO DISTINGUISH! All look similar!

Good Dynamic Range X-ray:
- Lungs: dark (50)
- Ribs: bright (200)
- Pneumonia: medium (120)
→ EASY TO DISTINGUISH! Clear differences!
```

## Scoring System

### Weights (Total = 1.0):
```
Sharpness:      20%  (Most important - need clear edges)
Edge Strength:  20%  (Most important - anatomical structures)
Entropy:        20%  (Most important - information preservation)
Contrast:       15%  (Important - feature distinction)
SNR:            10%  (Important - noise reduction)
Dynamic Range:  10%  (Important - tonal range)
Hist Uniformity: 5%  (Less important - distribution)
```

### Calculation:
```
For each metric:
1. Normalize to 0-1 scale:
   normalized = (value - min) / (max - min)

2. Apply weight:
   weighted_score = normalized × weight

3. Sum all weighted scores:
   TOTAL_SCORE = Σ(weighted_scores)
```

### Example:
```
Technique: Median + CLAHE
- Sharpness:     0.85 × 0.20 = 0.170
- Edge Strength: 0.92 × 0.20 = 0.184
- Entropy:       0.78 × 0.20 = 0.156
- Contrast:      0.88 × 0.15 = 0.132
- SNR:           0.65 × 0.10 = 0.065
- Dynamic Range: 0.95 × 0.10 = 0.095
- Hist Uniform:  0.45 × 0.05 = 0.023
                        TOTAL = 0.825
```

---

## Visual Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    METRIC PURPOSES                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CONTRAST (Std Dev)     → Can we distinguish features?      │
│  SHARPNESS (Laplacian)  → Are edges well-defined?           │
│  EDGE STRENGTH (Sobel)  → Are structures visible?           │
│  ENTROPY (Information)  → Is detail preserved?              │
│  SNR (Signal/Noise)     → Is image clean?                   │
│  DYNAMIC RANGE          → Full tonal variation?             │
│                                                             │
│  ALL TOGETHER → Best preprocessing for CNN training         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

