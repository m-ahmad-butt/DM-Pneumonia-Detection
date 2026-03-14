# X-Ray Image Preprocessing — Complete Guide
### Simple and Detailed Explanation of Every Technique

## Table of Contents

1. Normalization (Min-Max and Z-score)
2. Denoising Techniques (NLM, Bilateral, Gaussian, Median, Mean)
3. Sharpening Techniques (Unsharp Masking, Laplacian)
4. Contrast Enhancement (CLAHE, Histogram Equalization)
5. Resizing Interpolation (Nearest, Linear, Cubic, Area)
6. Intensity Transforms (Gamma Correction, Brightness)
7. Final Rankings Summary

# PART 1 — NORMALIZATION

## Understanding the Concept First

X-ray images contain pixels with values ranging from 0 to 255 (in an 8-bit image). These values indicate how bright or dark a specific area is. The purpose of normalization is to bring these values onto a common scale so that different images can be compared or fed into a model.

## 1.1 Min-Max Normalization

### What it Does

It compresses every pixel value between 0 and 1. The formula is:

```
New Value = (Original - Minimum) / (Maximum - Minimum)
```

This means the darkest pixel becomes 0, the brightest pixel becomes 1, and everything else falls in between.

### What is the Problem?

If the image contains something extremely bright—like a surgical clip or a metal artifact—it becomes the Maximum. All other pixels then appear small and compressed relative to that one bright object. In the histogram, it was observed that useful tissue values were only between 0.4 and 0.8—using only half of the total range.

### When to Use

In simple cases where there are no extreme outliers in the image. For mathematical simplicity.

### When to Avoid

When the image contains bright artifacts (surgical clips, pacemakers). Often avoided in medical imaging.

## 1.2 Z-Score Normalization (Standardization)

### What it Does

It centers pixel values around the mean (average) and scales them based on the standard deviation:

```
New Value = (Original - Mean) / Standard Deviation
```

The result typically ranges around -2 to +2.

### Why it is Better

- Outliers don't have as much impact because the scale isn't fixed.
- It provides a zero-centered output, which deep learning models prefer.
- Tissue contrast is better spread out.

### When to Use

For deep learning pipelines—it's almost standard in medical imaging. It also works well even when the image contains metal artifacts.

### When to Avoid

When an image is needed for display (where a 0-255 range is required), it cannot be used directly—it must be converted.

# PART 2 — DENOISING TECHNIQUES

## Understand First: What is Noise?

Noise in an X-ray image refers to random variations that don't come from actual tissue but from the limitations of the X-ray detector. It looks like a grainy texture. The problem is that if real tissue information is destroyed while removing noise, the diagnosis could be incorrect.

## 2.1 Non-Local Means (NLM) Denoising

### What it Does

It searches for similar patches (small regions) throughout the entire image. When denoising a pixel, it doesn't just take the average of adjacent pixels—it averages patches that look similar across the whole image.

### What Happened in the Histogram?

The histogram became completely spiky and jagged—instead of original smooth curves, there were sharp spikes and gaps between them. This happened because:

```
Before: pixel 130 = 150 count
        pixel 131 = 148 count    (smooth gradation)
        pixel 132 = 152 count

After:  pixel 130 = 800 count    (spike)
        pixel 131 =   5 count    (gap)
        pixel 132 = 750 count    (spike)
```

### What is the Problem?

There is often only a difference of 3 to 5 intensity units between early pneumonia and healthy lung tissue. After NLM, this difference disappears because similar pixels cluster around a single value. That subtle difference—which was essential for diagnosis—is mathematically destroyed.

### Use it or Avoid it?

Avoid it for X-rays. The image looks visually clean, but diagnostic information is lost. This is a dangerous tradeoff.

## 2.2 Bilateral Filter

### What it Does

It considers two things simultaneously:

- **Spatial distance**: How far the neighbor pixel is.
- **Intensity difference**: How different the color/brightness of the neighbor pixel is.

If a neighbor's intensity is very different (indicating an edge), it is not included in the average. Therefore, edges are preserved.

### What Happened in the Histogram?

The shape remained similar to the original. There were some mild peaks, but no large gaps or extreme spikes occurred. The 0-200 range was preserved.

### When to Use

When you want to remove noise while preserving edges. The bilateral filter is a reasonable choice for X-rays, but parameters must be carefully tuned.

### When to Avoid

If the parameters are too aggressive, a "cartoon effect" can occur—flat regions become very flat and edges become very sharp. This is also unnatural.

## 2.3 Gaussian Blur

### What it Does

The center pixel is most important, and other neighbors become less important based on distance (bell curve / Gaussian weighting).

```
Weights:  1  2  1
          2  4  2   (center 4 = highest weight)
          1  2  1
```

### What Happened in the Histogram?

The peak dropped slightly (from 18,000 to 16,000), and the shape remained smooth. No drastic distortion occurred.

### When to Use

For mild noise removal. Generally safe in pre-processing.

### When to Avoid

Edges become softened—rib boundaries might blur slightly. Avoid if edge clarity is essential.

## 2.4 Median Blur — BEST for X-rays

### What it Does

It selects the median value from each pixel's neighborhood—no averaging, just the middle value.

```
Neighborhood: [130, 132, 131, 200, 129, 131, 130]
                                 ^ this is a noise spike
Mean gives:   140.4  (polluted by noise)
Median gives: 131    (noise is ignored)
```

### Why it is Best

- It rejects noise without creating fake values.
- It only uses values that already exist in the image.
- Edges remain sharp.
- The histogram stays closest to the original.

### When to Use

First choice for X-ray denoising. Especially effective for salt and pepper noise. A medical imaging gold standard.

### When to Avoid

In images with very fine detail, detail may be lost if the kernel size is too large. Use a small kernel size.

## 2.5 Mean Blur (Average Blur)

### What it Does

It takes a simple average of all pixels in each pixel's neighborhood—no weighting, no edge consideration, just a straightforward average.

### What Happened in the Histogram?

The histogram became fragmented and jagged—a problem similar to NLM. Bright lung fields bleed into darker regions.

### When to Use

Almost never for medical images. Only in very basic/fast use cases where quality doesn't matter much.

### When to Avoid

Avoid for X-rays. There is no edge awareness. Everything is averaged indiscriminately.

# PART 3 — SHARPENING TECHNIQUES

## Understanding the Concept

Sharpening is the opposite of denoising—it amplifies high-frequency information (edges, details). The problem is that noise is also a high-frequency signal—so sharpening amplifies both noise and real edges, and neither the model nor the radiologist can tell what is real and what is a noise-amplified artifact.

## 3.1 Unsharp Masking

### What it Does

```
Sharpened = Original + alpha × (Original - Blurred version)
```

This means it subtracts the blurred version to find differences and then adds them back to the original. It's called "unsharp" because a blurred (unsharp) copy is used internally.

### What Happened in the Histogram?

The most distortion among all techniques. Spikes went up to 20,000+. Gaps were severe. The image appeared darker and over-contrasted.

```
What happened:
Pixel 149 pushed → 150
Pixel 151 pushed → 150
Result: 3x pixels at intensity 150 = spike
Intensities 149, 151 almost empty = gap
```

### When to Avoid

Avoid for X-rays. Histogram destruction is most extreme. Edge halos are created (a bright ring around the rib) that isn't real.

## 3.2 Laplacian Sharpening

### What it Does

It computes the second derivative of the image—detecting the 'change of change', which is high at edge locations. This edge information is then added to the original.

### What Happened in the Histogram?

The spikes were moderate (max ~14,000 vs Unsharp Masking's 20,000+). The original shape was partially preserved.

### When to Use

If sharpening is necessary, it's better than Unsharp Masking. However, keep it very carefully at a low strength.

### When to Avoid

Avoid with aggressive settings. It can highlight noise as if it were real edges.

# PART 4 — CONTRAST ENHANCEMENT

## Understand First: Contrast Enhancement vs Denoising

These techniques don't change the spatial arrangement of pixel values—they only redistribute the intensity distribution to improve contrast in the image. This is a different category.

## 4.1 Histogram Equalization

### What it Does

It forcibly spreads all pixel values across the entire 0-255 range. The goal is to make the histogram flat—so that there are roughly equal numbers of pixels at every intensity level.

### What Happened in the Histogram?

The most destructive enhancement technique:

- The range increased from 0-200 to 0-255—**new intensity values were invented that were not in the original at all.**
- Massive spikes (17,000+) with deep empty gaps.
- False textures appeared in the background corners.

```
In the original:
- Pixels were distributed between 0-200
- There were no pixels in 201-255

After HE:
- Pixels were forcibly spread up to 255
- Values were invented
```

### Why it is Dangerous

This is a fundamental problem: histogram equalization creates new intensity values that never existed in the image. Tissue densities that had no existence are artificially created. What the radiologist is seeing is not real anatomy.

### When to Avoid

Almost always avoid in medical imaging. Absolutely not for diagnostic X-rays.

## 4.2 CLAHE — BEST Contrast Enhancement

### What it Does

CLAHE = Contrast Limited Adaptive Histogram Equalization. Three important things:

**Adaptive**: The whole image isn't processed at once. It divides the image into small tiles and equalizes each tile's histogram separately.

**Contrast Limited**: A clip limit is set. If too many pixels fall into one intensity level (starting to form a spike), it clips them and redistributes them into neighboring intensities.

**Interpolation**: Smooth blending happens at tile borders so that seams are not visible.

### What Happened in the Histogram?

- The 0-200 range was preserved—no fake values were created.
- The shape became broader and smoother—better contrast.
- Only two small spikes around intensity 45 (which is acceptable).
- The overall distribution is more informative.

### Why it is Best

```
Dark region (possible pneumonia):
CLAHE brightens this locally    → pathology becomes visible
Surrounding bright bone:
CLAHE handles this in a different tile → no over-exposure
    
Regular HE:
Handles both globally → one gets re-exposed, the other is destroyed
```

### When to Use

CLAHE is the standard for medical imaging. It's used as preprocessing in almost all X-ray AI models. It's also best for visual inspection.

### When to Avoid

A cartoon effect can occur with very small tile sizes and aggressive settings. Default parameters are usually safe.

# PART 5 — RESIZING INTERPOLATION

## Understand First: Why is Resizing Necessary?

Deep learning models expect a fixed input size (like 128x128, 224x224, 512x512). Actual X-ray images come in various sizes. They must be resized. However, when pixels are resized, in-between values must be calculated—this is called interpolation.

Note: After resizing, the pixel count decreases significantly, so the histogram's y-axis also drops proportionally. This is normal—the shape is more important than the count.

## 5.1 Nearest Neighbor

### What it Does

Very straightforward: whatever original pixel is closest, take its value. No calculation, no averaging.

### What Happened in the Histogram?

Cleanest histogram—only original values, no new interpolated values.

### When to Use

For segmentation masks and label images. When the image contains class labels (pixel value = class number), they cannot be blended. Nearest neighbor is the only option.

### When to Avoid

For natural images, it results in a blocky appearance at low resolutions.

## 5.2 Bilinear (Linear) Interpolation

### What it Does

It takes a weighted average of the 4 nearest pixels. Weights are assigned based on distance.

### What Happened in the Histogram?

Smooth, close to the original. Some interpolated values are created, but they are controlled.

### When to Use

The default and safe choice for general-purpose deep learning pipelines. A good balance of speed and quality.

### When to Avoid

When maximum sharpness is required—cubic produces a slightly softer output.

## 5.3 Bicubic (Cubic) Interpolation

### What it Does

A weighted average of the 16 nearest pixels. It uses a cubic polynomial—more complex math, smoother results.

### What Happened in the Histogram?

Even better preserved than linear. Rib edges and lung markings remained sharper.

### When to Use

The best choice for medical images. When diagnostic quality matters. Most frameworks (PyTorch, OpenCV) recommend this for medical imaging.

### When to Avoid

It's slightly slower than linear. Minor "ringing" artifacts may appear on high-contrast edges, but they are usually negligible.

## 5.4 Area Interpolation

### What it Does

It takes the average of all original pixels that fall within the source area corresponding to the destination pixel. It is a mathematically correct anti-aliasing method for strong downscaling.

### What Happened in the Histogram?

Fragmented and spiky—alternating spikes and gaps due to uneven averaging.

### When to Use

Mathematically, it's technically best for very aggressive downscaling. For thumbnail generation in computer vision.

### When to Avoid

Avoid for medical images due to histogram distortion. Diagnostic information is affected unevenly.

# PART 6 — INTENSITY TRANSFORMS

## Understand First

These transforms don't look at a pixel's neighbors—they only change the pixel's own value. No spatial blurring or averaging occurs. However, intensity values change, so the histogram shifts.

## 6.1 Gamma Correction

### What it Does

```
Output = (Input / 255) ^ gamma × 255
```

- gamma < 1 : image brightens (shadows are lifted)
- gamma > 1 : image darkens (highlights are compressed)
- gamma = 1 : no change

It's a non-linear transform—dark and bright pixels change in different proportions.

### What Happened in the Histogram?

The range was compressed (from 0-200 to 0-160). The y-axis went up to 35,000—meaning pixels became denser in specific bins. Some isolated spikes were visible.

### When to Use

Use gamma values close to 1.0 (like 0.8 or 1.2) for mild adjustment. When you need to adjust overall brightness without clipping.

### When to Avoid

Aggressive gamma values can crush shadows or lose highlights. Avoid gamma > 1.5 or < 0.5 in diagnostic images.

## 6.2 Brightness Increase (Adding Constant)

### What it Does

It adds a constant to every pixel:

```
Output = Input + 50  (for example)
```

The entire histogram shifts to the right.

### The Biggest Problem: CLIPPING

```
Original pixel = 210
Add 50 = 260
But the maximum is 255
Result = 255 (clipped)

Original pixel = 220 → 270 → 255 (clipped)
Original pixel = 230 → 280 → 255 (clipped)
Original pixel = 240 → 290 → 255 (clipped)

All different tissue densities pile up at the same value
```

A massive spike of 28,000+ pixels was seen at 255 in the histogram.

### What was Destroyed?

```
Healthy lung:         intensity 160
Mild consolidation:   intensity 175   (15 unit difference — detectable)
Dense consolidation:  intensity 190   (30 unit difference — detectable)

After +70 brightness:
Healthy lung:         230
Mild consolidation:   245
Dense consolidation:  255 (clipped)

The 30-unit difference that detected real pathology—gone
```

### When to Use

Only when you are absolutely sure that no pixel will reach 255. In practice, almost never safely in medical images.

### When to Avoid

Whenever there are bright regions in the image. Clipped data is permanently lost—no post-processing can recover it.

# PART 7 — FINAL RANKINGS

## Best Choice in Every Category

### Denoising
```
1st: Median Blur       - safest, preserves edges, histogram intact
2nd: Bilateral Filter  - edge-aware, acceptable distortion
3rd: Gaussian Blur     - mild, ok for pre-processing
Avoid: Mean Blur       - no edge awareness
Avoid: NLM             - destroys subtle gradations
```

### Contrast Enhancement
```
1st: CLAHE             - gold standard, local adaptive, no fake values
Avoid: Histogram EQ    - invents fake intensity values, clinically unsafe
```

### Sharpening
```
Use carefully: Laplacian  - mild distortion, low strength only
Avoid: Unsharp Masking    - worst histogram destruction of all techniques
```

### Resizing
```
Best quality: Cubic (Bicubic)    - for diagnostic images
General use:  Bilinear           - fast and clean
Labels/masks: Nearest Neighbor   - only option that doesn't blend classes
Avoid:        Area               - histogram inconsistency
```

### Normalization
```
For ML:     Z-score              - zero-centered, outlier-robust
Simple:     Min-Max              - ok without outliers
```

### Intensity Transforms
```
Careful use: Gamma (near 1.0)    - gentle non-linear adjustment
Avoid:       Brightness +const   - clipping destroys highlights permanently
```

## The One Golden Rule

> **Any operation that creates clipping at 0 or 255 in the histogram, or starts creating artificial gaps and spikes, is destroying diagnostic information—and this loss is permanent.**

Whenever you apply a technique, look at the histogram first. If, instead of smooth original curves, you see:
- Extremely tall spikes
- Deep empty gaps
- Range extension (new values)
- Range compression (losing values)

...know that this technique has destroyed real tissue information.

*Document covers: Normalization, Denoising, Sharpening, Contrast Enhancement, Resizing, Intensity Transforms for chest X-ray preprocessing in medical imaging and deep learning pipelines.*