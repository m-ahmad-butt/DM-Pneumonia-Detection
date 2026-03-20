import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dotenv import load_dotenv

load_dotenv()

DATASET_DIR   = os.getenv('Project_PATH', '../chest_xray/train') + '/train/' if os.getenv('Project_PATH') else '../chest_xray/train/'
NORMAL_DIR    = os.path.join(DATASET_DIR, 'NORMAL')
SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
TARGET_SIZE   = (256, 256)
RESULTS_DIR   = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Use FILE_NAME from .env if set, otherwise fall back to first available image
_env_file = os.getenv('FILE_NAME')
if _env_file and os.path.exists(_env_file):
    FILE_NAME = _env_file
else:
    normal_files = [f for f in os.listdir(NORMAL_DIR) if f.lower().endswith(SUPPORTED_EXT)]
    FILE_NAME    = os.path.join(NORMAL_DIR, normal_files[0])



def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR: Cannot read '{path}'.")
        sys.exit(1)
    return img


def plot_image_histogram(ax_img, ax_hist, image, title, cmap="gray"):
    ax_img.imshow(image, cmap=cmap, aspect="auto")
    ax_img.set_title(title, fontsize=9, fontweight="bold", pad=3)
    ax_img.axis("off")

    flat = image.ravel().astype(np.float32)
    ax_hist.hist(flat, bins=100, color="#2c7bb6", alpha=0.8, linewidth=0)
    ax_hist.set_xlabel("Pixel Value", fontsize=7)
    ax_hist.set_ylabel("Count", fontsize=7)
    ax_hist.tick_params(labelsize=6)
    ax_hist.spines[["top", "right"]].set_visible(False)


def build_figure(category_name, variants, original_img, save_name):
    cols  = 1 + len(variants)
    fig_w = cols * 3.2
    fig_h = 5.5

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(f"Technique Comparison -- {category_name}", fontsize=12,
                 fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(2, cols, figure=fig,
                           height_ratios=[3, 1.5],
                           hspace=0.45, wspace=0.35)

    plot_image_histogram(
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        original_img, "Original"
    )

    for col_idx, (label, img) in enumerate(variants, start=1):
        plot_image_histogram(
            fig.add_subplot(gs[0, col_idx]),
            fig.add_subplot(gs[1, col_idx]),
            img, label
        )

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, save_name)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def main():
    print(f"\nLoading: {FILE_NAME}")
    orig         = load_gray(FILE_NAME)
    orig_resized = cv2.resize(orig, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)

    img_f    = orig_resized.astype(np.float32)
    minmax   = (img_f - img_f.min()) / (img_f.max() - img_f.min() + 1e-7)
    zscore   = np.clip((img_f - img_f.mean()) / (img_f.std() + 1e-7), -3, 3)
    clahe_img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(orig_resized)
    unsharp  = cv2.addWeighted(orig_resized, 1.5, cv2.GaussianBlur(orig_resized, (9, 9), 10), -0.5, 0)
    lap_norm = cv2.normalize(np.abs(cv2.Laplacian(orig_resized, cv2.CV_64F)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sobel_x  = cv2.Sobel(orig_resized, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y  = cv2.Sobel(orig_resized, cv2.CV_64F, 0, 1, ksize=3)

    build_figure("Blur / Denoising", [
        ("Gaussian\n(5x5)",  cv2.GaussianBlur(orig_resized, (5, 5), 0)),
        ("Median\n(k=3)",    cv2.medianBlur(orig_resized, 3)),
        ("Mean\n(5x5)",      cv2.blur(orig_resized, (5, 5))),
        ("Non-Local\nMeans", cv2.fastNlMeansDenoising(orig_resized, None, h=10, templateWindowSize=7, searchWindowSize=21)),
        ("Bilateral\n(d=9)", cv2.bilateralFilter(orig_resized, 9, 75, 75)),
    ], orig_resized, "blur_comparison.png")

    build_figure("Normalization", [
        ("Min-Max\n[0,1]",       minmax),
        ("Z-Score\n(clamped±3)", zscore),
    ], orig_resized, "normalization_comparison.png")

    build_figure("Contrast Enhancement", [
        ("Global Hist\nEqualization", cv2.equalizeHist(orig_resized)),
        ("CLAHE\n(clip=2, 8x8)",      clahe_img),
    ], orig_resized, "contrast_comparison.png")

    build_figure("Sharpening", [
        ("Unsharp\nMasking",      unsharp),
        ("Laplacian\nSharpening", cv2.addWeighted(orig_resized, 1.0, lap_norm, -0.5, 0)),
    ], orig_resized, "sharpening_comparison.png")

    build_figure("Edge Detection", [
        ("Sobel",     cv2.normalize(np.sqrt(sobel_x**2 + sobel_y**2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)),
        ("Laplacian", cv2.normalize(np.abs(cv2.Laplacian(orig_resized, cv2.CV_64F)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)),
        ("Canny",     cv2.Canny(orig_resized, 50, 150)),
    ], orig_resized, "edge_detection_comparison.png")

    build_figure("Resizing Interpolation", [
        ("Linear",  cv2.resize(orig, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)),
        ("Cubic",   cv2.resize(orig, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)),
        ("Area",    cv2.resize(orig, TARGET_SIZE, interpolation=cv2.INTER_AREA)),
        ("Nearest", cv2.resize(orig, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)),
    ], orig_resized, "resizing_comparison.png")


if __name__ == "__main__":
    main()