"""
Shared image preprocessing for both training and inference.

Keeping this in one module matters: whatever transform is applied to an
image when a PCA model is *trained* must be applied identically when a new
image is *scored*, or the projection into PCA space is meaningless.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# All images are normalized to this size before PCA. 128x128 keeps the
# flattened vector (16,384 dims for grayscale) small enough for PCA to fit
# quickly on a laptop/CPU while preserving enough structure for the model
# to pick up meaningful variation.
IMAGE_SIZE = (128, 128)


def load_and_preprocess(image_path: str | Path) -> np.ndarray:
    """
    Load an image from disk and return a flattened, normalized grayscale
    vector of shape (IMAGE_SIZE[0] * IMAGE_SIZE[1],), dtype float32, values
    in [0, 1].
    """
    img = Image.open(image_path)
    return preprocess_pil_image(img)


def preprocess_pil_image(img: Image.Image) -> np.ndarray:
    """Same as load_and_preprocess but takes an already-opened PIL Image."""
    # Grayscale: X-rays are single-channel; collapsing any stray RGB/RGBA
    # upload to grayscale keeps the feature space consistent.
    img = ImageOps.exif_transpose(img)  # respect camera/scanner orientation
    img = img.convert("L")

    # Histogram equalization evens out exposure/contrast differences between
    # scanners and phone photos of printed films, which otherwise dominate
    # the first few principal components with brightness variation rather
    # than anatomical structure.
    img = ImageOps.equalize(img)

    img = img.resize(IMAGE_SIZE, Image.LANCZOS)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.flatten()


def colorfulness(img: Image.Image) -> float:
    """
    Average per-pixel channel spread (max(R,G,B) - min(R,G,B)), scaled to
    [0, 1]. Real X-rays -- scans or phone photos of films -- are essentially
    grayscale and score near 0; ordinary color photos score far higher.
    Used as a cheap first gate against obviously-not-an-X-ray uploads.
    """
    rgb = ImageOps.exif_transpose(img).convert("RGB")
    rgb.thumbnail((256, 256))
    arr = np.asarray(rgb, dtype=np.float32)
    spread = arr.max(axis=2) - arr.min(axis=2)
    return float(spread.mean() / 255.0)


def batch_preprocess(
    image_paths: list[str | Path],
) -> tuple[np.ndarray, list[int]]:
    """
    Preprocess many images into a single (n_samples, n_features) matrix.

    Returns (matrix, valid_indices), where valid_indices are the positions
    in the *input* image_paths list that were successfully read. A caller
    that has parallel labels/paths must index them with valid_indices
    before zipping them against the returned matrix's rows -- otherwise a
    single unreadable file silently shifts every later row out of
    alignment with its label instead of just being dropped.
    """
    vectors = []
    valid_indices = []
    skipped = []
    for i, p in enumerate(image_paths):
        try:
            vectors.append(load_and_preprocess(p))
            valid_indices.append(i)
        except Exception as exc:  # corrupt file, unsupported format, etc.
            skipped.append((str(p), str(exc)))
    if skipped:
        print(f"[preprocessing] skipped {len(skipped)} unreadable file(s):")
        for path, err in skipped[:10]:
            print(f"    {path}: {err}")
    matrix = np.vstack(vectors) if vectors else np.empty((0, IMAGE_SIZE[0] * IMAGE_SIZE[1]))
    return matrix, valid_indices
